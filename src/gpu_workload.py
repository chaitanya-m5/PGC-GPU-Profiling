
import argparse
import math
import time
import torch


# ------------------------------------------------------------
# EXPERIMENT 7: GPU PROFILING AND UTILIZATION
# ------------------------------------------------------------

def make_weights(n):
    generator = torch.Generator().manual_seed(1)
    weights = torch.randn(n, n, generator=generator) / math.sqrt(n)
    return weights.cuda()


def make_host_batches(rows, cols, pinned, unique=4):
    generator = torch.Generator().manual_seed(0)
    batches = []

    for _ in range(unique):
        batch = torch.randn(rows, cols, generator=generator)

        if pinned:
            batch = batch.pin_memory()

        batches.append(batch)

    return batches


def compute(x, w, reps):
    y = x

    for _ in range(reps):
        y = y @ w
        y = torch.relu(y)
        y = y.mul(0.5).add(1.0)

    return y.sum(dim=1)


# ------------------------------------------------------------
# BASELINE: PAGEABLE MEMORY, DEFAULT STREAM
# ------------------------------------------------------------

def run_baseline(host, w, reps, nbatches):
    results = []

    for i in range(nbatches):
        x = host[i % len(host)].cuda()
        result = compute(x, w, reps)
        results.append(result.cpu())

    torch.cuda.synchronize()
    return torch.stack(results)


# ------------------------------------------------------------
# OPTIMIZED: PINNED MEMORY, TWO STREAMS, DOUBLE BUFFERING
# ------------------------------------------------------------

def run_optimized(host, w, reps, nbatches):
    rows, cols = host[0].shape

    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    buffers = [
        torch.empty(rows, cols, device="cuda"),
        torch.empty(rows, cols, device="cuda")
    ]

    copy_done = [torch.cuda.Event() for _ in range(2)]
    compute_done = [torch.cuda.Event() for _ in range(2)]

    host_output = torch.empty(nbatches, rows).pin_memory()

    torch.cuda.synchronize()

    # Initially, both GPU buffers are free.
    for event in compute_done:
        event.record(compute_stream)

    def issue_copy(batch_index):
        buffer_index = batch_index % 2

        # Do not overwrite a buffer before its computation is complete.
        copy_stream.wait_event(compute_done[buffer_index])

        with torch.cuda.stream(copy_stream):
            buffers[buffer_index].copy_(
                host[batch_index % len(host)],
                non_blocking=True
            )
            copy_done[buffer_index].record(copy_stream)

    # Preload the first batch.
    issue_copy(0)

    for i in range(nbatches):
        buffer_index = i % 2

        # Start copying the next batch while the current batch
        # can be processed.
        if i + 1 < nbatches:
            issue_copy(i + 1)

        # Wait until this buffer's data has reached the GPU.
        compute_stream.wait_event(copy_done[buffer_index])

        with torch.cuda.stream(compute_stream):
            result = compute(
                buffers[buffer_index],
                w,
                reps
            )

            # Mark buffer free after its computation finishes.
            compute_done[buffer_index].record(compute_stream)

            # Store the output into pinned host memory.
            host_output[i].copy_(result, non_blocking=True)

    torch.cuda.synchronize()
    return host_output


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 7: GPU Profiling and Utilization"
    )

    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized"],
        required=True
    )

    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-batches", type=int, default=10)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available.")

    pinned = args.mode == "optimized"
    workload = run_optimized if pinned else run_baseline

    host = make_host_batches(
        args.rows,
        args.cols,
        pinned
    )

    weights = make_weights(args.cols)

    print("=" * 55)
    print("EXPERIMENT 7: PROFILING AND GPU UTILIZATION")
    print("=" * 55)

    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"Mode      : {args.mode}")
    print(f"Batch size: {args.rows * args.cols * 4 / 1e6:.0f} MB")

    print("\nWarming up GPU...")
    workload(host, weights, args.reps, 3)
    torch.cuda.synchronize()

    print("Running experiment...")
    start_time = time.perf_counter()

    results = workload(
        host,
        weights,
        args.reps,
        args.batches
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    print("\n--------------- RESULTS ----------------")
    print(f"Batches    : {args.batches}")
    print(f"Total time : {elapsed:.3f} s")
    print(f"Per batch : {elapsed / args.batches * 1000:.2f} ms")
    print(f"Throughput : {args.batches / elapsed:.1f} batches/s")
    print(f"Checksum   : {results.double().sum().item():.6e}")

    # --------------------------------------------------------
    # CPU + CUDA PROFILING
    # --------------------------------------------------------

    if args.profile:
        from torch.profiler import profile, ProfilerActivity

        print("\nProfiling CPU and CUDA activity...")

        torch.cuda.synchronize()

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA
            ],
            record_shapes=True,
            with_stack=False
        ) as profiler:

            workload(
                host,
                weights,
                args.reps,
                args.profile_batches
            )

            torch.cuda.synchronize()

        trace_filename = f"trace_{args.mode}.json"
        profiler.export_chrome_trace(trace_filename)

        print(f"Trace saved: {trace_filename}")

        print("\n--- CUDA OPERATOR SUMMARY ---")

        print(
            profiler.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=15
            )
        )


if __name__ == "__main__":
    main()
