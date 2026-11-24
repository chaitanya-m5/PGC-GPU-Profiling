
import json
import sys
from pathlib import Path

def load_trace(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("traceEvents", [])

def event_name(event):
    return event.get("name", "Unknown")

def is_kernel(event):
    name = event_name(event).lower()
    cat = event.get("cat", "").lower()
    return (
        "kernel" in name
        or "gpu" in cat and ("launch" in name or "kernel" in name)
        or "cudakernel" in name
    )

def is_copy(event):
    name = event_name(event).lower()
    return any(word in name for word in [
        "memcpy", "memcopy", "memset", "copy", "transfer"
    ])

def get_intervals(events, predicate):
    intervals = []
    for e in events:
        if e.get("ph") != "X":
            continue
        if not predicate(e):
            continue
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is None or dur is None or dur <= 0:
            continue
        intervals.append((ts, ts + dur, e))
    return intervals

def union_duration(intervals):
    if not intervals:
        return 0.0

    intervals = sorted((s, e) for s, e, _ in intervals)
    total = 0.0
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end

    total += current_end - current_start
    return total

def overlap_duration(a, b):
    total = 0.0
    for s1, e1, _ in a:
        for s2, e2, _ in b:
            start = max(s1, s2)
            end = min(e1, e2)
            if end > start:
                total += end - start
    return total

def find_idle_gaps(kernel_intervals, threshold_us=50):
    if not kernel_intervals:
        return []

    intervals = sorted((s, e) for s, e, _ in kernel_intervals)
    merged = []
    start, end = intervals[0]

    for s, e in intervals[1:]:
        if s <= end:
            end = max(end, e)
        else:
            merged.append((start, end))
            start, end = s, e

    merged.append((start, end))

    gaps = []
    for i in range(len(merged) - 1):
        gap = merged[i + 1][0] - merged[i][1]
        if gap > threshold_us:
            gaps.append(gap)

    return gaps

def analyze(filename):
    events = load_trace(filename)

    kernels = get_intervals(events, is_kernel)
    copies = get_intervals(events, is_copy)

    print("\n" + "=" * 60)
    print("       GPU PROFILING TRACE ANALYSIS")
    print("=" * 60)

    if not kernels:
        print("\nNo GPU kernel events detected using the name filter.")
        print("The trace may use different event names.")
        print("Open the trace in Perfetto to inspect event names.")
        return

    # Kernel activity
    gpu_start = min(s for s, e, _ in kernels)
    gpu_end = max(e for s, e, _ in kernels)
    total_span = gpu_end - gpu_start
    kernel_busy = union_duration(kernels)
    utilization = (kernel_busy / total_span) * 100 if total_span else 0

    # Copy activity
    copy_time = union_duration(copies)
    overlap = overlap_duration(kernels, copies)

    # Idle gaps
    gaps = find_idle_gaps(kernels, threshold_us=50)

    print(f"\nKernel events detected : {len(kernels)}")
    print(f"Copy events detected  : {len(copies)}")
    print(f"GPU timeline span     : {total_span / 1000:.3f} ms")
    print(f"Kernel busy time      : {kernel_busy / 1000:.3f} ms")
    print(f"Estimated utilization  : {utilization:.2f}%")
    print(f"Copy activity time   : {copy_time / 1000:.3f} ms")
    print(f"Copy/kernel overlap  : {overlap / 1000:.3f} ms")
    print(f"Idle gaps > 50 us   : {len(gaps)}")

    if gaps:
        print(f"Total idle gap time  : {sum(gaps) / 1000:.3f} ms")
        print(f"Largest idle gap     : {max(gaps) / 1000:.3f} ms")

    print("\n--- TOP GPU KERNELS ---")
    kernel_totals = {}
    for _, _, e in kernels:
        name = event_name(e)
        kernel_totals[name] = kernel_totals.get(name, 0) + e["dur"]

    for name, duration in sorted(
        kernel_totals.items(), key=lambda x: x[1], reverse=True
    )[:10]:
        print(f"{duration / 1000:9.3f} ms | {name[:100]}")

    print("\n--- COPY OPERATIONS ---")
    for _, _, e in sorted(copies, key=lambda x: x[0])[:10]:
        print(f"{e['dur'] / 1000:8.3f} ms | {event_name(e)[:100]}")

    print("\n--- INTERPRETATION ---")
    if utilization >= 80:
        print("GPU is busy for most of the measured timeline.")
    elif utilization >= 50:
        print("GPU has moderate activity with noticeable idle periods.")
    else:
        print("GPU has substantial idle time; feeding work or transfers may be limiting.")

    if copies and overlap > 0:
        print("Copy and kernel activities overlap in the trace.")
    elif copies:
        print("Copy events exist, but little/no overlap was detected.")
    else:
        print("No copy events detected by the name filter.")

    print("\nAnalysis complete.")
    print("=" * 60)

if __name__ == "__main__":
    trace_file = sys.argv[1] if len(sys.argv) > 1 else "trace_baseline.json"

    if not Path(trace_file).exists():
        print(f"Trace file not found: {trace_file}")
        print("Usage: python analyze_trace.py trace_baseline.json")
        sys.exit(1)

    analyze(trace_file)
