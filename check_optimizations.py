"""
Diagnostic for the four sampling-speed levers.
Run from the project root: `python3 check_optimizations.py`
"""
import os
import sys
import subprocess
import torch

OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def header(s):
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")


# ---------- 0. Git state ----------
header("0. Git state (which commit are you actually on?)")
try:
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "log", "-1", "--pretty=%h %s"], text=True
    ).strip()
    print(f"Branch: {branch}")
    print(f"HEAD:   {commit}")
    if "Speed up NeighborLoader" in commit:
        print(f"{OK} HEAD is the optimization commit.")
    else:
        print(f"{WARN} HEAD doesn't look like the optimization commit. "
              f"Expected message containing 'Speed up NeighborLoader'.")
        print("       Run: git log --oneline -5")
except Exception as e:
    print(f"{FAIL} git check failed: {e}")


# ---------- 1. Source-level checks ----------
header("1. Source-level checks (are the edits actually in the files?)")

def grep_file(path, needle):
    try:
        with open(path) as f:
            return needle in f.read()
    except Exception:
        return False

checks = [
    ("pre_processing.py",  "from torch_geometric.utils import sort_edge_index", "Lever 1 import"),
    ("pre_processing.py",  "sort_edge_index(",                                   "Lever 1 call"),
    ("funcs_for_optuna.py", "num_neighbors = [10, 5]",                           "Lever 3a in funcs_for_optuna"),
    ("testing.py",         "num_neighbors = [10, 5]",                            "Lever 3a in testing"),
    ("helper_functions.py", "num_neighbors=[10, 5]",                             "Lever 3a default in find_optimal_batch_size"),
    ("helper_functions.py", "if avail_gb > 24:",                                 "Lever 2 adaptive workers"),
    ("helper_functions.py", "**neighbor_loader_kwargs()\n            )",         "Lever 4 probe kwargs"),
]
for path, needle, label in checks:
    ok = grep_file(path, needle)
    print(f"{OK if ok else FAIL} {label:50s} ({path})")


# ---------- 2. neighbor_loader_kwargs runtime check ----------
header("2. neighbor_loader_kwargs() — what does it ACTUALLY return on this machine?")
sys.path.insert(0, os.getcwd())
try:
    from helper_functions import neighbor_loader_kwargs
    from utilities import check_ram_usage, _read_cgroup_memory
    cg = _read_cgroup_memory()
    used_pct, avail_gb = check_ram_usage()
    print(f"Platform:         {sys.platform}")
    print(f"CPU count:        {os.cpu_count()}")
    print(f"cgroup mem limit: {cg}  (None means no cgroup -> workstation path)")
    print(f"RAM available:    {avail_gb:.1f} GB  (used: {used_pct:.1f}%)")
    kw = neighbor_loader_kwargs()
    print(f"\nneighbor_loader_kwargs() returned: {kw}")
    nw = kw.get("num_workers", 0)
    if sys.platform == "win32":
        expected = 0
        note = "Windows -> 0 workers (spawn is fragile)"
    elif cg is not None:
        expected = min(2, os.cpu_count() or 1)
        note = "cgroup detected -> capped at 2"
    elif avail_gb > 24:
        expected = min(6, os.cpu_count() or 1)
        note = ">24 GB free -> 6 workers"
    elif avail_gb > 12:
        expected = min(4, os.cpu_count() or 1)
        note = ">12 GB free -> 4 workers"
    else:
        expected = min(2, os.cpu_count() or 1)
        note = "<=12 GB free -> 2 workers"
    print(f"Expected:         num_workers={expected}  ({note})")
    print(f"{OK if nw == expected else FAIL} num_workers matches expectation")
    if nw == 0 and sys.platform == "win32":
        print(f"{WARN} Lever 2 has NO effect on Windows — sampling stays single-process.")
except Exception as e:
    print(f"{FAIL} Could not import / run: {e}")


# ---------- 3. Dataset cache check (most important) ----------
header("3. Dataset cache check — is edge_index actually sorted?")
print("Lever 1 only takes effect AFTER the .pt cache is rebuilt.")
print("If the cache is stale, none of the sampling gains from Lever 1 apply.\n")

candidates = []
for root, _dirs, files in os.walk("Datasets"):
    for f in files:
        if f.endswith(".pt") and "processed" in root:
            candidates.append(os.path.join(root, f))

if not candidates:
    print(f"{WARN} No .pt files found under Datasets/. "
          f"Run python3 pre_process_datasets.py first.")
else:
    print(f"Found {len(candidates)} processed dataset file(s):")
    for path in candidates:
        mtime = os.path.getmtime(path)
        import datetime as _dt
        mtime_str = _dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        size_mb = os.path.getsize(path) / 1e6
        print(f"\n  -> {path}")
        print(f"     modified: {mtime_str}   size: {size_mb:.1f} MB")
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            # PyG InMemoryDataset stores (data, slices) tuple
            data = blob[0] if isinstance(blob, tuple) else blob
            ei = getattr(data, "edge_index", None)
            if ei is None:
                print(f"     {WARN} no edge_index attribute")
                continue
            row = ei[0]
            sorted_by_row = bool(torch.all(row[1:] >= row[:-1]).item())
            print(f"     edge_index shape: {tuple(ei.shape)}")
            print(f"     sorted by row:    {sorted_by_row}")
            if sorted_by_row:
                print(f"     {OK} Lever 1 applied — cache was rebuilt after sort.")
            else:
                print(f"     {FAIL} edge_index NOT sorted — cache is STALE.")
                print(f"            Fix: rm {path}   then rerun pre_process_datasets.py")
        except Exception as e:
            print(f"     {FAIL} could not inspect: {e}")


# ---------- 4. pyg-lib detection ----------
header("4. pyg-lib (the C++ sampler — big multiplier on top of Lever 1)")
try:
    import pyg_lib
    print(f"{OK} pyg-lib {pyg_lib.__version__} installed — CSC fast-path active.")
except ImportError:
    print(f"{WARN} pyg-lib NOT installed.")
    print("       Even with sorted edges, sampling uses the slow Python fallback.")
    print("       Expected sampling speedup once installed: 2-5x.")
try:
    import torch_sparse
    print(f"{OK} torch-sparse {torch_sparse.__version__} installed.")
except ImportError:
    print(f"{WARN} torch-sparse not installed (helps with SparseTensor fast-path).")


# ---------- 5. Quick wall-clock sanity ----------
header("5. NeighborLoader microbenchmark")
print("Building a tiny loader and timing 50 batches to see workers in action.")
print("If CPU sampling is the bottleneck, more workers should reduce per-batch time.\n")
try:
    from torch_geometric.loader import NeighborLoader
    import time
    found = None
    for path in candidates:
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            data = blob[0] if isinstance(blob, tuple) else blob
            if hasattr(data, "edge_index") and hasattr(data, "x"):
                found = (path, data)
                break
        except Exception:
            continue
    if found is None:
        print(f"{WARN} skipped — no usable dataset cache.")
    else:
        path, data = found
        print(f"Using: {path}  ({data.num_nodes} nodes, {data.num_edges} edges)")
        mask = getattr(data, "train_mask", None)
        if mask is None or mask.dim() > 1:
            mask = torch.arange(min(10_000, data.num_nodes))
        for nw in (0, kw.get("num_workers", 2)):
            loader = NeighborLoader(
                data, num_neighbors=[10, 5], batch_size=512,
                input_nodes=mask, shuffle=True,
                num_workers=nw,
                persistent_workers=(nw > 0),
            )
            it = iter(loader)
            next(it)  # warmup
            t0 = time.perf_counter()
            n = 0
            for _ in range(50):
                try:
                    next(it)
                    n += 1
                except StopIteration:
                    break
            dt = time.perf_counter() - t0
            print(f"  num_workers={nw}: {n} batches in {dt:.2f}s "
                  f"({1000 * dt / max(n,1):.1f} ms/batch)")
            del loader, it
except Exception as e:
    print(f"{WARN} microbenchmark skipped: {e}")

print("\nDone.")
