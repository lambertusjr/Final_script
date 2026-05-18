"""One-time migration: fan out the combined batch_size_cache.json into
per-(dataset, model) files matching the new naming convention in utilities.py.

Run once on the main node before the first batched re-run. Safe to re-run;
existing per-combo files are merged with the legacy combined cache (legacy
values win only for keys not already present, so newer per-combo edits are
preserved).
"""
import json
import os
import shutil
import sys
from collections import defaultdict

LEGACY = "batch_size_cache.json"
BACKUP = "batch_size_cache.json.bak"


def parse_key(key):
    """key format: '{dataset}_{model}_{phase}', e.g. 'IBM_AML_HiSmall_GIN_tuning'.
    Both phase and model are single tokens (no underscores); dataset may contain underscores.
    """
    parts = key.rsplit("_", 2)
    if len(parts) != 3:
        return None
    dataset, model, phase = parts
    return dataset, model, phase


def main():
    if not os.path.exists(LEGACY):
        print(f"No {LEGACY} found. Nothing to migrate.")
        return 0

    with open(LEGACY) as f:
        legacy = json.load(f)

    grouped = defaultdict(dict)
    skipped = []
    for key, value in legacy.items():
        parsed = parse_key(key)
        if parsed is None:
            skipped.append(key)
            continue
        dataset, model, phase = parsed
        grouped[(dataset, model)][key] = value

    if skipped:
        print(f"Warning: {len(skipped)} key(s) couldn't be parsed and were skipped:")
        for k in skipped:
            print(f"  - {k}")

    written = 0
    for (dataset, model), entries in sorted(grouped.items()):
        out = f"batch_size_cache_{dataset}_{model}.json"
        existing = {}
        if os.path.exists(out):
            try:
                with open(out) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: existing {out} unreadable, will overwrite")
        # Merge: new per-combo file's existing entries win over legacy.
        merged = {**entries, **existing}
        with open(out, "w") as f:
            json.dump(merged, f, indent=2)
        written += 1
        print(f"  wrote {out} ({len(merged)} entries)")

    shutil.move(LEGACY, BACKUP)
    print(f"\nDone. {written} per-combo cache file(s) written.")
    print(f"Legacy {LEGACY} renamed to {BACKUP}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
