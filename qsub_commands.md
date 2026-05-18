# Submitting jobs

All jobs go through the single templated script `submit_RP.sh`, fired via the `./launch.sh` helper.

## One-time setup (run once on the main node before the first batched re-run)

```bash
python3 split_batch_size_cache.py   # fans the combined cache into per-combo files
```

## Single job

```bash
./launch.sh DATASET MODEL NODE GPU      # e.g. ./launch.sh AMLSim GCN 55 0
./launch.sh DATASET MODEL NODE:GPU      # compact form, e.g. ./launch.sh AMLSim GCN 55:0
```

`NODE` is the numeric suffix only; `submit_RP.sh` maps it to `host=comp0${NODE}`. `GPU` is the device index pinned via `CUDA_VISIBLE_DEVICES`.

## Batch from manifest

Edit `jobs.txt` (one `DATASET MODEL NODE:GPU` per line, `#` comments allowed) and run:

```bash
./launch.sh -f jobs.txt
```

Check queued jobs with:

```bash
qstat -u $USER
```

## How parallel-safe naming works

- **Optuna DBs**: `optimization_results_on_{dataset}_{model}.db` — one per combo.
- **Result pkls**: `{model}_..._{JOB_ID}.pkl` — `JOB_ID` is derived from `PBS_JOBID` in `submit_RP.sh`, so every job writes uniquely named fragments. No rsync collision.
- **Batch size cache**: `batch_size_cache_{dataset}_{model}.json` — one per combo, mirrors the Optuna DB pattern.

Concurrent jobs on different (dataset, model) combos can rsync back simultaneously without overwriting each other.
