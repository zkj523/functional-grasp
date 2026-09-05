# Per-trial archiving

Each trial is frozen into a record that can be read without access to the machine
that produced it.

```
real_grasp_<timestamp>_<object>_<position>_<trial>_<result>/
├── manifest.yaml        object, pose, method, checkpoint paths, result, timestamps
├── quality_gate.json    every gate metric with its threshold
├── registration/        estimated pose, residual distances, resampled point cloud
├── capture/             the RGB-D frame the decision was made from
├── evidence/            post-execution images
├── logs/                full terminal output
├── code_snapshot/       the exact scripts that ran
└── CHECKSUMS.sha256     integrity of all of the above
```

Three choices worth noting.

**The code is snapshotted, not referenced.** A commit hash is not enough when the
working tree had uncommitted changes, which during active research it usually does.

**The result label records who decided it.** `result.user_declared` distinguishes a
human judgement from an automatic one, so the two are never silently mixed.

**Checksums cover the whole record.** If an archived run is later edited — even
accidentally — the mismatch is detectable.
