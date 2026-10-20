## 1. Configuration

Everything that controls the size and scope of the experiments is in the next cell. `RUN_PROFILE` (or the environment variable `QKKT_PROFILE`) selects the problem sizes; `smoke` runs every code path in a few minutes, `small` is the default and `medium` is the profile used for the paper. `DEFAULT_DEVICE` is `"cpu"` in the CPU build and `"gpu"` in the GPU build; the environment variable `QKKT_DEVICE` overrides it. The GPU build also extends the two sparse families to KKT dimension 79 360.

The configuration and a hash of it are written to `results/config.json`; every result row carries the hash.
