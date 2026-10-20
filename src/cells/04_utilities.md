## 4. Utilities

Seeded random streams (`rng_for` hashes the global seed with the instance keys, so every instance is reproducible on its own), timers, conversions between CVXOPT and NumPy, and the result tables. Rows are flushed to `results/partial/` every 20 s, so a session that stops early still leaves usable CSV files. `PeakMemory` samples the process RSS (and the CuPy pool) during each `coneqp` call.
