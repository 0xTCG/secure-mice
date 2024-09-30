# Secure Multiple Imputation with Chained Equations (Secure MICE)

This repository contains a source code for Secure MICE algorithm implemented in Sequre. While this codebase can be used as a standalone, Secure MICE is actually implemented as an integral part of [Sequre framework](https://github.com/0xTCG/sequre/tree/mice).

## Run Secure-MICE benchmarks

To run Secure-MICE benchmarks, first run all cells in [Jupyter notebook](applications/offline/mi.ipynb) to generate data and obtain all non-secure, baseline results.

Then run all other benchmarks via docker/podman:

```bash
sudo podman run --mount type=bind,source=$(pwd)/data,destination=/sequre/data --mount type=bind,source=$(pwd)/applications/config,destination=/sequre/applications/config --security-opt label=disable -e "CODON_DEBUG=lt" --privileged --rm -t hsmile/secure-mice:latest scripts/run.sh -release benchmarks --mi --jit --local
```

Use [configuration module](applications/config/mi.toml) to configure different benchmark scenarios.
