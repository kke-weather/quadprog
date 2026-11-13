# quadprog: Quadratic Programming Solver (Python)

[![.github/workflows/build-and-test.yaml](https://github.com/quadprog/quadprog/actions/workflows/build-and-test.yaml/badge.svg?branch=master)](https://github.com/quadprog/quadprog/actions/workflows/build-and-test.yaml)

```
Solve a strictly convex quadratic program

Minimize     1/2 x^T G x - a^T x
Subject to   C.T x >= b

This routine uses the the Goldfarb/Idnani dual algorithm [1].

References
---------
... [1] D. Goldfarb and A. Idnani (1983). A numerically stable dual
    method for solving strictly convex quadratic programs.
    Mathematical Programming, 27, 1-33.
```

### Installation

`pip install quadprog`

### Nonnegative weights with a constrained sum

The fork adds `solve_simplex(G, a, sum_bounds=None, overwrite=False)`. It solves
the same objective as `solve_qp` with nonnegative variables and a sum of one.
Pass `(lower, upper)` to constrain the sum to an interval instead. The existing
Goldfarb/Idnani kernel and the order of constraints are unchanged. Only the
solution is returned; constraints and temporary workspaces are created in C.

Inputs are copied by default. For caller-owned working arrays, `overwrite=True`
avoids these copies and destroys both inputs. This requires a writable native
float64 Fortran-contiguous `G` and a writable contiguous float64 vector `a`, with
no overlapping buffers. Invalid buffers raise `TypeError`, while solver failures
raise `ValueError` as in `solve_qp`. The upper triangle of G and all of a must be
finite, and sum bounds must not contain NaN. A nonfinite solution is reported as
a solver `ValueError` rather than returned silently.
Memory allocation failures propagate as `MemoryError`. Workspaces are local to
each call; the kernel runs without the GIL. Kernel sources and their compiler
flags are unchanged from upstream 0.1.14.

With `overwrite=True`, inputs may be destroyed even when solving raises
`ValueError`; a contract `TypeError` leaves them untouched. Do not retry with
the same work buffers. Consumers of this fork-only API should pin its Git
commit instead of replacing it with the upstream package.

Build and test with Python 3.11 or newer and a C compiler:

```sh
uv build --wheel
python -m pytest tests
```
### Dependencies
- Runtime
   - `numpy`
- Installation
   - `numpy`
   - C compiler if installing from sdist
   - `Cython` if building from source

### Developing

See [docs/DEVELOP.md](docs/DEVELOP.md).
