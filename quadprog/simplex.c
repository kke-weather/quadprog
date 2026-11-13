/* Fixed nonnegative/sum constraints, using the unchanged qpgen2_ kernel. */
#define PY_SSIZE_T_CLEAN
#include "simplex.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <math.h>

#ifdef _MSC_VER
#define DOUBLE_ALIGNMENT __alignof(double)
#else
#define DOUBLE_ALIGNMENT _Alignof(double)
#endif

int qpgen2_(double *, double *, int, double *, double *, double *, double *, double *, int, int,
            int *, int *, int *, double *, int);

static int valid_buffer(Py_buffer *v, int ndim, char order) {
    return v->ndim == ndim && v->itemsize == sizeof(double) && v->format &&
           strcmp(v->format, "d") == 0 && PyBuffer_IsContiguous(v, order) &&
           (uintptr_t)v->buf % DOUBLE_ALIGNMENT == 0;
}

static int overlap(Py_buffer *a, Py_buffer *b) {
    uintptr_t p = (uintptr_t)a->buf, q = (uintptr_t)b->buf;
    return p < q ? q-p < (size_t)a->len : p-q < (size_t)b->len;
}

int qp_solve_simplex(PyObject *go, PyObject *ao, PyObject *xo, double lo, double hi, int bounded) {
    Py_buffer g = {0}, a = {0}, x = {0};
    int status = -1;
    double *block = NULL;
    int *iblock = NULL;
    int flags = PyBUF_STRIDES | PyBUF_FORMAT | PyBUF_WRITABLE;
    if (PyObject_GetBuffer(go, &g, flags) || PyObject_GetBuffer(ao, &a, flags) || PyObject_GetBuffer(xo, &x, flags)) {
        if (PyErr_ExceptionMatches(PyExc_ValueError) || PyErr_ExceptionMatches(PyExc_BufferError)) {
            PyErr_SetString(PyExc_TypeError, "QP buffers must be writable native float64 arrays");
        }
        goto finish;
    }
    if (!valid_buffer(&g, 2, 'F') || !valid_buffer(&a, 1, 'C') || !valid_buffer(&x, 1, 'C') ||
        g.shape[0] < 1 || g.shape[0] > INT_MAX-5 || g.shape[0] != g.shape[1] ||
        g.shape[0] != a.shape[0] || g.shape[0] != x.shape[0] || (bounded != 0 && bounded != 1)) {
        PyErr_SetString(PyExc_TypeError, "invalid owned QP buffers or dimension");
        goto finish;
    }
    int n = (int)g.shape[0], q = n + (bounded ? 2 : 1), r = n;
    if (n > INT_MAX/(n+5)) {
        PyErr_SetString(PyExc_TypeError, "QP dimension exceeds kernel integer bounds");
        goto finish;
    }
    size_t nc = (size_t)n * q, nr = (size_t)r*(r+5), nw = 2*(size_t)n + 2*(size_t)q + nr/2;
    if (nc > INT_MAX || nr > INT_MAX || nw > INT_MAX || nc > SIZE_MAX-(2*(size_t)q+nw+8)) {
        PyErr_SetString(PyExc_TypeError, "QP dimension exceeds kernel integer bounds");
        goto finish;
    }
    size_t nd = nc + 2*(size_t)q + nw + 8;
    if (nd > SIZE_MAX/sizeof(double) || g.len != (size_t)n*n*sizeof(double) ||
        a.len != (size_t)n*sizeof(double) || x.len != (size_t)n*sizeof(double) ||
        overlap(&g, &a) || overlap(&g, &x) || overlap(&a, &x)) {
        PyErr_SetString(PyExc_TypeError, "QP buffer sizes or overlap are invalid");
        goto finish;
    }
    /* Match solve_qp: only the upper triangle of G is significant. */
    for (int j = 0; j < n; ++j) {
        if (!isfinite(((double *)a.buf)[j])) {
            PyErr_SetString(PyExc_TypeError, "G and a must contain finite values");
            goto finish;
        }
        for (int i = 0; i <= j; ++i) {
            if (!isfinite(((double *)g.buf)[i + (size_t)j*n])) {
                PyErr_SetString(PyExc_TypeError, "G and a must contain finite values");
                goto finish;
            }
        }
    }
    block = calloc(nd, sizeof(double));
    iblock = calloc((size_t)q + 2, sizeof(int));
    if (!block || !iblock) {
        PyErr_NoMemory();
        goto finish;
    }
    double *C = block + 1, *d = C + nc + 2, *lagr = d + q + 2, *work = lagr + q + 2;
    int *iact = iblock + 1, nact = 0, iters[2] = {0, 0};
    const double guard = 1234567.89012345;
    C[-1] = C[nc] = d[-1] = d[q] = lagr[-1] = lagr[q] = work[-1] = work[nw] = guard;
    iblock[0] = iblock[q+1] = 0x5a17c0de;
    for (int i = 0; i < n; ++i) {
        C[i] = 1.0;
        if (bounded) C[n+i] = -1.0;
        C[(size_t)(i + (bounded ? 2 : 1))*n+i] = 1.0;
    }
    d[0] = bounded ? lo : 1.0;
    if (bounded) d[1] = -hi;
    double objective = 0.0;
    Py_BEGIN_ALLOW_THREADS
    status = qpgen2_(g.buf, a.buf, n, x.buf, lagr, &objective, C, d, q, bounded ? 0 : 1,
                     iact, &nact, iters, work, 0);
    Py_END_ALLOW_THREADS
    if (status == 0) {
        for (int i = 0; i < n; ++i) {
            if (!isfinite(((double *)x.buf)[i])) {
                PyErr_SetString(PyExc_ValueError, "nonfinite QP solution");
                status = -1;
                break;
            }
        }
    }
    if (C[-1] != guard || C[nc] != guard || d[-1] != guard || d[q] != guard ||
        lagr[-1] != guard || lagr[q] != guard || work[-1] != guard || work[nw] != guard ||
        iblock[0] != 0x5a17c0de || iblock[q+1] != 0x5a17c0de) {
        PyErr_SetString(PyExc_RuntimeError, "QP workspace guard mismatch");
        status = -1;
    }
finish:
    free(block);
    free(iblock);
    if (g.obj) PyBuffer_Release(&g);
    if (a.obj) PyBuffer_Release(&a);
    if (x.obj) PyBuffer_Release(&x);
    return status;
}
