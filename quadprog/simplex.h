#ifndef QUADPROG_SIMPLEX_H
#define QUADPROG_SIMPLEX_H
#include <Python.h>

int qp_solve_simplex(PyObject *G, PyObject *a, PyObject *x, double lower, double upper, int bounded);

#endif
