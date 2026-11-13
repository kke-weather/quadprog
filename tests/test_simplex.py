from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from quadprog import solve_qp, solve_simplex


def reference(G, a, bounds=None):
    n = len(a)
    if bounds is None:
        C = np.vstack((np.ones(n), np.eye(n))).T
        b = np.r_[1., np.zeros(n)]
    else:
        C = np.vstack((np.ones(n), -np.ones(n), np.eye(n))).T
        b = np.r_[bounds[0], -bounds[1], np.zeros(n)]
    return solve_qp(G, a, C, b, meq=1 if bounds is None else 0)[0]


@pytest.mark.parametrize('n', [1, 3, 11, 32, 300])
@pytest.mark.parametrize('bounds', [None, (.8, 1.2), (1., 1.), (1.5, 2.)])
def test_same_kernel_bits_and_input_ownership(n, bounds):
    rng = np.random.default_rng(n)
    A = rng.normal(size=(max(3, n // 2), n))
    G = 2 * A.T @ A + 1e-10 * np.eye(n)
    a = rng.normal(size=n)
    G_before, a_before = G.copy(), a.copy()
    try:
        expected = reference(G, a, bounds)
    except ValueError as error:
        for overwrite in (False, True):
            with pytest.raises(ValueError) as actual:
                solve_simplex(np.array(G, order='F'), a.copy(), bounds, overwrite=overwrite)
            assert str(actual.value) == str(error)
        return
    x = solve_simplex(G, a, bounds)
    assert np.isfinite(x).all()
    np.testing.assert_array_equal(x.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(G, G_before)
    np.testing.assert_array_equal(a, a_before)
    owned_G, owned_a = np.array(G, order='F'), a.copy()
    overwritten = solve_simplex(owned_G, owned_a, bounds, overwrite=True)
    np.testing.assert_array_equal(overwritten.view(np.uint64), expected.view(np.uint64))
    assert not np.shares_memory(overwritten, owned_G)
    assert not np.shares_memory(overwritten, owned_a)


@pytest.mark.parametrize('kind', ['order', 'dtype', 'shape', 'readonly', 'overlap', 'empty', 'unaligned'])
def test_overwrite_contract_errors_are_not_solver_errors(kind):
    G, a = np.eye(3, order='F'), np.ones(3)
    if kind == 'order':
        G = np.eye(3, order='C')
    elif kind == 'dtype':
        a = a.astype(np.float32)
    elif kind == 'shape':
        a = np.ones((3, 1))
    elif kind == 'readonly':
        G.flags.writeable = False
    elif kind == 'overlap':
        a = G[:, 0]
    elif kind == 'empty':
        G, a = np.empty((0, 0)), np.empty(0)
    elif kind == 'unaligned':
        G = np.ndarray((3, 3), buffer=bytearray(73), offset=1, dtype=np.float64, order='F')
    with pytest.raises(TypeError):
        solve_simplex(G, a, overwrite=True)


def test_solver_errors():
    with pytest.raises(ValueError, match='constraints are inconsistent'):
        solve_simplex(np.eye(3), np.ones(3), (1.2, .8))
    with pytest.raises(ValueError, match='not positive definite'):
        solve_simplex(np.zeros((3, 3)), np.ones(3))


@pytest.mark.parametrize('bounds', [(1.,), (1., 2., 3.), 'ab', 1.])
def test_bounds_contract(bounds):
    with pytest.raises(TypeError):
        solve_simplex(np.eye(3), np.ones(3), bounds)


def test_nonfinite_and_conversion_contracts():
    for value in (np.nan, np.inf, -np.inf):
        G = np.eye(3)
        G[0, 0] = value
        with pytest.raises(TypeError):
            solve_simplex(G, np.ones(3))
        with pytest.raises(TypeError):
            solve_simplex(np.eye(3), [1., value, 1.])
    with pytest.raises(TypeError):
        solve_simplex(np.eye(3), np.ones(3), (0., np.nan))
    with pytest.raises(TypeError):
        solve_simplex([[1., 2.], [3.]], [1., 1.])
    unaligned = memoryview(bytearray(25))[1:].cast('d')
    with pytest.raises(TypeError):
        solve_simplex(np.eye(3, order='F'), unaligned, overwrite=True)


def test_lower_triangle_is_ignored():
    G = np.array([[2., .2], [np.nan, 1.]])
    expected = reference(G, np.array([.3, .8]))
    actual = solve_simplex(G, np.array([.3, .8]))
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_kernel_integer_overflow_is_rejected_before_reading_data():
    G = np.lib.stride_tricks.as_strided(np.ones(1), shape=(50000, 50000), strides=(8, 400000))
    with pytest.raises(TypeError, match='kernel integer bounds'):
        solve_simplex(G, np.zeros(50000), overwrite=True)


def test_type_error_does_not_mutate_inputs():
    G, a = np.eye(3, order='C'), np.arange(3, dtype=float)
    before_G, before_a = G.copy(), a.copy()
    with pytest.raises(TypeError):
        solve_simplex(G, a, overwrite=True)
    np.testing.assert_array_equal(G, before_G)
    np.testing.assert_array_equal(a, before_a)


@pytest.mark.parametrize('a', [['1', '2'], np.ones(2, dtype=complex)])
def test_default_rejects_nonreal_dtypes(a):
    with pytest.raises(TypeError):
        solve_simplex(np.eye(2), a)


def test_bounds_overflow_is_a_type_error():
    with pytest.raises(TypeError):
        solve_simplex(np.eye(2), np.ones(2), (0, 10**400))


def test_calls_do_not_share_workspace():
    G = np.array([[2., .2], [.2, 1.]])
    queries = [np.array([i / 10, .5]) for i in range(40)]
    expected = [reference(G, a) for a in queries]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda a: solve_simplex(G, a), queries))
    for x, y in zip(actual, expected):
        np.testing.assert_array_equal(x.view(np.uint64), y.view(np.uint64))
