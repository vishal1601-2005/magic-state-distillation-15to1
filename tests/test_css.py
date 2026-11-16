import numpy as np
import pytest
from magic15 import (CSSProtocol, is_triorthogonal, bravyi_kitaev_15to1, triorthogonal_14to2, cube_8T_to_CCZ, all_protocols,
                     rank_gf2, exact_round, iterate_rounds)

def test_general_framework_reproduces_the_original_15to1_code():
    bk = bravyi_kitaev_15to1()
    for eps in (0.2, 1e-2, 1e-3):
        pa, po = bk.exact(eps); pa0, po0 = exact_round(eps)
        assert abs(pa - pa0) < 1e-12 and abs(po - po0) < 1e-12 * max(1, po0 / 1e-12)
    a, b = bk.iterate(0.01, 3), iterate_rounds(0.01, 3)
    assert all(np.allclose(x, y, rtol=1e-9) for x, y in zip(a, b))
    assert bk.leading_order() == (3, 35)

def test_cube_code_gives_28_eps_squared_and_first_order_acceptance_8_eps():
    cube = cube_8T_to_CCZ()
    assert cube.leading_order() == (2, 28)
    for eps in (1e-3, 1e-4):
        pa, po = cube.exact(eps)
        assert abs(po / (28 * eps ** 2) - 1) < 0.05 and abs(pa - (1 - 8 * eps)) < 100 * eps ** 2
    assert rank_gf2(cube.K) == 4                         # one X stabiliser plus three logical X operators

def test_triorthogonal_14to2_has_coefficient_7():
    p = triorthogonal_14to2()
    assert is_triorthogonal(np.array(p.K)) and p.n == 14
    assert p.leading_order() == (2, 7)                   # equals (1 + 3k) for k = 2
    assert rank_gf2(p.Hx) == 3 and rank_gf2(p.K) == 5     # 3 X stabilisers + 2 logical X

def test_invalid_protocol_is_refused_negative_control():
    H = np.array([[((c >> b) & 1) for c in range(1, 8)] for b in range(3)], dtype=np.uint8)      # Steane code
    G = np.vstack([np.ones((1, 7), np.uint8), H])
    assert not is_triorthogonal(G)
    with pytest.raises(ValueError): CSSProtocol.from_triorthogonal("Steane", G)

def test_enumerator_totals_are_consistent_for_every_protocol():
    for p in all_protocols():
        A, L = p.weight_enumerators()
        assert A.sum() == 2 ** (p.n - rank_gf2(p.Hx))     # accepted patterns form the null space of Hx
        assert (L <= A).all() and A[0] == 1 and L[0] == 0

def test_heterogeneous_noise_reduces_to_uniform_noise():
    for p in all_protocols():
        a = p.exact(0.03); b = p.exact_heterogeneous(np.full(p.n, 0.03))
        assert np.allclose(a, b, rtol=1e-9)

def test_monte_carlo_matches_exact_for_each_protocol():
    rng = np.random.default_rng(3); eps, shots = 0.05, 400_000
    for p in all_protocols():
        acc, bad = p.mc(eps, shots, rng); pa, po = p.exact(eps)
        sa = (pa * (1 - pa) / shots) ** 0.5; assert abs(acc / shots - pa) < 5 * sa
        pb = pa * po; sb = (pb * (1 - pb) / shots) ** 0.5; assert abs(bad / shots - pb) < 5 * sb

def test_single_round_threshold_of_15to1_is_about_0_141():
    bk = bravyi_kitaev_15to1()
    lo, hi = 0.05, 0.3
    for _ in range(60):
        mid = (lo + hi) / 2
        if bk.exact(mid)[1] < mid: lo = mid
        else: hi = mid
    assert abs(lo - 0.1414) < 0.001
