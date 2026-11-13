import numpy as np
from magic15 import undetected_patterns, exact_round, mc_round, iterate_rounds

def test_pattern_counts():
    P = undetected_patterns(); w = P.sum(axis=1)
    assert len(P) == 2048
    assert (w == 3).sum() == 35 and (w == 0).sum() == 1 and (w < 3).sum() == 1   # minimum distance 3, 35 weight-3 words

def test_leading_order():
    for eps in (1e-3, 1e-4):
        pa, po = exact_round(eps)
        assert abs(po / (35 * eps ** 3) - 1) < 0.05
        assert abs(pa - (1 - 15 * eps)) < 200 * eps ** 2

def test_monte_carlo_matches_exact():
    rng = np.random.default_rng(1); eps = 0.05; shots = 400_000
    acc, bad = mc_round(eps, shots, rng)
    pa, po = exact_round(eps)
    sig_a = (pa * (1 - pa) / shots) ** 0.5
    assert abs(acc / shots - pa) < 5 * sig_a
    p_bad = pa * po                                  # P(accepted and logical error)
    sig_b = (p_bad * (1 - p_bad) / shots) ** 0.5
    assert abs(bad / shots - p_bad) < 5 * sig_b

def test_rounds_improve_and_cost_grows():
    r = iterate_rounds(0.01, 3)
    assert r[0][2] < r[0][0] and r[1][2] < r[0][2] and r[2][2] < r[1][2]
    assert r[0][3] < r[1][3] < r[2][3]
