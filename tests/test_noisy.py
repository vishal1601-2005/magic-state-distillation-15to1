import numpy as np
from magic15 import (exact_round, exact_round_noisy, mc_round_noisy, iterate_rounds_noisy, iterate_rounds, eps_effective)

def test_zero_clifford_noise_recovers_ideal_protocol():
    for eps in (1e-2, 1e-3):
        assert exact_round_noisy(eps, 0.0) == exact_round(eps)
    a, b = iterate_rounds_noisy(0.01, 0.0, 3), iterate_rounds(0.01, 3)
    assert all(np.allclose(x, y) for x, y in zip(a, b))

def test_error_floor_is_set_by_clifford_noise():
    p = 1e-3
    r = iterate_rounds_noisy(0.01, p, 4)
    assert r[-1][2] > p * 0.9                       # cannot go (much) below the operation noise ...
    assert r[-1][2] < p * 1.1 + 1e-9                # ... and does converge to about p
    assert r[1][2] < r[0][2]                        # but early rounds still help
    assert abs(r[-1][2] - r[-2][2]) < 1e-6 * p      # converged

def test_leading_behaviour_matches_hand_calculation():
    eps, p = 1e-3, 1e-4
    pa, po = exact_round_noisy(eps, p)
    e = eps_effective(eps, p)
    assert abs(po - (35 * e ** 3 + p)) < 0.02 * (35 * e ** 3 + p)   # po ~ 35 eps_eff^3 + p to leading order

def test_monte_carlo_matches_exact():
    rng = np.random.default_rng(5); eps, p, shots = 0.05, 0.02, 600_000
    acc, bad = mc_round_noisy(eps, p, shots, rng)
    pa, po = exact_round_noisy(eps, p)
    sa = (pa * (1 - pa) / shots) ** 0.5
    assert abs(acc / shots - pa) < 5 * sa
    pb = pa * po; sb = (pb * (1 - pb) / shots) ** 0.5
    assert abs(bad / shots - pb) < 5 * sb
