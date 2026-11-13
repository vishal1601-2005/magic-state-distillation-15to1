"""Toy noise-floor extension: distillation with imperfect Clifford operations (phenomenological model, NOT circuit-level).

Model, in addition to the ideal protocol in distill.py:
  * encoding noise: each of the 15 input qubits suffers an extra independent Z error with probability p (so the effective input error is
    eps_eff = eps (1 - p) + p (1 - eps));
  * output noise: the surviving output qubit suffers one more Z error with probability p after decoding.
This is deliberately simple. It shows the qualitative fact that distillation cannot push the error below the noise of the operations
that implement it, and it gives that floor. It does not model where faults occur in a real circuit, correlated faults, or measurement
errors; a circuit-level treatment (for example with Stim) would, and is future work.
"""
import numpy as np
from .distill import exact_round, N, hamming_checks

def _xor(a, b):
    """Probability that exactly one of two independent events with probabilities a and b occurs."""
    return a * (1 - b) + b * (1 - a)

def eps_effective(eps, p):
    return _xor(eps, p)

def exact_round_noisy(eps, p):
    """(acceptance probability, output error given acceptance) for one noisy round."""
    pa, po = exact_round(eps_effective(eps, p))
    return pa, _xor(po, p)

def mc_round_noisy(eps, p, shots, rng):
    """Monte Carlo twin of exact_round_noisy: returns (accepted, logical_errors)."""
    e = (rng.random((shots, N)) < eps_effective(eps, p)).astype(np.uint8)
    ok = ~((e @ hamming_checks().T) % 2).any(axis=1)
    logical = (e.sum(axis=1) % 2 == 1) ^ (rng.random(shots) < p)
    return int(ok.sum()), int((ok & logical).sum())

def iterate_rounds_noisy(eps0, p, rounds):
    """Chain exact noisy rounds. Returns a list of (eps_in, acceptance, eps_out, raw_T_per_output)."""
    out, eps, cost = [], eps0, 1.0
    for _ in range(rounds):
        pa, po = exact_round_noisy(eps, p)
        cost *= N / pa
        out.append((eps, pa, po, cost))
        eps = po
    return out
