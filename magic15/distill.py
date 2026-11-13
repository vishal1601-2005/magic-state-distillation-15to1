"""15-to-1 magic-state distillation (Bravyi-Kitaev) at the protocol level.

Model (stated assumptions, see README):
  * 15 noisy T states, each carrying an independent Z error with probability eps (the usual twirled-noise model);
  * perfect Clifford operations and measurements (no faulty gates);
  * the [[15,1,3]] punctured Reed-Muller code: its four X-type stabiliser generators have the columns of the
    Hamming [15,11] parity-check matrix, so a Z-error pattern e is *accepted* iff H e = 0 (mod 2);
  * an accepted pattern is a logical error iff its Hamming weight is odd (the Z stabiliser group is the even-weight
    subcode of the Hamming code; the logical Z is the all-ones word).
"""
import numpy as np

N = 15

def hamming_checks():
    """4 x 15 binary matrix whose column j is the binary expansion of j+1."""
    return np.array([[((c >> b) & 1) for c in range(1, N + 1)] for b in range(4)], dtype=np.uint8)

def undetected_patterns():
    """All 2^15 error patterns with zero syndrome (a (2048, 15) array)."""
    allp = ((np.arange(1 << N)[:, None] >> np.arange(N)) & 1).astype(np.uint8)
    keep = ~((allp @ hamming_checks().T) % 2).any(axis=1)
    return allp[keep]

_PAT = None
def _weights():
    global _PAT
    if _PAT is None:
        P = undetected_patterns()
        _PAT = P.sum(axis=1).astype(int)
    return _PAT

def exact_round(eps):
    """(acceptance probability, output error probability given acceptance), exact for the model above."""
    w = _weights()
    p = lambda k: eps ** k * (1 - eps) ** (N - k)
    pa = sum(p(k) for k in w)
    pl = sum(p(k) for k in w if k % 2 == 1)
    return pa, pl / pa

def mc_round(eps, shots, rng):
    """Monte Carlo: returns (accepted, logical_errors) counts out of `shots` rounds."""
    e = (rng.random((shots, N)) < eps).astype(np.uint8)
    ok = ~((e @ hamming_checks().T) % 2).any(axis=1)
    bad = ok & (e.sum(axis=1) % 2 == 1)
    return int(ok.sum()), int(bad.sum())

def iterate_rounds(eps0, rounds):
    """Chain exact rounds. Returns a list of (eps_in, acceptance, eps_out, raw_T_per_output)."""
    out, eps, cost = [], eps0, 1.0
    for _ in range(rounds):
        pa, po = exact_round(eps)
        cost *= N / pa            # expected raw T states consumed per accepted output of this round
        out.append((eps, pa, po, cost))
        eps = po
    return out
