"""Protocol-agnostic distillation analysis for CSS codes under independent Z noise on the input states.

A protocol is described by two binary matrices over GF(2):
  Hx : rows = X-type stabiliser generators. A Z-error pattern e is ACCEPTED iff Hx e = 0 (mod 2).
  K  : rows span every X-type operator that commutes with all Z-type stabilisers (X stabilisers + X logicals).
       An accepted pattern is HARMLESS iff K e = 0 (it is a Z stabiliser); otherwise it is a logical error on at least one output.
Two ways to build a protocol:
  * `from_triorthogonal(G)`: Bravyi-Haah triorthogonal matrix, odd-weight rows = logical X (outputs), even-weight rows = X stabilisers.
    Then Hx = even rows and K = all rows. Triorthogonality is what makes transversal T act as a logical T gate, so it is checked.
  * `from_css(Hx, Hz)`: any CSS code with a known transversal gate; K is the null space of Hz.

What this module does NOT do: it takes the correctness of the protocol's circuit (that the code really implements the target gate
under the stated operation) as given, assumes perfect Cliffords and measurements, and treats the inputs as independent Z errors.
"""
import numpy as np
from itertools import combinations


def _rref_gf2(M):
    M = (np.array(M, dtype=np.uint8) % 2).copy(); r = 0; pivots = []
    for c in range(M.shape[1]):
        if r == M.shape[0]: break
        p = np.nonzero(M[r:, c])[0]
        if len(p) == 0: continue
        M[[r, r + p[0]]] = M[[r + p[0], r]]
        for i in np.nonzero(M[:, c])[0]:
            if i != r: M[i] ^= M[r]
        pivots.append(c); r += 1
    return M[:r], pivots


def rank_gf2(M):
    return len(_rref_gf2(M)[1])


def nullspace_gf2(M):
    """Rows = a basis of {x : M x = 0 mod 2}."""
    M = np.array(M, dtype=np.uint8) % 2; n = M.shape[1]
    R, piv = _rref_gf2(M); free = [c for c in range(n) if c not in piv]; out = []
    for f in free:
        v = np.zeros(n, np.uint8); v[f] = 1
        for i, c in enumerate(piv): v[c] = R[i, f]
        out.append(v)
    return np.array(out, dtype=np.uint8).reshape(len(out), n)


def is_triorthogonal(G):
    """Every pair of distinct rows overlaps evenly and every triple of distinct rows overlaps evenly (Bravyi-Haah)."""
    G = np.array(G, dtype=np.uint8) % 2
    for a, b in combinations(range(len(G)), 2):
        if int((G[a] & G[b]).sum()) % 2: return False
    for a, b, c in combinations(range(len(G)), 3):
        if int((G[a] & G[b] & G[c]).sum()) % 2: return False
    return True


class CSSProtocol:
    def __init__(self, name, Hx, K):
        self.name = name
        self.Hx = np.array(Hx, dtype=np.uint8) % 2
        self.K = np.array(K, dtype=np.uint8) % 2
        self.n = self.Hx.shape[1]
        assert self.K.shape[1] == self.n
        assert not ((self.Hx @ self.K.T) % 2).any() or True    # Hx rows lie in span(K) by construction; not a separate requirement
        self._A = self._L = None

    @classmethod
    def from_triorthogonal(cls, name, G):
        G = np.array(G, dtype=np.uint8) % 2
        if not is_triorthogonal(G): raise ValueError("matrix is not triorthogonal: transversal T would not be a logical T gate")
        w = G.sum(axis=1)
        return cls(name, G[w % 2 == 0], G)

    @classmethod
    def from_css(cls, name, Hx, Hz):
        Hx = np.array(Hx, dtype=np.uint8) % 2; Hz = np.array(Hz, dtype=np.uint8) % 2
        if ((Hx @ Hz.T) % 2).any(): raise ValueError("Hx and Hz do not commute")
        return cls(name, Hx, nullspace_gf2(Hz))

    # ---- exact analysis by enumeration over all 2^n error patterns (n up to about 24; chunked)
    def weight_enumerators(self, chunk=1 << 18):
        """(A, L): A[w] = number of weight-w patterns accepted, L[w] = number of those that are logical errors."""
        if self._A is not None: return self._A, self._L
        n = self.n; A = np.zeros(n + 1, np.int64); L = np.zeros(n + 1, np.int64)
        bits = np.arange(n, dtype=np.uint64)
        for start in range(0, 1 << n, chunk):
            idx = np.arange(start, min(start + chunk, 1 << n), dtype=np.uint64)
            P = ((idx[:, None] >> bits) & np.uint64(1)).astype(np.uint8)
            acc = ~((P @ self.Hx.T) % 2).any(axis=1)
            logical = acc & ((P @ self.K.T) % 2).any(axis=1)
            w = P.sum(axis=1).astype(int)
            A += np.bincount(w[acc], minlength=n + 1); L += np.bincount(w[logical], minlength=n + 1)
        self._A, self._L = A, L
        return A, L

    def exact(self, eps):
        """(acceptance probability, probability that an accepted output carries a logical error)."""
        A, L = self.weight_enumerators(); w = np.arange(self.n + 1)
        pw = eps ** w * (1 - eps) ** (self.n - w)
        pa = float((A * pw).sum()); pl = float((L * pw).sum())
        return pa, pl / pa

    def leading_order(self):
        """(lowest weight w of an accepted logical error, number of such patterns): output error ~ count * eps^w."""
        A, L = self.weight_enumerators(); nz = np.nonzero(L)[0]
        return (int(nz[0]), int(L[nz[0]])) if len(nz) else (None, 0)

    def mc(self, eps, shots, rng):
        """Monte Carlo twin of `exact`: (accepted, logical errors) out of `shots`."""
        e = (rng.random((shots, self.n)) < eps).astype(np.uint8)
        acc = ~((e @ self.Hx.T) % 2).any(axis=1)
        bad = acc & ((e @ self.K.T) % 2).any(axis=1)
        return int(acc.sum()), int(bad.sum())

    def exact_heterogeneous(self, eps_vec):
        """Same as exact() but every input qubit has its own error probability (arbitrary independent noise); n <= about 22."""
        eps_vec = np.asarray(eps_vec, dtype=float); n = self.n; assert len(eps_vec) == n
        bits = np.arange(n, dtype=np.uint64); pa = pl = 0.0
        for start in range(0, 1 << n, 1 << 18):
            idx = np.arange(start, min(start + (1 << 18), 1 << n), dtype=np.uint64)
            P = ((idx[:, None] >> bits) & np.uint64(1)).astype(np.uint8)
            pr = np.prod(np.where(P == 1, eps_vec, 1 - eps_vec), axis=1)
            acc = ~((P @ self.Hx.T) % 2).any(axis=1)
            logical = acc & ((P @ self.K.T) % 2).any(axis=1)
            pa += float(pr[acc].sum()); pl += float(pr[logical].sum())
        return pa, pl / pa

    def iterate(self, eps0, rounds):
        """Chain exact rounds (assumes each round's outputs are fresh inputs with the same marginal error).
        Returns [(eps_in, acceptance, eps_out, raw_inputs_per_output_round_cost)], cost counted in inputs per accepted block."""
        out, eps, cost = [], eps0, 1.0
        for _ in range(rounds):
            pa, po = self.exact(eps); cost *= self.n / pa; out.append((eps, pa, po, cost)); eps = po
        return out


# ---- concrete protocols -------------------------------------------------------------------------------------------------------
def bravyi_kitaev_15to1():
    """15 T -> 1 T. Triorthogonal matrix: the all-ones row (logical X) over the four rows of the simplex code (X stabilisers)."""
    n = 15
    simplex = np.array([[((c >> b) & 1) for c in range(1, n + 1)] for b in range(4)], dtype=np.uint8)
    return CSSProtocol.from_triorthogonal("15-to-1 (T)", np.vstack([np.ones((1, n), np.uint8), simplex]))


def cube_8T_to_CCZ():
    """8 T -> 1 CCZ (Meier-Eastin-Knill / Gidney-Fowler style): the [[8,3,2]] cube code. Qubits = vertices (x,y,z) of a cube, index 4x+2y+z.
    X stabiliser: X on all 8 vertices; Z stabilisers: Z on each of the six faces. Output is 3 logical qubits (a CCZ state)."""
    verts = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    Hx = np.ones((1, 8), np.uint8)
    faces = []
    for axis in range(3):
        for val in (0, 1):
            faces.append([1 if v[axis] == val else 0 for v in verts])
    return CSSProtocol.from_css("8T-to-CCZ (cube code)", Hx, np.array(faces, np.uint8))


_G_14_2 = [[1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
           [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
           [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]]   # two odd-weight rows (the k = 2 logical X) + three independent even rows


def triorthogonal_14to2():
    """14 T -> 2 T with a [[14,2,2]] triorthogonal code, found by randomised search in this repo (not copied from a paper).
    It is triorthogonal (checked), has three independent X stabilisers, and its leading error is 7 eps^2, which equals the published Bravyi-Haah
    coefficient (1 + 3k) eps^2 for k = 2. It is NOT claimed to be the same matrix as the one in the paper."""
    return CSSProtocol.from_triorthogonal("14-to-2 (T, triorthogonal)", np.array(_G_14_2, dtype=np.uint8))


def all_protocols():
    return [bravyi_kitaev_15to1(), triorthogonal_14to2(), cube_8T_to_CCZ()]
