import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from magic15 import exact_round, mc_round, iterate_rounds
rng = np.random.default_rng(7)
eps = np.logspace(-3, -1.3, 14)
ex = np.array([exact_round(e)[1] for e in eps])
mc = []
for e in eps:
    a, b = mc_round(e, 2_000_000, rng); mc.append(b / a if a else np.nan)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].loglog(eps, ex, "-", label="exact (enumeration)"); ax[0].loglog(eps, mc, "o", ms=4, label="Monte Carlo (2e6 shots)")
ax[0].loglog(eps, 35 * eps ** 3, "--", label=r"$35\,\epsilon^3$"); ax[0].loglog(eps, eps, ":", c="gray", label="no distillation")
ax[0].set_xlabel("input error $\\epsilon$"); ax[0].set_ylabel("output error (accepted)"); ax[0].legend(); ax[0].set_title("One round of 15-to-1")
for e0 in (0.02, 0.01, 0.005):
    r = iterate_rounds(e0, 3)
    ax[1].semilogy([x[3] for x in r], [x[2] for x in r], "o-", label=f"$\\epsilon_0$={e0}")
ax[1].set_xlabel("expected raw T states per output"); ax[1].set_ylabel("output error"); ax[1].legend(); ax[1].set_title("1, 2, 3 rounds: error vs cost")
plt.tight_layout(); plt.savefig("fig_distillation.png", dpi=150)
for e0 in (0.01,):
    for i, (ei, pa, po, c) in enumerate(iterate_rounds(e0, 3), 1): print(f"eps0={e0} round {i}: in={ei:.3e} accept={pa:.4f} out={po:.3e} raw T/output={c:.1f}")
