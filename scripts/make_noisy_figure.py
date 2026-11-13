import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from magic15 import iterate_rounds_noisy
fig, ax = plt.subplots(figsize=(5.6, 4))
for p, st in ((0.0, "-"), (1e-5, "--"), (1e-4, "--"), (1e-3, "--")):
    r = iterate_rounds_noisy(0.01, p, 5)
    ax.semilogy(range(0, 6), [0.01] + [x[2] for x in r], "o" + st, ms=4, label=f"Clifford noise p = {p:g}" if p else "ideal Cliffords")
    print(f"p={p:g}: " + "  ".join(f"{x[2]:.2e}" for x in r))
ax.set_xlabel("distillation rounds"); ax.set_ylabel("output error"); ax.set_ylim(1e-40, 2e-2)
ax.set_title("Toy noise-floor model: input error 0.01"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("fig_noise_floor.png", dpi=150)
