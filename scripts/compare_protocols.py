import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from magic15 import all_protocols
P = all_protocols()
print(f"{'protocol':32s} {'in':>3s} {'out':>3s} {'leading error':>16s} {'accept@1%':>10s} {'err@1%':>10s} {'inputs/block@1%':>15s} {'1-round threshold':>18s}")
outs = {"15-to-1 (T)": 1, "14-to-2 (T, triorthogonal)": 2, "8T-to-CCZ (cube code)": 1}
for p in P:
    w, c = p.leading_order(); pa, po = p.exact(0.01)
    lo, hi = 0.01, 0.45
    for _ in range(80):
        mid = (lo + hi) / 2
        (lo, hi) = (mid, hi) if p.exact(mid)[1] < mid else (lo, mid)
    print(f"{p.name:32s} {p.n:3d} {outs[p.name]:3d} {c:>4d} eps^{w:<9d} {pa:10.4f} {po:10.3e} {p.n / pa:15.1f} {lo:18.4f}")
fig, ax = plt.subplots(figsize=(5.8, 4.2)); eps = np.logspace(-3, -1.1, 30)
for p in P:
    w, c = p.leading_order(); y = [p.exact(e)[1] for e in eps]
    l, = ax.loglog(eps, y, "-", label=f"{p.name}: {c} eps^{w}"); ax.loglog(eps, c * eps ** w, ":", color=l.get_color(), lw=1)
ax.loglog(eps, eps, "-", color="gray", lw=0.8, label="no distillation")
ax.set_xlabel("input error"); ax.set_ylabel("error of an accepted output block"); ax.legend(fontsize=7); ax.set_title("One round, ideal Cliffords (dotted: leading order)")
plt.tight_layout(); plt.savefig("fig_protocols.png", dpi=150)
