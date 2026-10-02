"""Studie: Pillowing-Tiefe des Luftschlauchs - Druck-Scan mit adaptiver Rampe + Spruengen.
D: r=15 (User-Original), t=0.3, E~15 MPa (mu=5), p bis 0.1 bar
E: r=6 (gressere freie Felder), sonst gleich
Diagnose je Schritt: r_naht (Seam-Dent), r_min/r_max (Loben), umax."""
import warnings
import numpy as np
import felupe as fe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = open("C:/Users/efame275/femair/luftmatratze_schlauch.py").read()
HEAD = SRC.split("# --------------------------------------------------------------------- Rechnung")[0]
P_TARGET = 0.01   # MPa = 0.1 bar


def build(t, mu, r_dot):
    src = HEAD.replace("t      = 0.3", f"t      = {t}")
    src = src.replace("mu, bulk = 0.5, 5.0", f"mu, bulk = {mu}, {10 * mu}")
    src = src.replace("r      = 15.0", f"r      = {r_dot}")
    exec(src, globals())


def diag(R):
    u = field[0].values
    x0 = np.isclose(points[:, 0], points[:, 0].min())
    rd = np.hypot(points[x0][:, 1] + u[x0][:, 1], points[x0][:, 2] + u[x0][:, 2])
    th = np.arctan2(points[x0][:, 2], points[x0][:, 1])
    rad0 = np.hypot(points[x0][:, 1], points[x0][:, 2])
    seam = np.isclose(th, 0.0, atol=0.03) & np.isclose(rad0, R)
    return dict(r_seam=float(rd[seam].mean()), r_min=float(rd.min()), r_max=float(rd.max()),
                umax=float(np.abs(u).max()))


def run_scan(t, mu, r_dot, p_target, tag):
    build(t, mu, r_dot)
    p, dp = 0.0, p_target / 6
    dp_min = p_target / 1500
    hist = [(0.0, diag(R))]
    snaps = []
    attempts = 0
    while p < p_target - 1e-15 and attempts < 250:
        attempts += 1
        p_try = min(p + dp, p_target)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st = fe.Step(items=[solid, load], ramp={load: np.array([p, p_try])},
                             boundaries=boundaries)
                fe.Job(steps=[st], verbose=False).evaluate()
            p = p_try
            dp = min(dp * 1.3, p_target / 3)
            hist.append((p, diag(R)))
        except Exception:
            if dp > dp_min:
                dp /= 2
                continue
            # Sprung-Versuche ueber den Limit-Point (snap)
            jumped = False
            for f in (0.15, 0.3, 0.5, 0.8):
                p_try = p + f * (p_target - p)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        st = fe.Step(items=[solid, load],
                                     ramp={load: np.array([p, p_try])}, boundaries=boundaries)
                        fe.Job(steps=[st], verbose=False).evaluate()
                    snaps.append((p * 1e5, p_try * 1e5))  # mbar
                    p = p_try
                    dp = p_target / 12
                    hist.append((p, diag(R)))
                    jumped = True
                    break
                except Exception:
                    continue
            if not jumped:
                break
    print(f"[{tag}] erreicht p = {p*1e5:.2f} mbar von {p_target*1e5:.0f} mbar Ziel "
          f"| {len(hist)-1} konvergierte Schritte | Spruenge: "
          f"{['%.2f->%.2f mbar' % s for s in snaps] if snaps else 'keine'}")
    for pp, d in hist[::max(1, len(hist) // 10)]:
        print(f"    p={pp*1e5:7.2f} mbar: r_seam={d['r_seam']:6.2f} r=[{d['r_min']:6.2f},"
              f"{d['r_max']:6.2f}] R={R:.2f} umax={d['umax']:5.2f}")
    return hist, snaps


def finish_and_save(tag, fname_pzo, fname_png):
    u = field[0].values.copy()
    F = field[0].extract(grad=True)
    P = solid.umat.gradient([F, solid.results.statevars])[0]
    Fq = np.transpose(F, (2, 3, 0, 1))
    Pq = np.transpose(P, (2, 3, 0, 1))
    J = np.linalg.det(Fq)
    cauchy = np.einsum("qcij,qcjk->qcik", Pq, Fq.transpose(0, 1, 3, 2)) / J[..., None, None]
    s_t = fe.project(cauchy.transpose(2, 3, 0, 1), region)
    vm = np.sqrt(0.5 * ((s_t[:, 0, 0] - s_t[:, 1, 1]) ** 2 + (s_t[:, 1, 1] - s_t[:, 2, 2]) ** 2
                        + (s_t[:, 2, 2] - s_t[:, 0, 0]) ** 2)
                 + 3 * (s_t[:, 0, 1] ** 2 + s_t[:, 1, 2] ** 2 + s_t[:, 2, 0] ** 2))
    np.savez_compressed(fname_pzo, points=points, cells=cells, u=u, stress=s_t, vm=vm,
                        p_max=hist[-1][0], L=L, W=W, t=t, s=s_adj, r=r, R=R, N_rows=N_rows)
    print(f"[{tag}] vm_max={vm.max():.3f} MPa | gespeichert: {fname_pzo}")

    # Querschnitts-Plot (echter Massstab)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    x0 = np.isclose(points[:, 0], 0.0)
    Y = points[x0][:, 1] + u[x0][:, 1]
    Z_ = points[x0][:, 2] + u[x0][:, 2]
    rad0 = np.hypot(points[x0][:, 1], points[x0][:, 2])
    inner = rad0 < R
    th = np.degrees(np.arctan2(points[x0][:, 2], points[x0][:, 1]))
    o = np.argsort(th)
    for mask, col, lab in ((~inner, "crimson", "Aussenfolie"), (inner, "steelblue", "Innenfolie")):
        oo = o[mask[o]]
        ax.plot(Y[oo], Z_[oo], "-o", lw=1.0, ms=2.5, color=col, label=lab)
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="gray", ls="--", lw=1.2, label=f"R={R:.1f}"))
    for tdg in (90, 180, 270):
        a = np.radians(tdg)
        ax.plot([0, (R + 8) * np.cos(a)], [0, (R + 8) * np.sin(a)], color="darkgreen", ls=":", lw=0.9)
    ax.set_aspect("equal")
    lim = np.abs(np.array([Y, Z_])).max() + 8
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("y [mm]"); ax.set_ylabel("z [mm]")
    ax.set_title(f"Schlauch {tag}: p={hist[-1][0]*1e5:.2f} mbar, r_dot={r} mm, t={t} mm, "
                 f"E~{3*mu:.0f} MPa (echter Massstab)")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fname_png, dpi=130)
    plt.close()
    print(f"[{tag}] Plot: {fname_png}")


results = {}
for tag, r_dot, t, mu in (("D r=15", 15.0, 0.3, 5.0), ("E r=6", 6.0, 0.3, 5.0)):
    hist, snaps = run_scan(t, mu, r_dot, P_TARGET, tag)
    results[tag] = (hist, snaps)
    finish_and_save(tag, f"C:/Users/efame275/femair/schlauch_{tag[0]}.npz",
                    f"C:/Users/efame275/femair/schlauch_{tag[0]}_querschnitt.png")

print("\n=== ZUSAMMENFASSUNG ===")
for tag, (hist, snaps) in results.items():
    p_end, d_end = hist[-1]
    print(f"{tag}: p_end={p_end*1e5:.2f} mbar | r_seam={d_end['r_seam']:.2f} | "
          f"r=[{d_end['r_min']:.2f},{d_end['r_max']:.2f}] (R={R:.2f}) | "
          f"Ausbuchtung in/out: {R-d_end['r_min']:.2f} / {d_end['r_max']-R:.2f} mm")
