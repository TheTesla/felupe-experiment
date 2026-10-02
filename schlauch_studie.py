"""Studie: Wie hoch kommt der Luftschlauch mit Folienstärke/Material?
Adaptive Druckinkremente (halbiere bei Fail) bis Ziel 0.002 MPa (20 mbar / 0.02 bar).
Beste Konfiguration -> voller Produktionslauf + npz. Bilder danach via visualize_schlauch.py."""
import warnings
import numpy as np
import felupe as fe

SRC = open("C:/Users/efame275/femair/luftmatratze_schlauch.py").read()
HEAD = SRC.split("# --------------------------------------------------------------------- Rechnung")[0]
P_TARGET = 0.002   # MPa = 20 mbar = 0.02 bar


def build(t, mu, bulk=None):
    bulk = bulk if bulk is not None else 10 * mu
    src = HEAD.replace("t      = 0.3", f"t      = {t}").replace("mu, bulk = 0.5, 5.0",
                                                               f"mu, bulk = {mu}, {bulk}")
    exec(src, globals())
    return solid, load, boundaries, region, field


def scan(t, mu, bulk=None, tag=""):
    """Adaptive Rampe; return (erreichter Druck, Schritte, snapshot-Objekte)."""
    solid, load, boundaries, region, field = build(t, mu, bulk)
    p, dp = 0.0, P_TARGET / 6
    hist = [0.0]
    while p < P_TARGET - 1e-12 and dp > P_TARGET / 2000:
        p_try = min(p + dp, P_TARGET)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st = fe.Step(items=[solid, load], ramp={load: np.array([p, p_try])},
                             boundaries=boundaries)
                fe.Job(steps=[st], verbose=False).evaluate()
            p = p_try
            hist.append(p)
            dp = min(dp * 1.25, P_TARGET / 4)
        except Exception:
            dp /= 2
    print(f"  {tag}: t={t} mu={mu} bulk={bulk}: erreicht {p*1000:.2f} mbar "
          f"({'ZIEL' if p >= P_TARGET - 1e-12 else 'SNAP-LIMIT'}) | {len(hist)-1} Schritte")
    return p, solid, load, boundaries, region, field, hist


results = []
for t, mu, bulk, tag in [(0.5, 0.5, None, "A"), (1.0, 0.5, None, "B"), (1.0, 5.0, 50.0, "C")]:
    try:
        r = scan(t, mu, bulk, tag)
        results.append((r[0], t, mu, bulk, r))
    except Exception as e:
        print(f"  {tag}: t={t} mu={mu}: BUILD/SCAN FAIL {type(e).__name__}: {e}")

results.sort(reverse=True, key=lambda x: x[0])
best_p, best_t, best_mu, best_bulk, best = results[0]
print(f"\nBeste Konfiguration: t={best_t} mm, mu={best_mu}, bulk={best_bulk} "
      f"-> {best_p*1000:.2f} mbar")

# ---- Produktionslauf der besten Konfiguration (saubere Sequenz bis Ziel/Limit)
p_end = best_p
solid, load, boundaries, region, field = build(best_t, best_mu, best_bulk)
n = 14
ps = np.array([p_end * (i / n) ** 1.5 for i in range(n + 1)])
ps[0], ps[-1] = 0.0, p_end
prev = 0.0
for p in ps[1:]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = fe.Step(items=[solid, load], ramp={load: np.array([prev, p])}, boundaries=boundaries)
        fe.Job(steps=[st], verbose=False).evaluate()
    prev = p

u = field[0].values
rd = np.hypot(points[:, 1] + u[:, 1], points[:, 2] + u[:, 2])
x0 = np.isclose(points[:, 0], 0.0)
print(f"\nPRODUKTION: t={best_t} mu={best_mu}: p={p_end*1000:.2f} mbar | "
      f"umax={np.abs(u).max():.2f} mm | r_min={rd[x0].min():.2f} r_max={rd[x0].max():.2f} (R={R:.2f})")

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
print("max. von-Mises:", vm.max(), "MPa")

np.savez_compressed(
    "C:/Users/efame275/femair/luftmatratze_schlauch_result.npz",
    points=points, cells=cells, u=u, stress=s_t, vm=vm, p_max=p_end,
    L=L, W=W, t=best_t, s=s_adj, r=r, R=R, N_rows=N_rows,
)
print("Ergebnisse gespeichert (t in npz).")
