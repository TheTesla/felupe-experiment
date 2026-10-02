"""PRODUKTION Config C (t=1.0, mu=5, bulk=50) separat: Rampe bis 0.002 MPa, npz."""
import warnings
import numpy as np
import felupe as fe

SRC = open("C:/Users/efame275/femair/luftmatratze_schlauch.py").read()
HEAD = SRC.split("# --------------------------------------------------------------------- Rechnung")[0]
src = HEAD.replace("t      = 0.3", "t      = 1.0").replace("mu, bulk = 0.5, 5.0",
                                                           "mu, bulk = 5.0, 50.0")
exec(src, globals())

p_end = 0.002
n = 12
ps = np.array([p_end * (i / n) ** 1.3 for i in range(n + 1)])
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
    print(f"p={p*1000:6.2f} mbar: umax={np.abs(u).max():6.2f} mm | "
          f"r=[{rd[x0].min():.2f},{rd[x0].max():.2f}] R={R:.2f}")

u = field[0].values
print("\nfertig. umax:", np.abs(u).max(), "mm")

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
    L=L, W=W, t=1.0, s=s_adj, r=r, R=R, N_rows=N_rows,
)
print("Ergebnisse gespeichert (t=1.0 mm, mu=5 MPa).")
