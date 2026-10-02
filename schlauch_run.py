"""Produktionslauf Schlauch: stabile Drucksequenz bis 0.000115 MPa + Speichern.
(Das ist die Schnappgrenze des Punktschweiss-Schlauchs bei dieser Geometrie.)
Die Sequenz wird im Skript ersetzt: maximal stabil erreichbarer Druck."""
import warnings
import numpy as np
import felupe as fe

src = open("C:/Users/efame275/femair/luftmatratze_schlauch.py").read()
header = src.split("# --------------------------------------------------------------------- Rechnung")[0]
exec(header)

seq = [0.0, 2e-5, 4.5e-5, 8e-5, 9e-5, 1.0e-4, 1.05e-4, 1.1e-4, 1.15e-4]
for p in seq[1:]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps = np.array([seq[seq.index(p) - 1], p])
        st = fe.Step(items=[solid, load], ramp={load: ps}, boundaries=boundaries)
        fe.Job(steps=[st], verbose=False).evaluate()

u = field[0].values
print("fertig. p_end:", seq[-1], "MPa | umax:", np.abs(u).max(), "mm")

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
    points=points, cells=cells, u=u, stress=s_t, vm=vm, p_max=seq[-1],
    L=L, W=W, t=t, s=s_adj, r=r, R=R, N_rows=N_rows,
)
print("Ergebnisse gespeichert.")
