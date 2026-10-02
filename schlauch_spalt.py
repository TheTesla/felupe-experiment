"""Schlauch v4: ECHTE Luftkammer - Innfolie r=R-w0/2, Aussenfolie r=R+w0/2 (Anfangsspalt w0).
Der Druck drueckt die Folien auseinander; Dots + Naht + Enden verbinden.
Test: mu=5 (E~15), t=0.3, r_dot=15, w0=15mm, p=0.02 bar -> starke Loben erwartet."""
import warnings
import numpy as np
import felupe as fe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = open("C:/Users/efame275/femair/luftmatratze_schlauch.py").read()
HEAD = SRC.split("# --------------------------------------------------------------------- Rechnung")[0]
# Geometrie mit Spalt: ring_points(R-t)->(R-w0/2), (R)->(R+w0/2) fuer Aussen
w0 = 15.0
src = HEAD.replace("mu, bulk = 0.5, 5.0", "mu, bulk = 5.0, 50.0")
src = src.replace("ring_points(R - t), ring_points(R), ring_points(R), ring_points(R + t)",
                  f"ring_points(R - w0/2), ring_points(R - w0/2 + t), "
                  f"ring_points(R + w0/2 - t), ring_points(R + w0/2)")
src = src.replace("mask = np.isclose(rad_pt, R)",
                  "mask = (rad_pt > R - w0/2 + t*0.5) & (rad_pt < R + w0/2 - t*0.5)")
src = src.replace(
    'i_node = np.where(np.isclose(points[:, 0], 0.0) & np.isclose(points[:, 1], R)\n'
    '                  & np.isclose(points[:, 2], 0.0))[0]',
    'i_node = np.where(np.isclose(points[:, 0], 0.0) & (np.abs(np.hypot(points[:, 1], points[:, 2]) - R) < w0/2)\n'
    '                  & np.isclose(points[:, 2], 0.0))[0]\n'
    'if len(i_node) == 0:\n'
    '    d2 = points[:, 0]**2 + (np.hypot(points[:, 1], points[:, 2]) - R)**2 + points[:, 2]**2\n'
    '    d2[points[:, 0] > 1e-9] = np.inf\n'
    '    i_node = [int(np.argmin(d2))]\n'
    'i_node = [int(i_node[0])]')
exec(src, globals())

# Kavitaet = Kanal zwischen (R-w0/2+t) und (R+w0/2-t) -> Grenzflaechen bei beiden Radien
# mask waehlt Knoten BEIDER Kanalgrenzflaechen - Nur-Surface-Filter behaelt beide.
print("Kavitaetsflaechen:", region_pressure.mesh.ncells)

# Druckrampe bis 0.02 bar
p_end = 0.002
n = 12
ps = p_end * np.linspace(0, 1, n + 1) ** 1.5
ps[0], ps[-1] = 0.0, p_end
prev = 0.0
for p in ps[1:]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = fe.Step(items=[solid, load], ramp={load: np.array([prev, p])}, boundaries=boundaries)
        fe.Job(steps=[st], verbose=False).evaluate()
    prev = p
    u = field[0].values
    x0 = np.isclose(points[:, 0], 0.0)
    rd = np.hypot(points[x0][:, 1] + u[x0][:, 1], points[x0][:, 2] + u[x0][:, 2])
    print(f"p={p*1e5:6.2f} mbar: r=[{rd.min():.2f},{rd.max():.2f}] umax={np.abs(u).max():.3f}")

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
    "C:/Users/efame275/femair/luftmatratze_schlauch_spalt_result.npz",
    points=points, cells=cells, u=u, stress=s_t, vm=vm, p_max=p_end,
    L=L, W=W, t=t, s=s_adj, r=r, R=R, N_rows=N_rows, w0=w0,
)
print("gespeichert.")

# Querschnittsplot (echter Massstab)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
x0 = np.isclose(points[:, 0], 0.0)
Y = points[x0][:, 1] + u[x0][:, 1]
Z_ = points[x0][:, 2] + u[x0][:, 2]
th = np.degrees(np.arctan2(points[x0][:, 2], points[x0][:, 1]))
o = np.argsort(th)
# Lagen unterscheiden: rad vor Verformung
rad0 = np.hypot(points[x0][:, 1], points[x0][:, 2])
lagB = rad0 < R   # Unterfolie (innen)
oo_a = [k for k in o if not lagB[k]]
oo_b = [k for k in o if lagB[k]]
ax.plot(Y[oo_a], Z_[oo_a], "-o", lw=1, ms=2.5, color="crimson", label="Aussenfolie")
ax.plot(Y[oo_b], Z_[oo_b], "-o", lw=1, ms=2.5, color="steelblue", label="Innenfolie")
for RR, lab, col in ((R + w0/2, "Aussenfolie R+t", "gray"),
                     (R - w0/2, "Innenfolie R-t", "lightgray")):
    ax.add_patch(plt.Circle((0, 0), RR, fill=False, color=col, ls="--", lw=1))
for tdg in (90, 180, 270):
    a = np.radians(tdg)
    ax.plot([0, (R + w0) * np.cos(a)], [0, (R + w0) * np.sin(a)], color="darkgreen", ls=":", lw=0.9)
ax.set_aspect("equal")
lim = np.max(np.abs(np.array([Y, Z_]))) + 8
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("y [mm]"); ax.set_ylabel("z [mm]")
ax.set_title(f"Schlauch mit Anfangsspalt w0={w0:.0f} mm, p={p_end*1e5:.1f} mbar\n"
             f"t=0.3, E~15 MPa, r_dot={r} (echter Massstab)")
ax.legend(fontsize=8); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("C:/Users/efame275/femair/schlauch_spalt_querschnitt.png", dpi=130)
print("Plot ok")
