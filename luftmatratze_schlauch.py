"""Luftmatratze v3: zwei gegenueberliegende Kanten verklebt -> Luftschlauch (FElupe).

Aus der Doppelfolien-Matte (v2: Punkt-Schweissungen im Dreiecksverband) werden die
beiden LANGEN Kanten (y=+-W/2) miteinander verklebt -> geschlossener Schlauch
entlang x (wie ein Luftboom). Querschnitt: Kreis mit Mittellinien-Radius R = W/(2pi);
die beiden Folien bilden die Schlauchwand (je 1 Element dick), verbunden nur durch:
  - kreisfoermige Schweisspunkte (Radius r) im gleichseitigen Dreiecksverband
    (6 aequidistante Nachbarn, Abstand s, automatisch angepasst damit das Muster
    um den Umfang exakt schliesst: N_rows Reihen a 2pi/N_rows),
  - die Klebnaht der verklebten Kanten (eine volle Meridian-Linie, theta=0),
  - die verschlossenen Schlauchenden (x=+-L/2 -> Halbmodell: x=L/2).
Innendruck p im Ringspalt zwischen den Folien -> Wand spannt, Pillowing zwischen
den Punkten (Innfolie woelbt zum Schlauchinneren, Aussenfolie nach aussen).

Modellierung: Startkonfiguration direkt rund (das flach->rund-Aufblaehen wird
uebersprungen; Spannungslage unter Druck ist membrane-path-unabhaengig).
Halbmodell x in [0, L/2] mit Symmetrie ux=0 bei x=0.

FELUPE-BC-SEMANTIK: skip=True bedeutet - diese Komponente wird NICHT vorgeschrieben!
"""
import os
import numpy as np
import felupe as fe

# ------------------------------------------------------------------ Parameter
L      = 600.0   # Laenge (x) [mm]
W      = 300.0   # Breite (y) [mm] = Schlauchumfang
t      = 0.3     # Folienstaerke [mm]
s      = 60.0    # gewuenschter Schweisspunkt-Abstand (Dreiecksseite) [mm]
r      = 15.0    # Schweisspunkt-Radius [mm]
cell   = 6.0     # Elementkante [mm]
p_max  = 0.002   # Innendruck [MPa] = 0.02 bar (darueber kippt das Innfoelien-Pillowing)
nsteps = 21

mu, bulk = 0.5, 5.0   # Neo-Hooke [MPa] (E ~ 1.5 MPa, weiches PVC)

if os.environ.get("SMOKE"):   # schneller Funktionstest
    L, W, cell, nsteps = 120.0, 96.0, 8.0, 4
    s, r = 30.0, 8.0

# ----------------------------------------------------- Schlauch-Geometrie
R = W / (2 * np.pi)                       # Mittellinien-Radius [mm]
N_theta = int(round(W / cell))            # Zellen um den Umfang
N_x = int(round((L / 2) / cell))          # Zellen laengs (Halbmodell)
xs = np.linspace(0.0, L / 2, N_x + 1)
dtheta = 2 * np.pi / N_theta
thetas = np.arange(N_theta) * dtheta

# Dreiecksmuster um den Umfang schliessen: N_rows Reihen, s angepasst
N_rows = max(3, int(round(W / (s * np.sqrt(3) / 2))))
s_adj = 2 * W / (np.sqrt(3) * N_rows)     # exakte Dreiecksseite
print(f"Schlauch: R={R:.2f} mm | Umfang={W:.0f} | {N_theta} theta-Zellen, {N_x} x-Zellen")
print(f"Muster: {N_rows} Reihen um den Umfang -> s: {s:.1f} -> {s_adj:.2f} mm "
      f"(Reihenabstand {W/N_rows:.1f} mm auf dem Umfang)")

# Dot-Zentren (Halbmodell): Reihe j=0 ist die Klebnaht (theta=0), j=1..N_rows-1 Dots
dots = []
for j in range(1, N_rows):
    th_d = 2 * np.pi * j / N_rows
    if j % 2 == 1:      # Spalten bei x = m*s_adj (inkl. Symmetrieebene x=0)
        cols = [m * s_adj for m in range(int(np.floor((L / 2) / s_adj)) + 1)]
    else:               # um s_adj/2 versetzt
        cols = [s_adj / 2 + m * s_adj
                for m in range(int(np.floor((L / 2 - s_adj / 2) / s_adj)) + 1)]
    dots += [(xd, th_d) for xd in cols if xd <= L / 2 + 1e-9]
print(f"Schweisspunkte im Halbmodell: {len(dots)}  (Vollschlauch ~ {2*len(dots) - sum(1 for xd,_ in dots if abs(xd)<1e-9)})")

# ------------------------------------------------------------- Mesh (4 Ringe)
nring = (N_x + 1) * N_theta
Xg, THg = np.meshgrid(xs, thetas, indexing="ij")     # (N_x+1, N_theta)

def ring_points(rad):
    return np.column_stack([Xg.ravel(order="C"),
                            rad * np.cos(THg).ravel(order="C"),
                            rad * np.sin(THg).ravel(order="C")])

OFF = [0, nring, 2 * nring, 3 * nring]
points = np.vstack([ring_points(R - t), ring_points(R), ring_points(R), ring_points(R + t)])
# Ringe: 0=B0 (r=R-t, Innfolie aussen -> Schlauchinneres), 1=B1 (r=R), 2=T0 (r=R),
#        3=T1 (r=R+t, Aussenfolie aussen -> Umgebung). Kavitaet = Spalt B1<->T0.

def nid(l, i, k):
    return OFF[l] + i * N_theta + (k % N_theta)

def foil_cells(lb, lt):
    cl = []
    for i in range(N_x):
        for k in range(N_theta):
            # Bottom-Quad in theta-Richtung zuerst -> positive Volumina (Outward-Normale)
            cl.append([nid(lb, i, k), nid(lb, i, k + 1), nid(lb, i + 1, k + 1), nid(lb, i + 1, k),
                       nid(lt, i, k), nid(lt, i, k + 1), nid(lt, i + 1, k + 1), nid(lt, i + 1, k)])
    return cl

cells = np.array(foil_cells(0, 1) + foil_cells(2, 3), dtype=np.int32)

# ------------------------------------------- Schweisszellen (Dot-Patches, pixeliert)
xc = 0.5 * (xs[:-1] + xs[1:])
thc = thetas + 0.5 * dtheta
weld_cell = np.zeros((N_x, N_theta), dtype=bool)
for xd, thd in dots:
    dth = np.angle(np.exp(1j * (thc - thd)))            # gewrapped
    weld_cell |= (xc[:, None] - xd) ** 2 + (R * dth[None, :]) ** 2 <= r ** 2

# Schweissknoten = Ecken der Schweisszellen (theta wrappt: Spalte kN-1 -> Knoten 0 = Naht)
# + Naht (k=0) + Endnaht (i=N_x)
weld_node = np.zeros((N_x + 1, N_theta), dtype=bool)
weld_node[:-1, :] |= weld_cell      # Ecke (i, k) und via Spalte k+1 auch (i, k+1)
weld_node[1:, :] |= weld_cell       # Ecke (i+1, k) und (i+1, k+1)
weld_node[:, 0] = True              # Klebnaht der verklebten Kanten (Meridian theta=0)
weld_node[-1, :] = True             # verschlossenes Schlauchende x=L/2
print(f"Schweisszellen (pixelierte Dots): {weld_cell.sum()} | zusammenzufuehrende Knoten: {weld_node.sum()}")

# Merge: NUR verschweisste T0-Knoten -> B1 (Rest bleibt T0!)
flat = np.arange(nring)
t0_new = np.where(weld_node.ravel(order="C"), OFF[1] + flat, OFF[2] + flat)
n_bot = len(foil_cells(0, 1))
top = cells[n_bot:].copy()
in_t0 = (top >= OFF[2]) & (top < OFF[3])
top[in_t0] = t0_new[top[in_t0] - OFF[2]]
cells = np.vstack([cells[:n_bot], top])

used = np.unique(cells.ravel())
remap = -np.ones(len(points), dtype=np.int64)
remap[used] = np.arange(len(used), dtype=np.int64)
points = points[used]
cells = remap[cells].astype(np.int32)
print(f"Zellen: {len(cells)} | Knoten: {len(points)}")

mesh = fe.Mesh(points, cells, "hexahedron")
region = fe.RegionHexahedron(mesh)
field = fe.FieldContainer([fe.Field(region, dim=3)])
if region.dV.min() <= 0:
    raise RuntimeError("Nicht-positives Elementvolumen - Zell-Ordering pruefen!")

# ------------------------------------------------- Druckflaeche = Kavitaet (r=R)
rad_pt = np.hypot(points[:, 1], points[:, 2])
mask = np.isclose(rad_pt, R)
region_pressure = fe.RegionHexahedronBoundary(mesh, mask=mask)
print("Kavitaets-Grenzflaechen:", region_pressure.mesh.ncells)
if region_pressure.mesh.ncells == 0:
    raise RuntimeError("Kavitaet leer - Merge-Logik pruefen!")
field_pressure = fe.FieldContainer([fe.Field(region_pressure, dim=3)])
field_pressure.link(field)

solid = fe.SolidBody(fe.NeoHooke(mu=mu, bulk=bulk), field)
load = fe.SolidBodyPressure(field_pressure, pressure=0.0)

# ------------------------------------------------------------- Randbedingungen
boundaries = fe.BoundaryDict()
boundaries["fix-x"] = fe.Boundary(field[0], fx=0.0, skip=(False, True, True))  # Symmetrie x=0
# Rigidmodes y/z/Drehung um x: ein Knoten auf der steifen Nahtlinie (x=0, theta=0)
i_node = np.where(np.isclose(points[:, 0], 0.0) & np.isclose(points[:, 1], R)
                  & np.isclose(points[:, 2], 0.0))[0]
assert len(i_node) == 1, f"Nahtknoten mehrdeutig: {i_node}"
m_uy = np.zeros(len(points), dtype=bool); m_uy[i_node[0]] = True
m_uz = np.zeros(len(points), dtype=bool); m_uz[i_node[0]] = True
boundaries["fix-uy"] = fe.Boundary(field[0], mask=m_uy, skip=(True, False, True))
boundaries["fix-uz"] = fe.Boundary(field[0], mask=m_uz, skip=(True, True, False))
print(f"BC: ux=0 auf x=0 ({len(boundaries['fix-x'].points)} Knoten) | uy=uz=0 an 1 Nahtknoten")

# --------------------------------------------------------------------- Rechnung
# Feine Druckrampe: kleine absolute Schritte am Anfang (flache, knickempfindliche Wand)
pressure_steps = np.array(sorted(set(
    list(p_max * np.linspace(0.0, 1.0, nsteps) ** 2)
    + list(np.geomspace(2e-5, p_max, 8))
)))
pressure_steps[0] = 0.0
pressure_steps[-1] = p_max
step = fe.Step(items=[solid, load], ramp={load: pressure_steps}, boundaries=boundaries)
job = fe.Job(steps=[step])

def cb(dx, x, iteration, xnorm, fnorm, success):
    print(f"  iter {iteration:2d}: |r|={fnorm:.3e} | umax={np.abs(field[0].values).max():.3f} mm")

job.evaluate(callback=cb)

u = field[0].values
print("\nfertig. max. Verschiebung:", np.abs(u).max(), "mm")

# -------------------------------------------------------------- Spannungen (Cauchy)
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
print("max. von-Mises-Spannung:", vm.max(), "MPa")

np.savez_compressed(
    "C:/Users/efame275/femair/luftmatratze_schlauch_result.npz",
    points=points, cells=cells, u=u, stress=s_t, vm=vm, p_max=p_max,
    L=L, W=W, t=t, s=s_adj, r=r, R=R, N_rows=N_rows,
)
print("Ergebnisse gespeichert.")
