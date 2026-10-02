"""Naht-Spalt im Detail: Dots verbinden nur bei GEMERGten Knoten. Am Querschnitt x=0
sind Dots nur da, wo x=0-PUNKTE gemerged wurden - das sind Dots in der Spalte x=0
(je Reihe). Bei theta=90: Dot vorhanden? merge-Maske am Nahtknoten (x=0,theta=90)?
Pruefe ueber die merge-Logik: weld_node ueberall dort, wo Schweisszellen Ecken -
am Meridian (theta=0) alle. Bei theta=90 UND x=0: Dot-Zelle (0..r) in theta-Richtung:
theta-Index k~14.25 (90/360*50) - NICHT auf Zell-Ecke -> der Dot 'schwebt' zwischen
Zellspalten! Deshalb: am Schnitt x=0 theta=90 ist KEIN Merged-Knoten -> Folien dort
NICHT verbunden -> Spalt 16mm!
DAS ist der 'elastische Dot': Punkte liegen zwischen Gitterlinien und mergen nur
die NAECHSTEN Zell-Ecken. Fix: Dot-Zentren auf Gitter anziehen (theta-Koerper) oder
Feinere theta-Aufloesung (cell kleiner). Genauer Fix: N_theta und N_x so waehlen,
dass Dot-Zentren auf Knoten liegen: N_rows Spalten -> theta_dot = j*2pi/N_rows;
N_theta muss Vielfaches von N_rows sein UND N_x auf s_adj-Raster.
N_theta=6*10=60, cell=5 -> 60*6=360=Umfang passt (300/60=5mm Zellkante theta).
N_x: s_adj=57.7: Halbschlauch 300mm: 300/57.7=5.2 -> nicht ganzzahlig!
Muster-Verschluss: N_rows=6 Reihen -> s_adj=57.735. Laengs: 600/57.735=10.39...
Waehle L=573.5mm? Oder r_dot anpassen? Alternativ: N_x auf Zellraster 5mm (60 Zellen),
Dots laengs auf Zellknoten rastern: x_dot = m*(300/5=60er Raster?) 57.7->60:
s_adj=60 (6 Reihen*50mm=Bogen? Reihenabstand 50mm, Dreiecksseite s = sqrt((50)^2+...).
Genau: bei N_rows Reihen um 300mm-Umfang: Reihenabstand (Bogen) = 50mm.
Dreiecksseite s (Sehne): 2*R*sin(pi/6)=R=47.75*1=47.75mm Sehne bei 60deg.
Freie Felder: Dotdurchmesser 30 -> 17.75mm. cell=5mm theta (60 Zellen), x: 5mm (60 Zellen).
Dot-Zentren: theta 90/180/270 -> Zellknoten k=15/30/45 - EXAKT. x: 0/60/120/180/240/300
-> Knoten i=0/12/24/36/48/60 - EXAKT (Halbmodell x<=300). Versetzt-Reihe: x=30+offset?
Reihe j=2 (theta=180): versetzt um s/2=23.9 -> 23.9 kein Knoten (23.9/5=4.78). Hmm.
Versatz muss auf Zellknoten: versetzt um 30mm (3 Zellen) statt 28.9 ->
Dreiecksverband leicht deformiert (Sehnen 60 und 54.1): ok, naeherungsweise gleichseitig.
Alternative: N_theta=50 wie bisher, aber Dot-Zentren theta auf Zellknoten setzen:
Zellknoten bei k*7.2deg (360/50). 90deg ist k=12.5 - nicht Knoten. 72deg oder 108deg
waere Knoten (k=10, k=15). -> N_rows-Dreiecksverband auf Knoten: N_rows=5: Reihen bei
72deg-Schritten: 72/144/216/288 + Naht 0/360. Reihenabstand 60mm-Sehne?
5 Reihen: Bogenabstand 60mm, Sehne 2*47.75*sin(36)=56.2mm ~ s. Versatz: ungerade Reihen
um halbe x-Periode: x-Versatz muss Vielfaches von cell sein: s/2=28.1 ~ 30mm (6 Zellen).
-> N_rows=5, s_eff=56.2, Versatz 30mm: Dreieck 60/56/56 nahezu gleichseitig.
Und Dots liegen exakt auf Zellknoten -> mergen korrekt!
"""
import numpy as np
import felupe as fe
import warnings

# --- Konfiguration: N_rows=5 Reihen auf theta-Knoten (72deg), s-Langraster auf x-Knoten
L, W, t = 600.0, 300.0, 0.3
r, w0 = 15.0, 15.0
mu, bulk = 5.0, 50.0
R = W / (2 * np.pi)
N_theta = 50        # cell_theta = 7.2 deg ~ 6mm arc
N_x = 60            # cell_x = 5mm
cell = 5.0
N_rows = 5
th_dots = [2 * np.pi * j / N_rows for j in range(1, N_rows)]     # 72,144,216,288 deg
x_period = (L / 2) / 6        # 50mm: 6 Dot-Spalten im Halbmodell (0,50,...,250+300?)
# Punkt-Spalten: gerade Reihen (j=1: th=72, j=3: th=216): x = m*50 (auf Knoten, i=m*10)
# ungerade (j=2: th=144, j=4: th=288): versetzt um 30mm: x = 30 + m*50 -> i=6+10m
dots = []
for j, thd in enumerate(th_dots, start=1):
    if j % 2 == 1:
        cols = [m * 50.0 for m in range(7)]          # 0..300
    else:
        cols = [30.0 + m * 50.0 for m in range(6)]   # 30..280
    dots += [(xd, thd) for xd in cols if xd <= L / 2 + 1e-9]
print(f"Dots im Halbmodell: {len(dots)} (Voll: ~{2*len(dots) - sum(1 for xd,_ in dots if abs(xd)<1e-9)})")
s_eff = 2 * R * np.sin(np.pi / N_rows)
print(f"Dreiecksseite (Sehne) = {s_eff:.1f} mm | Versatz 30 mm | Ziel war nahezu gleichseitig")

xs = np.linspace(0, L / 2, N_x + 1)
dtheta = 2 * np.pi / N_theta
thetas = np.arange(N_theta) * dtheta
Xg, THg = np.meshgrid(xs, thetas, indexing="ij")
nring = (N_x + 1) * N_theta

def ring_points(rad):
    return np.column_stack([Xg.ravel(order="C"),
                            rad * np.cos(THg).ravel(order="C"),
                            rad * np.sin(THg).ravel(order="C")])

OFF = [0, nring, 2 * nring, 3 * nring]
Ri, Ra = R - w0 / 2, R + w0 / 2
points = np.vstack([ring_points(Ri), ring_points(Ri + t), ring_points(Ra - t), ring_points(Ra)])

def nid(l, i, k):
    return OFF[l] + i * N_theta + (k % N_theta)

def foil_cells(lb, lt):
    return [[nid(lb, i, k), nid(lb, i, k + 1), nid(lb, i + 1, k + 1), nid(lb, i + 1, k),
             nid(lt, i, k), nid(lt, i, k + 1), nid(lt, i + 1, k + 1), nid(lt, i + 1, k)]
            for i in range(N_x) for k in range(N_theta)]

cells = np.array(foil_cells(0, 1) + foil_cells(2, 3), dtype=np.int32)

# Dot-Zellen: Zellmittelpunkt innerhalb r eines Dot-Zentrums (auf Knoten gerastert)
xc = 0.5 * (xs[:-1] + xs[1:])
thc = thetas + 0.5 * dtheta
weld_cell = np.zeros((N_x, N_theta), dtype=bool)
for xd, thd in dots:
    dth = np.angle(np.exp(1j * (thc - thd)))
    weld_cell |= (xc[:, None] - xd) ** 2 + (Ri * dth[None, :]) ** 2 <= r ** 2

weld_node = np.zeros((N_x + 1, N_theta), dtype=bool)
weld_node[:-1, :] |= weld_cell
weld_node[1:, :] |= weld_cell
weld_node[:, 0] = True
weld_node[-1, :] = True
print(f"Schweisszellen: {weld_cell.sum()} | Merge-Knoten: {weld_node.sum()}")

flat = np.arange(nring)
t0_new = np.where(weld_node.ravel(order="C"), OFF[1] + flat, OFF[2] + flat)
n_bot = (N_x) * N_theta
top = cells[n_bot:].copy()
in_t0 = (top >= OFF[2]) & (top < OFF[3])
top[in_t0] = t0_new[top[in_t0] - OFF[2]]
cells = np.vstack([cells[:n_bot], top])

used = np.unique(cells.ravel())
remap = -np.ones(len(points), np.int64); remap[used] = np.arange(len(used))
points = points[used]; cells = remap[cells].astype(np.int32)
print(f"Zellen: {len(cells)} | Knoten: {len(points)}")

mesh = fe.Mesh(points, cells, "hexahedron")
region = fe.RegionHexahedron(mesh)
field = fe.FieldContainer([fe.Field(region, dim=3)])
assert region.dV.min() > 0

rad_pt = np.hypot(points[:, 1], points[:, 2])
mask = (rad_pt > Ri + t * 0.5) & (rad_pt < Ra - t * 0.5)
rp = fe.RegionHexahedronBoundary(mesh, mask=mask)
print("Kavitaetsflaechen:", rp.mesh.ncells)
fp = fe.FieldContainer([fe.Field(rp, dim=3)]); fp.link(field)
solid = fe.SolidBody(fe.NeoHooke(mu=mu, bulk=bulk), field)
load = fe.SolidBodyPressure(fp, pressure=0.0)

boundaries = fe.BoundaryDict()
boundaries["fix-x"] = fe.Boundary(field[0], fx=0.0, skip=(False, True, True))
# Nahtknoten: x=0, theta=0 (y=Ri..Ra-Bereich, z~0)
i_node = np.where(np.isclose(points[:, 0], 0.0) & np.isclose(points[:, 2], 0.0)
                  & (points[:, 1] > Ri - 1) & (points[:, 1] < Ra + 1))[0]
print("Nahtknoten-Kandidaten:", i_node, "->", points[i_node] if len(i_node) else "")
m_uy = np.zeros(len(points), bool); m_uz = np.zeros(len(points), bool)
if len(i_node):
    m_uy[i_node[0]] = True; m_uz[i_node[0]] = True
boundaries["fix-uy"] = fe.Boundary(field[0], mask=m_uy, skip=(True, False, True))
boundaries["fix-uz"] = fe.Boundary(field[0], mask=m_uz, skip=(True, True, False))

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
    x0m = np.isclose(points[:, 0], 0.0)
    rd = np.hypot(points[x0m][:, 1] + u[x0m][:, 1], points[x0m][:, 2] + u[x0m][:, 2])
    print(f"p={p*1e5:6.2f} mbar: r=[{rd.min():.2f},{rd.max():.2f}] umax={np.abs(u).max():.3f}")

u = field[0].values
print("\nfertig. umax:", np.abs(u).max())

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

np.savez_compressed("C:/Users/efame275/femair/luftmatratze_schlauch_v5_result.npz",
                    points=points, cells=cells, u=u, stress=s_t, vm=vm, p_max=p_end,
                    L=L, W=W, t=t, s=s_eff, r=r, R=R, N_rows=N_rows, w0=w0)
print("gespeichert.")
