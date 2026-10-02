"""Luftmatratze v2: zwei Folien mit Punktschweissungen im Dreiecksverband (FElupe).

Modell:
- Zwei flache Folien (je t mm) uebereinander, Ruhezustand: kontaktierend (Spalt = 0).
- Verbunden NUR durch kreisfoermige Schweisspunkte (Radius r), angeordnet im
  gleichseitigen Dreiecksverband (Kantenlaenge s -> jeder Punkt hat 6 aequidistante
  Nachbarn). Schweisspunkte = zusammengefuehrte Knoten der Folien-Innenflaechen.
- Perimeter: durchgehende Schweissnaht (Knoten zusammengefuehrt) -> luftdicht.
- Innendruck p auf die inneren Folienflaechen ausserhalb der Schweisspunkte
  (Folien woelben sich zwischen den Punkten auf = "Pillowing").
- Viertelmodell mit Symmetrie x=0 und y=0 (Muster & Geometrie sind symmetrisch).

FELUPE-BC-SEMANTIK: skip=True bedeutet - diese Komponente wird NICHT vorgeschrieben!
"""
import numpy as np
import felupe as fe

# ------------------------------------------------------------------ Parameter
L      = 600.0   # Laenge (x-Richtung) [mm]  - Beispielgroesse: Reise-Luftmatratze
W      = 300.0   # Breite (y-Richtung) [mm]
t      = 0.3     # Folienstaerke [mm]
s      = 60.0    # Schweisspunkt-Abstand = Dreiecksseite [mm] (6 Nachbarn je Punkt)
r      = 15.0    # Schweisspunkt-Radius [mm]
cell   = 5.0     # Elementkante in der Ebene [mm]
p_max  = 0.008   # Innendruck (Ueberdruck) [MPa] = 0.08 bar
nsteps = 31

mu   = 0.5       # Neo-Hooke Schubmodul [MPa] -> E ~ 1.5 MPa (weiches PVC)
bulk = 10 * mu   # [MPa]

# ------------------------------------------- Viertelmodell-Gitter (x,y >= 0)
xs = np.arange(round(L / 2 / cell) + 1) * cell
ys = np.arange(round(W / 2 / cell) + 1) * cell
nx, ny = len(xs), len(ys)
print(f"Gitter: {nx} x {ny} Knoten in der Ebene, Zellkante {cell} mm")

# Dreiecksgitter: Reihen im Abstand dy = s*sqrt(3)/2, gerade Reihen ohne Versatz,
# ungerade um s/2 versetzt -> gleichseitige Dreiecke, 6 aequidistante Nachbarn.
dy = s * np.sqrt(3) / 2
lattice = []
j = 0
while j * dy <= W / 2 + 1e-9:
    yj = j * dy
    if j % 2 == 0:
        xd = np.arange(0.0, L / 2 + 1e-9, s)
    else:
        xd = np.arange(s / 2, L / 2 + 1e-9, s)
    for x in xd:
        lattice.append((x, yj))
    j += 1
lattice = np.array(lattice)
print("Schweisspunkte im Viertelmodell:", len(lattice),
      "(voll: ~", 4 * len(lattice) - 2 * int((lattice[:, 0] == 0).sum() + (lattice[:, 1] == 0).sum()), ")")

# Schweisszellen: Zellmittelpunkt innerhalb r eines Gitterpunkts
xc = 0.5 * (xs[:-1] + xs[1:])
yc = 0.5 * (ys[:-1] + ys[1:])
CX, CY = np.meshgrid(xc, yc, indexing="ij")
weld_cell = np.zeros(CX.shape, dtype=bool)
for px, py in lattice:
    weld_cell |= (CX - px) ** 2 + (CY - py) ** 2 <= r ** 2
print("Schweisszellen (pixelierte Punkte):", weld_cell.sum(),
      f"| Flaeche pixeliert {weld_cell.sum()*cell*cell:.0f} mm^2 vs. Kreis {np.pi*r*r:.0f} mm^2/Punkt")

# Schweissknoten: Knoten, der eine Schweisszelle beruehrt, oder Perimeter (Randnaht)
weld_node = np.zeros((nx, ny), dtype=bool)
weld_node[:-1, :-1] |= weld_cell
weld_node[1:, :-1] |= weld_cell
weld_node[:-1, 1:] |= weld_cell
weld_node[1:, 1:] |= weld_cell
weld_node[-1, :] = True   # Randnaht x = L/2
weld_node[:, -1] = True   # Randnaht y = W/2

# ------------------------------------------------------------------- Mesh-Aufbau
# 4 Knotenebenen: B0 (z=-t), B1 (z=0, Unterfolie innen), T0 (z=0, Oberfolie innen),
# T1 (z=+t). B1 und T0 sind DOPPELT (getrennte Folien) - ausser an Schweisspunkten.
def plane(z):
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([X.ravel(order="C"), Y.ravel(order="C"), np.full(X.size, z)])
    return pts  # id(i,j) = i*ny + j

pB0, pB1, pT0, pT1 = plane(-t), plane(0.0), plane(0.0), plane(t)
OFF0, OFF1 = 0, nx * ny
OFF2, OFF3 = 2 * nx * ny, 3 * nx * ny
points = np.vstack([pB0, pB1, pT0, pT1])

def cells_between(off_a, off_b):
    cl = []
    for i in range(nx - 1):
        for jj in range(ny - 1):
            a = off_a + i * ny + jj
            b = off_b + i * ny + jj
            cl.append([a, a + ny, a + ny + 1, a + 1,      # (i,j),(i+1,j),(i+1,j+1),(i,j+1)
                       b, b + ny, b + ny + 1, b + 1])
    return cl

cells = cells_between(OFF0, OFF1) + cells_between(OFF2, OFF3)
cells = np.array(cells, dtype=np.int32)

# Schweisspunkte: NUR die T0-Knoten auf Schweisszellen -> B1-Knoten (Folien verbunden)
i_idx, j_idx = np.unravel_index(np.arange(nx * ny), (nx, ny))
merge_mask = weld_node[i_idx, j_idx].ravel(order="C")
flat = np.arange(nx * ny)
t0_new = np.where(merge_mask, OFF1 + flat, OFF2 + flat)  # merged -> B1, sonst bleibt T0
top = cells[len(cells_between(OFF0, OFF1)):].copy()      # Oberfolien-Zellen
in_t0 = (top >= OFF2) & (top < OFF3)
top[in_t0] = t0_new[top[in_t0] - OFF2]
cells = np.vstack([cells[: len(cells) - len(top)], top])

# unbenutzte Knoten entfernen (T0-Knoten an Schweisspunkten sind jetzt B1)
used = np.unique(cells.ravel())
remap = -np.ones(len(points), dtype=np.int64)
remap[used] = np.arange(len(used), dtype=np.int64)
points = points[used]
cells = remap[cells].astype(np.int32)
n_merged = int(merge_mask.sum())
print(f"Zellen: {len(cells)} (2 Folienlagen) | Knoten: {len(points)} | "
      f"zusammengefuehrte Schweissknoten: {n_merged}")

mesh = fe.Mesh(points, cells, "hexahedron")
region = fe.RegionHexahedron(mesh)
field = fe.FieldContainer([fe.Field(region, dim=3)])
if region.dV.min() < 0:
    raise RuntimeError("Negative Elementvolumina - Zell-Ordering pruefen!")

# ------------------------------------------- Druckflaeche = Kavitaet (z = 0)
# Alle Grenzflaechen auf z=0; Schweissflaechen sind automatisch "interior"
# (von Zellen beider Folien geteilt) und fallen bei only_surface weg.
mask_z0 = np.isclose(points[:, 2], 0.0)
region_pressure = fe.RegionHexahedronBoundary(mesh, mask=mask_z0)
print("Kavitaets-Grenzflaechen:", region_pressure.mesh.ncells)
field_pressure = fe.FieldContainer([fe.Field(region_pressure, dim=3)])
field_pressure.link(field)

solid = fe.SolidBody(fe.NeoHooke(mu=mu, bulk=bulk), field)
load = fe.SolidBodyPressure(field_pressure, pressure=0.0)

# ------------------------------------------------------------- Randbedingungen
# Symmetrie x=0 (nur ux), Symmetrie y=0 (nur uy), ein einzelner uz-Punkt.
boundaries = fe.BoundaryDict()
boundaries["fix-x"] = fe.Boundary(field[0], fx=0.0, skip=(False, True, True))
boundaries["fix-y"] = fe.Boundary(field[0], fy=0.0, skip=(True, False, True))
# Ein einziger uz-Punkt (Ecke x=y=0, Oberfolie) verhindert die z-Rigid-Body-Mode;
# das Kissenvolumen wird zusaetzlich von der Perimeternaht gehalten.
m_uz = np.zeros(len(points), dtype=bool)
j_uz_single = np.where(np.isclose(points[:, 0], 0.0) & np.isclose(points[:, 1], 0.0)
                       & np.isclose(points[:, 2], t))[0]
m_uz[j_uz_single[0]] = True
boundaries["fix-uz"] = fe.Boundary(field[0], mask=m_uz, skip=(True, True, False))
print("BC: ux=0:", len(boundaries["fix-x"].points), "Knoten | uy=0:", len(boundaries["fix-y"].points),
      "| uz fixiert: 1 Knoten")

# --------------------------------------------------------------------- Rechnung
pressure_steps = p_max * np.linspace(0.0, 1.0, nsteps) ** 2
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
    "C:/Users/efame275/femair/luftmatratze_punkt_result.npz",
    points=points, cells=cells, u=u, stress=s_t, vm=vm, p_max=p_max,
    L=L, W=W, t=t, s=s, r=r,
)
print("Ergebnisse gespeichert.")
