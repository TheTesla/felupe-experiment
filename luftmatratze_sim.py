"""Einfache Luftmatratze: hohler Gummikasten (Luftkissen), Innendruck, FElupe."""
import numpy as np
import felupe as fe

# ---------------------------------------------------------------- Geometrie (mm)
Lx, Ly, Lz = 200.0, 100.0, 40.0          # Aussenmass des Luftkissens
nx, ny, nz = 41, 21, 13                  # Knoten je Richtung (nz feiner: Wandprofil)
xs = np.linspace(-Lx / 2, Lx / 2, nx)
ys = np.linspace(-Ly / 2, Ly / 2, ny)
zs = np.linspace(-Lz / 2, Lz / 2, nz)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
points = np.column_stack([X.ravel(order="C"), Y.ravel(order="C"), Z.ravel(order="C")])

ncx, ncy, ncz = nx - 1, ny - 1, nz - 1
# Knoten-Index bei meshgrid(indexing="ij") + ravel("C"):  id = i*ny*nz + j*nz + k
cells = []
for k in range(ncz):
    for j in range(ncy):
        for i in range(ncx):
            if i in (0, ncx - 1) or j in (0, ncy - 1) or k in (0, ncz - 1):
                n0 = i * ny * nz + j * nz + k
                cells.append([
                    n0, n0 + ny * nz, n0 + ny * nz + nz, n0 + nz,          # Boden (z)
                    n0 + 1, n0 + ny * nz + 1, n0 + ny * nz + nz + 1, n0 + nz + 1,  # Deckel (z)
                ])
cells = np.array(cells, dtype=np.int32)

# Nur von Zellen genutzte Knoten behalten: innere Gitterpunkte ohne Element
# (Hohlraum) wuerden sonst ungebundene DOFs ohne Steifigkeit erzeugen.
used = np.unique(cells.ravel())
remap = -np.ones(len(points), dtype=np.int64)
remap[used] = np.arange(len(used), dtype=np.int64)
points = points[used]
cells = remap[cells].astype(np.int32)
print("Zellen (Wandschale):", len(cells), "| Knoten (genutzt):", len(points))

mesh = fe.Mesh(points, cells, "hexahedron")

# ---------------------------------------------------- Regionen, Feld, Material
region = fe.RegionHexahedron(mesh)
field = fe.FieldContainer([fe.Field(region, dim=3)])

# Innenseite (Kavitaetsoberflaeche) = alle Knoten strikt innerhalb der Aussenwand
t = xs[1] - xs[0]                        # Wandstaerke = 1 Zelle
mask = (
    (np.abs(points[:, 0]) < Lx / 2 - 1e-3)
    & (np.abs(points[:, 1]) < Ly / 2 - 1e-3)
    & (np.abs(points[:, 2]) < Lz / 2 - 1e-3)
)
print("Knoten auf Kavitaetsoberflaeche:", mask.sum())

region_pressure = fe.RegionHexahedronBoundary(mesh, mask=mask)
field_pressure = fe.FieldContainer([fe.Field(region_pressure, dim=3)])
field_pressure.link(field)

# Gummimembran (PVC-uehnlich weich, damit die Aufblähung sichtbar wird)
mu = 0.17          # MPa  (E ~ 0.5 MPa)
bulk = 10 * mu     # MPa
umat = fe.NeoHooke(mu=mu, bulk=bulk)
solid = fe.SolidBody(umat, field)
load = fe.SolidBodyPressure(field_pressure, pressure=0.0)

# --------------------------------------------------------------- Randbedingungen
# Symmetrie-Ebenen in der Modellmitte: je eine Verschiebekomponente fixiert.
# ACHTUNG FElupe-Semantik: skip=True => diese Komponente wird NICHT vorgeschrieben!
boundaries = fe.BoundaryDict()
boundaries["fix-x"] = fe.Boundary(field[0], fx=0.0, skip=(False, True, True))   # ux auf x=0
boundaries["fix-y"] = fe.Boundary(field[0], fy=0.0, skip=(True, False, True))   # uy auf y=0
boundaries["fix-z"] = fe.Boundary(field[0], fz=0.0, skip=(True, True, False))   # uz auf z=0

# --------------------------------------------------------------------- Rechnung
p_max = 0.008  # MPa = 0.08 bar Ueberdruck
# Quadratische Lastschrittfolge: fein am Anfang (flache Membran = lastempfindlich)
pressure_steps = p_max * np.linspace(0.0, 1.0, 61) ** 2
pressure_steps[-1] = p_max
step = fe.Step(items=[solid, load], ramp={load: pressure_steps}, boundaries=boundaries)
job = fe.Job(steps=[step])


def cb(dx, x, iteration, xnorm, fnorm, success):
    print(f"  iter {iteration:2d}: |r|={fnorm:.3e} | umax={np.abs(field[0].values).max():.3f} mm")


job.evaluate(callback=cb)

u = field[0].values
print("\nfertig. max. Verschiebung:", np.abs(u).max(), "mm")

# -------------------------------------------------------------- Spannungen (Cauchy)
F = field[0].extract(grad=True)                       # (3, 3, nquad, nelems)
P = solid.umat.gradient([F, solid.results.statevars])[0]
J = np.linalg.det(np.transpose(F, (2, 3, 0, 1)))
Fq = np.transpose(F, (2, 3, 0, 1))                    # (nquad, nelems, 3, 3)
Pq = np.transpose(P, (2, 3, 0, 1))
cauchy = np.einsum("qcij,qcjk->qcik", Pq, Fq.transpose(0, 1, 3, 2)) / J[..., None, None]
# fe.project erwartet (tensor..., nquad, nelems)
stress_projected = fe.project(cauchy.transpose(2, 3, 0, 1), region)  # (npoints, 3, 3)

# von Mises
s = stress_projected
vm = np.sqrt(0.5 * (
    (s[:, 0, 0] - s[:, 1, 1]) ** 2 + (s[:, 1, 1] - s[:, 2, 2]) ** 2 + (s[:, 2, 2] - s[:, 0, 0]) ** 2
) + 3 * (s[:, 0, 1] ** 2 + s[:, 1, 2] ** 2 + s[:, 2, 0] ** 2))
print("max. von-Mises-Spannung:", vm.max(), "MPa")

np.savez_compressed(
    "C:/Users/efame275/femair/luftmatratze_result.npz",
    points=mesh.points, cells=cells, u=u, stress=s, vm=vm, p_max=p_max,
)
print("Ergebnisse gespeichert.")
