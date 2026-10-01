"""3D-Visualisierung: verformte Luftmatratze, farbcodiert nach von-Mises-Spannung."""
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

data = np.load("C:/Users/efame275/femair/luftmatratze_result.npz")
points = data["points"]
cells = data["cells"]
u = data["u"]
vm = data["vm"]
p_max = float(data["p_max"])

# UnstructuredGrid aus Hexaeder-Zellen (Zellen-Node-Ordering an felupe anpassen)
# felupe hex: (z-k, x-i, y-j) => id = i*ny*nz + j*nz + k ; pyvista hexa expected order:
# standard hexa: nodes of bottom face then top face, counterclockwise -> same layout
# (volumetrische Reihenfolge ist kompatibel, VTK hexahedron: 0-3 bottom CCW, 4-7 top CCW)
ny = int(round(max(points[:, 1])) - min(points[:, 1])) + 1  # not robust -> read from npz? recompute below
nz = 9
# cells were built with [n0, n0+ny*nz, n0+ny*nz+nz, n0+nz, n0+1, ...]
# => node order: (i,j,k),(i+1,j,k),(i+1,j+1,k),(i,j+1,k) bottom; same +1 in k top
# VTK hexahedron expects: bottom face quad (0,1,2,3) CCW viewed from below, top (4,5,6,7)
# our bottom: n0(x,y,k), n0+x(n+1 in x), ... -> consistent quad ordering
# connectivity array im VTK-Format aufbauen: [8, n0..n7] pro Zelle
cell_types = np.full(len(cells), pv.CellType.HEXAHEDRON, dtype=np.uint8)
conn = np.hstack([np.full((len(cells), 1), 8, dtype=np.int64), cells.astype(np.int64)]).ravel()
grid = pv.UnstructuredGrid(conn, cell_types, points)
# vm ist auf Punkten projiziert -> point_data
grid.point_data["von_mises"] = vm
grid.point_data["disp_mag"] = np.linalg.norm(u, axis=1)

print("grid:", grid)

# verformte Geometrie
deformed = grid.copy()
deformed.points = points + u

# Skalierung der Verschiebung fuer Visualisierung
scale = 3.0
warped = grid.copy()
warped.points = points + scale * u
warped.point_data["disp_mag"] = np.linalg.norm(scale * u, axis=1)

sargs = dict(title="von-Mises-Spannung [MPa]", vertical=True, title_font_size=14, label_font_size=11)

# ---- Plot 1: verformter Koerper, von Mises auf Oberflaeche
pl = pv.Plotter(off_screen=True, window_size=(1600, 1000))
pl.add_mesh(warped, scalars="von_mises", cmap="turbo", show_scalar_bar=True,
            scalar_bar_args=sargs, smooth_shading=False, show_edges=False)
pl.add_text(f"Luftmatratze, Innendruck p = {p_max:.2f} MPa ({p_max*10:.1f} bar)\n"
            f"max. Verschiebung {np.abs(u).max():.1f} mm  |  max. von Mises {vm.max():.3f} MPa",
            font_size=11, color="black")
pl.set_background("white")
pl.camera_position = "xz"
pl.camera.zoom(1.4)
pl.screenshot("C:/Users/efame275/femair/luftmatratze_vonmises.png")
print("plot 1 ok")

# ---- Plot 2: Verschiebungsbetrag
pl2 = pv.Plotter(off_screen=True, window_size=(1600, 1000))
pl2.add_mesh(warped, scalars="disp_mag", cmap="viridis", show_scalar_bar=True,
             scalar_bar_args=dict(title="Verschiebung [mm]", vertical=True))
pl2.add_text(f"Verschiebungsbetrag (ueberhoeht, Faktor {scale})", font_size=12, color="black")
pl2.set_background("white")
pl2.camera_position = "xz"
pl2.camera.zoom(1.4)
pl2.screenshot("C:/Users/efame275/femair/luftmatratze_verschiebung.png")
print("plot 2 ok")

# ---- VTU export fuer Paraview/PyVista
warped.save("C:/Users/efame275/femair/luftmatratze_verformt.vtu")
deformed.save("C:/Users/efame275/femair/luftmatratze_verformt_echt.vtu")
print("vtu ok")
