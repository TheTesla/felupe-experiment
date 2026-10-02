"""3D-Visualisierung der Punktschweiss-Luftmatratze (v2)."""
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
data = np.load("C:/Users/efame275/femair/luftmatratze_punkt_result.npz")
points, cells, u, vm = data["points"], data["cells"], data["u"], data["vm"]
p_max, L, W, t, s, r = [float(data[k]) for k in ("p_max", "L", "W", "t", "s", "r")]

cell_types = np.full(len(cells), pv.CellType.HEXAHEDRON, dtype=np.uint8)
conn = np.hstack([np.full((len(cells), 1), 8, dtype=np.int64), cells.astype(np.int64)]).ravel()
g0 = pv.UnstructuredGrid(conn, cell_types, points)

scale = 2.0
g = pv.UnstructuredGrid(conn, cell_types, points + scale * u)
g.point_data["von_mises"] = vm
g.point_data["disp"] = np.linalg.norm(u, axis=1)
print("grid:", g)

sargs = dict(title="von-Mises [MPa]", vertical=True, title_font_size=13, label_font_size=10)
txt = (f"Luftmatratze {L:.0f} x {W:.0f} mm, 2 Folien t={t} mm, Schweisspunkte r={r:.0f} mm im "
       f"Dreiecksverband s={s:.0f} mm\np = {p_max:.3f} MPa ({p_max*10:.2f} bar) | "
       f"max. Verschiebung {np.abs(u).max():.1f} mm | max. von Mises {vm.max():.2f} MPa\n"
       f"Verformung x{scale:.0f} ueberhoeht")

# Draufsicht (Deckfolie mit Dots sichtbar)
pl = pv.Plotter(off_screen=True, window_size=(1600, 1000))
pl.add_mesh(g, scalars="von_mises", cmap="turbo", scalar_bar_args=sargs, show_edges=False)
pl.add_text(txt, font_size=10, color="black")
pl.set_background("white")
pl.camera_position = "xy"
pl.camera.zoom(1.25)
pl.screenshot("C:/Users/efame275/femair/punkt_draufsicht.png")
print("draufsicht ok")

# Isometrie
pl2 = pv.Plotter(off_screen=True, window_size=(1600, 1000))
pl2.add_mesh(g, scalars="von_mises", cmap="turbo", scalar_bar_args=sargs)
pl2.add_text(txt, font_size=10, color="black")
pl2.set_background("white")
pl2.camera_position = "iso"
pl2.camera.zoom(1.25)
pl2.screenshot("C:/Users/efame275/femair/punkt_iso.png")
print("iso ok")

# Schnitt halber y: Pillowing sichtbar -> matplotlib (pyvista.slice leer durch Doppelfolien-Spalt)
# (profil_cut.py erzeugt punkt_schnitt.png aus denselben Daten)

g.save("C:/Users/efame275/femair/luftmatratze_punkt_verformt.vtu")
print("vtu ok")
