"""3D-Visualisierung des Spalt-Schlauchs (starkes Pillowing) - iso + querschnitt."""
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
d = np.load("C:/Users/efame275/femair/luftmatratze_schlauch_spalt_result.npz")
points, cells, u, vm = d["points"], d["cells"], d["u"], d["vm"]
p_max, L, W, t, s, r, R, w0 = [float(d[k]) for k in ("p_max", "L", "W", "t", "s", "r", "R", "w0")]

cell_types = np.full(len(cells), pv.CellType.HEXAHEDRON, dtype=np.uint8)
conn = np.hstack([np.full((len(cells), 1), 8, dtype=np.int64), cells.astype(np.int64)]).ravel()
scale = 3.0
g = pv.UnstructuredGrid(conn, cell_types, points + scale * u)
g.point_data["von_mises"] = vm

txt = (f"Luftschlauch mit Luftkammer (Anfangsspalt w0={w0:.0f} mm): 2 Folien an den Langkanten\n"
       f"verklebt, Schweisspunkte r={r:.0f} mm im Dreiecksverband s={s:.1f} mm\n"
       f"p = {p_max*1e5:.2f} mbar = {p_max*1e2:.0f} kPa = {p_max:.4f} bar | umax={np.abs(u).max():.2f} mm | "
       f"vm={vm.max():.3f} MPa | Verformung x{scale:.0f} ueberhoeht")

sargs = dict(title="von-Mises [MPa]", vertical=True, title_font_size=12, label_font_size=10)
pl = pv.Plotter(off_screen=True, window_size=(1600, 900))
pl.add_mesh(g, scalars="von_mises", cmap="turbo", scalar_bar_args=sargs)
pl.add_text(txt, font_size=10, color="black")
pl.set_background("white")
pl.camera_position = "iso"
pl.camera.zoom(1.3)
pl.screenshot("C:/Users/efame275/femair/schlauch_spalt_iso.png")
print("iso ok")

# Endansicht auf Querschnitt
pl2 = pv.Plotter(off_screen=True, window_size=(900, 900))
pl2.add_mesh(g, scalars="von_mises", cmap="turbo", scalar_bar_args=sargs)
pl2.add_text(f"Querschnitt, p={p_max*1e5:.0f} mbar, x{scale:.0f}", font_size=11, color="black")
pl2.set_background("white")
pl2.camera_position = "yx"
pl2.camera.zoom(1.6)
pl2.screenshot("C:/Users/efame275/femair/schlauch_spalt_endansicht.png")
print("endansicht ok")

g.save("C:/Users/efame275/femair/luftmatratze_schlauch_spalt_verformt.vtu")
print("vtu ok")
