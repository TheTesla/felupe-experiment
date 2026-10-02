"""Visualisierung Schlauch-Ergebnis (p=0.000115 Stand im npz? - nein, lade aus npz nur wenn vorhanden).
Erzeuge Bilder aus luftmatratze_schlauch_result.npz (letzter gespeicherter Stand)."""
import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True

data = np.load("C:/Users/efame275/femair/luftmatratze_schlauch_result.npz")
points, cells, u, vm = data["points"], data["cells"], data["u"], data["vm"]
p_max, L, W, t, s, r, R, N_rows = [float(data[k]) for k in
                                   ("p_max", "L", "W", "t", "s", "r", "R", "N_rows")]

cell_types = np.full(len(cells), pv.CellType.HEXAHEDRON, dtype=np.uint8)
conn = np.hstack([np.full((len(cells), 1), 8, dtype=np.int64), cells.astype(np.int64)]).ravel()
scale = 4.0
g = pv.UnstructuredGrid(conn, cell_types, points + scale * u)
g.point_data["von_mises"] = vm
print("grid:", g)

txt = (f"Luftschlauch: 2 Folien verklebt an den Langkanten, {L:.0f} mm lang, Umfang {W:.0f} mm "
       f"(R={R:.1f} mm), t={t} mm\nSchweisspunkte r={r:.0f} mm im Dreiecksverband "
       f"s={s:.1f} mm ({N_rows:.0f} Reihen) | p={p_max*1000:.3f} mPa "
       f"({p_max*1000:.2f} mbar) | umax={np.abs(u).max():.2f} mm | vm={vm.max():.3f} MPa\n"
       f"Verformung x{scale:.0f} ueberhoeht")

sargs = dict(title="von-Mises [MPa]", vertical=True, title_font_size=12, label_font_size=10)
pl = pv.Plotter(off_screen=True, window_size=(1600, 900))
pl.add_mesh(g, scalars="von_mises", cmap="turbo", scalar_bar_args=sargs)
pl.add_text(txt, font_size=10, color="black")
pl.set_background("white")
pl.camera_position = "iso"
pl.camera.zoom(1.3)
pl.screenshot("C:/Users/efame275/femair/schlauch_iso.png")
print("iso ok")

# Endansicht (Querschnitt, Pillowing sichtbar)
pl2 = pv.Plotter(off_screen=True, window_size=(1000, 1000))
pl2.add_mesh(g, scalars="von_mises", cmap="turbo", scalar_bar_args=sargs)
pl2.add_text(f"Querschnitt (x-Richtung), Verformung x{scale:.0f}", font_size=11, color="black")
pl2.set_background("white")
pl2.camera_position = "yx"
pl2.camera.zoom(1.6)
pl2.screenshot("C:/Users/efame275/femair/schlauch_querschnitt.png")
print("querschnitt ok")

g.save("C:/Users/efame275/femair/luftmatratze_schlauch_verformt.vtu")
print("vtu ok")
