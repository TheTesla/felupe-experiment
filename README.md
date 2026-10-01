# felupe-experiment — Luftmatratzen-FEM-Simulation

Einfache FEM-Simulation einer aufblasbaren Luftmatratze (Luftkissen) mit
[FElupe](https://github.com/adtzlr/felupe) inklusive 3D-Ergebnisvisualisierung mit PyVista.

## Modell

- Hohler Gummikasten 200 × 100 × 40 mm, Wandstärke 5 mm (Volumenschale aus Hexaeder-Elementen)
- Material: Neo-Hooke (`mu = 0.17 MPa`, `bulk = 1.7 MPa`, E ≈ 0,5 MPa — gummielastisch/PVC-ähnlich)
- Belastung: Innendruck 0,08 bar (0,008 MPa), auf die Innenseite der Wandschale (`SolidBodyPressure`)
- Randbedingungen: drei Symmetrie-Ebenen in der Modellmitte (x=0, y=0, z=0, je eine Komponente fixiert)
- Laststeuerung: quadratische Lastschrittfolge (fein am Anfang — die flache Membran ist lastempfindlich)

## Ergebnisse (p = 0,08 bar)

| Größe | Wert |
|---|---|
| max. Verschiebung (Kissenaufblähung) | ≈ 31,1 mm |
| max. von-Mises-Spannung | ≈ 0,170 MPa |
| konvergierte Lastschritte | 61/61 |

Die Deck-/Bodenmembran beult dabei freies auf (Maximum exakt in der Modellmitte,
glatter symmetrischer Bogen — verifiziert per Schnittkurven), die schmalen
Seitenwände bleiben nahezu formstabil.

## Dateien

| Datei | Inhalt |
|---|---|
| `luftmatratze_sim.py` | FEM-Simulation (FElupe), schreibt `luftmatratze_result.npz` |
| `visualize_luftmatratze.py` | 3D-Renderings + VTU-Export (PyVista) |
| `luftmatratze_vonmises.png` | verformte Matratze, von-Mises-Spannung (Verformung ×3) |
| `luftmatratze_draufsicht.png` | Draufsicht |
| `luftmatratze_schnitt.png` | Längs-/Querschnitt in echtem Maßstab (matplotlib) |
| `luftmatratze_seitenansicht.png` | Seitenansicht in echtem Maßstab |
| `luftmatratze_interaktiv.html` | drehbare 3D-Ansicht (vtk.js) — einfach im Browser öffnen |
| `luftmatratze_verformt.vtu` | ParaVieW/PyVista-Datensatz (von_mises, disp_mag) |
| `luftmatratze_result.npz` | Rohdaten (Punkte, Zellen, u, Spannungen) |

## Ausführen

```bash
pip install felupe pyvista
python luftmatratze_sim.py        # ca. 3–5 min
python visualize_luftmatratze.py
```

## Anmerkungen / Fallstricke

- **FElupe `Boundary(skip=...)` ist invers** zu intuitiver Erwartung:
  `skip=True` heißt — diese Komponente wird *nicht* vorgeschrieben
  (Quelle `felupe/dof/_boundary.py`, `apply_mask`).
- Innendruck auf eine Kavität: `RegionHexahedronBoundary(mesh, mask=...)`
  mit Maske „Knoten strikt innerhalb der Außenmaße"; die Maske muss Knoten
  von Elementen *auf* der Fläche wählen — Feinheit: nur so landet die
  Kavitätsoberfläche in der Grenzregion.
- Knoten ohne Elemente (Hohlraum eines Schalenmodells) fixiert FElupe
  automatisch (`mesh.points_without_cells` → `dof0`); wir entfernen sie
  trotzdem explizit, damit Ergebnisfelder keine Nullwert-Inseln haben.
- flache Membran + Enddruck: feine (quadratische) Lastschritte am Anfang nötig,
  sonst divergiert der Newton-Solver im ersten Aufblähübergang.
