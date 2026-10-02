# felupe-experiment — Luftmatratzen-FEM-Simulation

FEM-Simulation aufblasbarer Luftmatratzen mit [FElupe](https://github.com/adtzlr/felupe)
und 3D-Ergebnisvisualisierung mit PyVista.

## v2 — Punktschweiß-Kissen (aktuell, `luftmatratze_punktschweiss.py`)

Zwei dünne Folien, verbunden **nur** durch kreisförmige Schweißpunkte im
**gleichseitigen Dreiecksverband** (jeder Punkt hat 6 äquidistante Nachbarn im
Abstand `s`). Ein Perimeter-Schweißband schließt das Kissen luftdicht.
Der Innendruck wölbt die Folien zwischen den Punkten auf (**Pillowing**).

### Parameter (Skriptkopf)

| Parameter | Bedeutung | Beispielwert |
|---|---|---|
| `L` | Länge (x) [mm] | 600 |
| `W` | Breite (y) [mm] | 300 |
| `t` | Folienstärke [mm] | 0,3 |
| `s` | Schweißpunkt-Abstand = Dreiecksseite [mm] | 60 |
| `r` | Schweißpunkt-Radius [mm] | 15 |
| `cell` | Elementkante in der Ebene [mm] | 5 |
| `p_max` | Innendruck [MPa] | 0,008 (= 0,08 bar) |

### Modellierung

- Viertelmodell (Symmetrie x=0, y=0), zwei Hexaeder-Lagen (jeweils 1 Element dick)
- Schweißpunkte: Knoten der Folien-Innenflächen werden an den Punktabdrücken
  **zusammengeführt** (echte Punktverbindung, keine Kontakt-Formulierung)
- Perimeter: alle Randknoten zusammengeführt → geschlossene Naht
- Druck: `RegionHexahedronBoundary` auf die Kavität z=0 (die Schweißflächen sind
  dort *interior* und fallen automatisch aus der Grenzfläche heraus)
- Material: Neo-Hooke, `mu = 0.5 MPa` (E ≈ 1,5 MPa, weiches PVC)
- BC-ACHTUNG: FElupe `skip=True` = Komponente wird **nicht** vorgeschrieben

### Ergebnisse (Beispielwerte)

| Größe | Wert |
|---|---|
| max. Verschiebung (Pillowing) | ≈ 20,1 mm (Feldmitte; Deck nach oben, Boden nach unten) |
| max. von-Mises | ≈ 1,28 MPa — **Spannungsring am Rand jedes Schweißpunkts** |
| Schweißpunkte (Vollmodell) | ~52 im Dreiecksverband |
| konvergierte Lastschritte | 31/31 |

Plausibilitätscheck: Membranspannung σ ≈ p·R/(2t) mit Feldkrümmungsradius R ≈ 30 mm
ergibt ~0,4 MPa Feldspannung; Faktor 2–3 Spannungskonzentration am Punktrand →
Spitzenwert ~1,3 MPa ✓.

### Bilder (v2)

| Datei | Inhalt |
|---|---|
| `punkt_draufsicht.png` | Kissen von oben: Dot-Muster mit Spannungsringen |
| `punkt_iso.png` | Isometrie: Pillowing-Wülste, versetzte Punktreihen |
| `punkt_schnitt.png` | Schnitt y=0: Deck wölbt nach oben, Boden nach unten |
| `luftmatratze_punkt_verformt.vtu` | ParaVieW-Datensatz (von_mises, disp) |
| `luftmatratze_punkt_result.npz` | Rohdaten (inkl. L, W, t, s, r) |

## v1 — Geschlossenes Luftkissen (`luftmatratze_sim.py`)

Hohler Gummikasten (200 × 100 × 40 mm, Wand 5 mm) als Referenzmodell mit
durchgehender Kavität. Ergebnisse: 31 mm Aufblähung, 0,17 MPa bei 0,08 bar;
Schnitte in `luftmatratze_schnitt.png`, drehbare Ansicht in
`luftmatratze_interaktiv.html`.

## Ausführen

```bash
pip install felupe pyvista matplotlib
python luftmatratze_punktschweiss.py   # ca. 5–8 min (Viertelmodell, 3600 Zellen)
python visualize_punktschweiss.py
```

## Fallstricke (experimentell verifiziert)

1. **FElupe `Boundary(skip=...)` ist invers**: `skip=True` heißt — Komponente
   wird *nicht* vorgeschrieben (`felupe/dof/_boundary.py`, `apply_mask`).
2. **Knoten-Merging für Schweißpunkte**: Die Remap-Tabelle muss die *nicht*
   verschweißten Knoten auf sich selbst abbilden — sonst verschmelzen alle
   Folienknoten (Kavität leer → Druckfläche 0 → u ≡ 0; der Solver "konvergiert"
   dann mit der Nulllösung).
3. **Kavität mit `RegionHexahedronBoundary(mask=...)`**: Die Maske wählt Knoten;
   nur Facetten, die ausschließlich aus maskierten Knoten bestehen, werden
   Grenzflächen. Verbundene Stellen (Schweißpunkte) fallen automatisch weg.
4. **Flache Membranen**: quadratische Lastschrittfolge `p·linspace²` verwenden —
   äquidistante Schritte divergieren im ersten Aufblähübergang.
5. **Knoten ohne Elemente** fixiert FElupe zwar automatisch
   (`points_without_cells` → `dof0`), aber Ergebnisfelder enthalten dann
   Nullwert-Inseln → im Mesh direkt entfernen.
