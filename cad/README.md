# Hush in-ear visual concept

Editable solid CAD created with CadQuery 2.7. Units: millimetres. Approximate envelope: 15 × 19 × 30 mm.

- `hush-in-ear.step`: seven named solid components for CAD tools.
- `hush-in-ear.stl`: combined geometry for mesh tools.
- `hush-in-ear.glb`: materials and geometry for web/3D tools, in metres.
- `build_model.py`: editable dimensions, loft sections, H grille and a single smooth silicone dome. Run with Python and CadQuery installed.
- `export_glb.py`: Blender conversion script; run after building CAD.

The website uses tessellated CAD with a dependency-free WebGL viewer, draggable and keyboard-rotatable. A still render is the fallback when WebGL is unavailable.

This is an illustrative product concept, not production engineering. It does not establish fit, acoustic performance, component packaging, vent design, sealing, battery safety, tolerances, biocompatibility, or manufacturability. The shell is represented as a solid exterior envelope, not a validated hollow enclosure. The H grille is decorative concept geometry.
