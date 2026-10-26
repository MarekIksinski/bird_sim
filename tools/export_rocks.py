#!/usr/bin/env python3
"""Export each Rock* mesh from assets/Rocks.blend to assets/rocks/rock_NN.glb
(one GLB per variant, transforms applied, Z-up -> Y-up done by the glTF
exporter). Run: blender --background --python tools/export_rocks.py
Committed outputs: assets/rocks/*.glb — re-run only to regenerate.
"""
import bpy
from pathlib import Path

here = Path(__file__).resolve().parent.parent
blend = here / 'assets' / 'Rocks.blend'
outdir = here / 'assets' / 'rocks'
outdir.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=str(blend))

rocks = sorted([o for o in bpy.data.objects
                if o.type == 'MESH' and o.name.lower().startswith('rock')],
               key=lambda o: o.name)
for i, obj in enumerate(rocks, start=1):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.location = (0.0, 0.0, 0.0)   # drop the authoring layout offset:
                                     # each variant pivots at its own origin
    out = outdir / f'rock_{i:02d}.glb'
    bpy.ops.export_scene.gltf(
        filepath=str(out),
        export_format='GLB',
        use_selection=True,
        export_apply=True,      # bake object scale/rotation into mesh
        export_yup=True,        # glTF convention; ursina/panda flips to Y-up
    )
    print(f'wrote {out.name}: {len(obj.data.polygons)} polys')
print(f'done: {len(rocks)} rock variants')
