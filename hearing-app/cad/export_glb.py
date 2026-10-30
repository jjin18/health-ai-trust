import bpy,json
from pathlib import Path
r=Path(__file__).resolve().parent
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
for entry in json.loads((r/'parts.json').read_text()):
 bpy.ops.wm.stl_import(filepath=str(r/entry['file']))
 obj=bpy.context.object;obj.name=entry['name'];obj.location.z=-12
 obj.scale=(.001,.001,.001)
 # Center offsets follow the model's metres conversion.
 obj.location.z=-.012
 mat=bpy.data.materials.new(entry['name']);mat.diffuse_color=(*entry['color'],1);mat.use_nodes=True
 bs=mat.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(*entry['color'],1);bs.inputs['Metallic'].default_value=.85 if ('grille' in obj.name or 'faceplate' in obj.name) else .05;bs.inputs['Roughness'].default_value=.27 if 'Silicone' not in obj.name else .5
 obj.data.materials.append(mat)
 for poly in obj.data.polygons:poly.use_smooth=True
bpy.ops.export_scene.gltf(filepath=str(r/'hush-in-ear.glb'),export_format='GLB')
