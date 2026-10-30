"""Hush visual concept. Millimetres. Run with CadQuery 2.7.
Not an engineered acoustic, electronic, fit-tested or production enclosure.
"""
from pathlib import Path
import json, math
import cadquery as cq
OUT=Path(__file__).resolve().parent
parts=[]
def add(name,shape,color):
 parts.append((name,shape,color));return shape
# Soft asymmetric oval chassis, lofted through editable cross sections.
shell=cq.Workplane('XY').ellipse(6.2,8.2).workplane(offset=5).ellipse(7.4,9.2).workplane(offset=6).center(.5,0).ellipse(6.3,7.4).workplane(offset=6).ellipse(4.1,4.8).loft(combine=True)
add('Translucent shell',shell,(.79,.72,.62))
face=cq.Workplane('XY').ellipse(6.15,8.15).extrude(-.9)
# A real through-hole for microphone access in the concept faceplate.
face=face.cut(cq.Workplane('XY').center(-3.7,4.2).circle(.65).extrude(-2))
add('Champagne faceplate',face,(.73,.57,.39))
# Raised physical H grille: two rails joined by a bridge.
left=cq.Workplane('XY').center(-1.8,0).rect(1.15,7.5).extrude(-1.6)
right=cq.Workplane('XY').center(1.8,0).rect(1.15,7.5).extrude(-1.6)
bridge=cq.Workplane('XY').rect(4.7,1.2).extrude(-1.6)
h=left.union(right).union(bridge)
try:h=h.edges('|Z').fillet(.35)
except ValueError:pass
add('H grille',h,(.95,.79,.55))
# Recessed dark acoustic apertures running beside the H.
for x in [-3,3]:
 grille=cq.Workplane('XY').center(x,0).rect(.48,5.6).extrude(-1.05)
 add('Acoustic mesh '+str(x),grille,(.12,.11,.09))
add('Receiver stem',cq.Workplane('XY').workplane(offset=16).circle(2.4).extrude(9),( .65,.56,.44))
# One continuous smooth silicone dome with a central sound bore.
dome=cq.Workplane('XY').workplane(offset=18).circle(4.8).workplane(offset=2).circle(5.0).workplane(offset=3).circle(4.35).workplane(offset=3).circle(2.8).workplane(offset=1).circle(2.0).loft(combine=True)
dome=dome.cut(cq.Workplane('XY').circle(1.35).extrude(32))
add('Silicone dome',dome,(.94,.91,.84))
assembly=cq.Assembly(name='Hush in-ear concept')
meshes=[]
for idx,(name,obj,col) in enumerate(parts):
 assembly.add(obj,name=name.replace(' ','_'),color=cq.Color(*col))
 cq.exporters.export(obj,str(OUT/f'part-{idx}.stl'),tolerance=.15,angularTolerance=.15)
 verts,faces=obj.val().tessellate(.18,.15)
 pos=[];norm=[]
 for tri in faces:
  a,b,c=[verts[k] for k in tri];u=b-a;v=c-a;n=u.cross(v).normalized()
  for q in [a,b,c]:pos.extend([round(q.x,4),round(q.y,4),round(q.z-12,4)]);norm.extend([round(n.x,4),round(n.y,4),round(n.z,4)])
 meshes.append(dict(name=name,color=col,positions=pos,normals=norm))
assembly.save(str(OUT/'hush-in-ear.step'))
combined=parts[0][1]
for _,obj,_ in parts[1:]:combined=combined.union(obj)
cq.exporters.export(combined,str(OUT/'hush-in-ear.stl'),tolerance=.15,angularTolerance=.15)
(OUT.parent/'model-data.js').write_text('window.HUSH_MODEL='+json.dumps(meshes,separators=(',',':'))+';')
(OUT/'parts.json').write_text(json.dumps([{'name':n,'color':c,'file':f'part-{i}.stl'} for i,(n,_,c) in enumerate(parts)]))
print('Exported STEP, STL and website mesh.',len(parts),'parts;',sum(len(m['positions'])//9 for m in meshes),'triangles.')
