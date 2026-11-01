import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
const host=document.getElementById('audiology-scene');
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
if(host){
try{
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));renderer.setClearColor(0x090c0b,0);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.15;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.domElement.setAttribute('aria-label','Interactive 3D concept of audiometric headphones and a remote audiology workstation. Scroll to explore.');renderer.domElement.setAttribute('role','img');renderer.domElement.tabIndex=0;host.appendChild(renderer.domElement);
const scene=new THREE.Scene();const pmrem=new THREE.PMREMGenerator(renderer);const room=new RoomEnvironment();scene.environment=pmrem.fromScene(room,.04).texture;room.dispose();pmrem.dispose();scene.environmentIntensity=.65;
const camera=new THREE.PerspectiveCamera(33,1,.1,70);
const rig=new THREE.Group();scene.add(rig);
const metal=new THREE.MeshPhysicalMaterial({color:0xd5d0c4,metalness:1,roughness:.23,clearcoat:.3,clearcoatRoughness:.25});
const edgeMetal=new THREE.MeshPhysicalMaterial({color:0xa8a99f,metalness:1,roughness:.18});
const satin=new THREE.MeshPhysicalMaterial({color:0xe5e2d9,metalness:.6,roughness:.3,clearcoat:.4});
const leather=new THREE.MeshStandardMaterial({color:0x252b26,roughness:.8});
const inner=new THREE.MeshStandardMaterial({color:0x121813,roughness:.95});
const charcoal=new THREE.MeshPhysicalMaterial({color:0x343c34,metalness:.75,roughness:.23});
const red=new THREE.MeshStandardMaterial({color:0xa9725d,metalness:.4,roughness:.3});
const blue=new THREE.MeshStandardMaterial({color:0x668b99,metalness:.4,roughness:.3});
function mesh(geo,mat,parent=rig){const m=new THREE.Mesh(geo,mat);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;}
function roundedBox(w,h,d,r,material,parent){const s=new THREE.Shape();const x=-w/2,y=-h/2;s.moveTo(x+r,y);s.lineTo(x+w-r,y);s.quadraticCurveTo(x+w,y,x+w,y+r);s.lineTo(x+w,y+h-r);s.quadraticCurveTo(x+w,y+h,x+w-r,y+h);s.lineTo(x+r,y+h);s.quadraticCurveTo(x,y+h,x,y+h-r);s.lineTo(x,y+r);s.quadraticCurveTo(x,y,x+r,y);const g=new THREE.ExtrudeGeometry(s,{depth:d,bevelEnabled:true,bevelSegments:4,steps:1,bevelSize:.018,bevelThickness:.018,curveSegments:20});g.translate(0,0,-d/2);return mesh(g,material,parent);}
const headset=new THREE.Group();headset.position.set(-1.65,.35,.15);headset.rotation.set(.08,-.22,-.16);rig.add(headset);
const band=mesh(new THREE.TorusGeometry(1.23,.12,20,100,Math.PI),metal,headset);band.position.y=.26;
const bandPad=mesh(new THREE.TorusGeometry(1.15,.11,20,100,Math.PI),leather,headset);bandPad.position.set(0,.26,.015);bandPad.scale.z=1.35;
const bandEdge=mesh(new THREE.TorusGeometry(1.28,.023,10,100,Math.PI),edgeMetal,headset);bandEdge.position.set(0,.26,.055);
for(const side of [-1,1]){
 const arm=mesh(new THREE.CylinderGeometry(.047,.047,.64,20),edgeMetal,headset);arm.position.set(side*1.23,-.04,0);
 const cup=new THREE.Group();cup.position.set(side*1.23,-.55,.05);cup.rotation.set(0,side*-.33,side*.07);headset.add(cup);
 const shell=mesh(new THREE.SphereGeometry(1,48,40),satin,cup);shell.scale.set(.47,.65,.22);shell.position.z=-.02;
 const seam=mesh(new THREE.TorusGeometry(.435,.013,10,80),edgeMetal,cup);seam.scale.y=1.36;seam.position.z=.12;
 const cushion=mesh(new THREE.TorusGeometry(.35,.12,24,80),leather,cup);cushion.scale.set(1,1.42,.85);cushion.position.z=.26;
 const cavity=mesh(new THREE.CircleGeometry(.34,64),inner,cup);cavity.scale.y=1.43;cavity.position.z=.245;
 const lining=mesh(new THREE.TorusGeometry(.24,.035,12,64),inner,cup);lining.scale.y=1.4;lining.position.z=.255;
 const mark=mesh(new THREE.CylinderGeometry(.027,.027,.008,20),side===1?red:blue,cup);mark.rotation.x=Math.PI/2;mark.position.set(.29,-.38,.14);
 const hinge=mesh(new THREE.SphereGeometry(.08,20,16),edgeMetal,headset);hinge.position.set(side*1.23,-.29,.03);
}
const cablePoints=[new THREE.Vector3(-.46,-.76,.1),new THREE.Vector3(-.37,-1.16,.06),new THREE.Vector3(.02,-1.32,.05),new THREE.Vector3(.31,-1.15,-.05)];mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(cablePoints),60,.016,8,false),leather,headset);
const workstation=new THREE.Group();workstation.position.set(1.8,.16,-.6);workstation.rotation.y=-.16;rig.add(workstation);
const frame=roundedBox(2.7,1.83,.1,.09,charcoal,workstation);frame.position.y=.1;
const trim=roundedBox(2.735,1.865,.025,.095,edgeMetal,workstation);trim.position.set(0,.1,-.055);
const screenCanvas=document.createElement('canvas');screenCanvas.width=1024;screenCanvas.height=680;const ctx=screenCanvas.getContext('2d');
ctx.fillStyle='#18221e';ctx.fillRect(0,0,1024,680);const grd=ctx.createRadialGradient(510,260,30,510,300,650);grd.addColorStop(0,'#29382d');grd.addColorStop(1,'#111a16');ctx.fillStyle=grd;ctx.fillRect(0,0,1024,680);
ctx.strokeStyle='#5b6b5530';ctx.lineWidth=2;for(let x=120;x<=920;x+=160){ctx.beginPath();ctx.moveTo(x,130);ctx.lineTo(x,510);ctx.stroke();}for(let y=130;y<=510;y+=95){ctx.beginPath();ctx.moveTo(120,y);ctx.lineTo(920,y);ctx.stroke();}
function curve(points,color,dashed){ctx.strokeStyle=color;ctx.lineWidth=5;ctx.setLineDash(dashed?[12,10]:[]);ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();ctx.setLineDash([]);for(const [x,y] of points){ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}}
curve([[120,215],[280,235],[440,275],[600,355],[760,340],[920,410]],'#d3bda0',false);curve([[120,250],[280,265],[440,305],[600,395],[760,380],[920,440]],'#87a9a9',true);
const screenTexture=new THREE.CanvasTexture(screenCanvas);screenTexture.colorSpace=THREE.SRGBColorSpace;
const screenMat=new THREE.MeshPhysicalMaterial({map:screenTexture,emissiveMap:screenTexture,emissive:0xffffff,emissiveIntensity:.55,roughness:.17,metalness:.05,clearcoat:.35,clearcoatRoughness:.16});const screen=mesh(new THREE.PlaneGeometry(2.55,1.67),screenMat,workstation);screen.position.set(0,.1,.085);
const cameraDot=mesh(new THREE.SphereGeometry(.013,12,12),inner,workstation);cameraDot.position.set(0,.985,.06);
const neck=mesh(new THREE.CylinderGeometry(.055,.075,.6,20),metal,workstation);neck.position.set(0,-1.08,-.025);
const base=roundedBox(.95,.53,.048,.08,metal,workstation);base.rotation.x=-Math.PI/2;base.position.set(0,-1.39,.03);
const ground=mesh(new THREE.PlaneGeometry(30,30),new THREE.ShadowMaterial({opacity:.28}),scene);ground.rotation.x=-Math.PI/2;ground.position.y=-1.25;ground.receiveShadow=true;
const key=new THREE.DirectionalLight(0xfff4dc,4.5);key.position.set(-3,6,5);key.castShadow=true;key.shadow.mapSize.set(1024,1024);key.shadow.camera.left=-6;key.shadow.camera.right=6;key.shadow.camera.top=5;key.shadow.camera.bottom=-5;key.shadow.normalBias=.025;key.shadow.bias=-.0001;scene.add(key);
const rim=new THREE.DirectionalLight(0x9ab8bd,3);rim.position.set(4,2,-4);scene.add(rim);const fill=new THREE.DirectionalLight(0xf2dbc0,1.8);fill.position.set(-4,-1,3);scene.add(fill);
const link=new THREE.Group();rig.add(link);const linkMat=new THREE.MeshBasicMaterial({color:0xc1bc9d,transparent:true,opacity:.55});
const linkPath=new THREE.CatmullRomCurve3([new THREE.Vector3(-.32,.1,0),new THREE.Vector3(.08,.1,-.15),new THREE.Vector3(.45,.1,-.35)]);mesh(new THREE.TubeGeometry(linkPath,24,.006,5,false),linkMat,link);
const signal=mesh(new THREE.SphereGeometry(.025,12,12),new THREE.MeshBasicMaterial({color:0xf5ddb5}),link);
let visible=true,ticking=false,scrollProgress=0,drag=false,rotation=0,pointerX=0,pointerY=0,frameId=0;const look=new THREE.Vector3();const mobile=()=>innerWidth<=650;const start=performance.now();
function size(){const r=host.getBoundingClientRect();if(r.width<1||r.height<1)return;renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();requestRender();}
function updateScroll(){const hero=document.getElementById('home');const rect=hero.getBoundingClientRect();const travel=Math.max(1,hero.offsetHeight-innerHeight);scrollProgress=Math.max(0,Math.min(1,-rect.top/travel));hero.style.setProperty('--progress',scrollProgress.toFixed(3));requestRender();}
function requestRender(){if(!frameId&&visible)frameId=requestAnimationFrame(draw);}
function draw(now){frameId=0;const seconds=Math.max(0,(now-start)/1000);const intro=reduced.matches?1:Math.min(1,seconds/2.6);const ease=1-Math.pow(1-intro,3);const p=reduced.matches?1:scrollProgress;const targetP=Math.min(1,p*1.25);const isMobile=mobile();
if(isMobile){camera.position.set(.8+pointerX*.08,.9+pointerY*.04,14.0);look.set(.1,.45,0);rig.scale.setScalar(.97);rig.position.set(0,-.28,0);}else{camera.position.set(THREE.MathUtils.lerp(-1.1,2.2,targetP)+pointerX*.14,THREE.MathUtils.lerp(.65,1.45,targetP)+pointerY*.08,THREE.MathUtils.lerp(8.8,10.2,targetP)+(1-ease)*2);look.set(.05,.48,0);rig.scale.setScalar(1.12);rig.position.set(0,-.45,0);}
camera.lookAt(look);rig.rotation.y=rotation+(reduced.matches?0:Math.sin(seconds*.3)*.018);headset.rotation.y=-.22+(reduced.matches?0:Math.sin(seconds*.4)*.035);scene.environmentRotation.y=reduced.matches?0:Math.sin(seconds*.12)*.16;signal.position.copy(linkPath.getPoint(reduced.matches?.5:(seconds*.24)%1));renderer.render(scene,camera);host.classList.add('ready');
if(!reduced.matches&&visible)requestRender();}
new ResizeObserver(size).observe(host);new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)requestRender();},{rootMargin:'150px'}).observe(host);
window.addEventListener('scroll',updateScroll,{passive:true});window.addEventListener('resize',()=>{size();updateScroll();});host.addEventListener('pointermove',e=>{const r=host.getBoundingClientRect();pointerX=(e.clientX-r.left)/r.width-.5;pointerY=(e.clientY-r.top)/r.height-.5;if(drag){rotation+=e.movementX*.004;}requestRender();});host.addEventListener('pointerleave',()=>{pointerX=pointerY=0;});renderer.domElement.addEventListener('pointerdown',e=>{if(e.pointerType!=='mouse')return;drag=true;renderer.domElement.setPointerCapture(e.pointerId);});renderer.domElement.addEventListener('pointerup',()=>drag=false);renderer.domElement.addEventListener('pointercancel',()=>drag=false);renderer.domElement.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight'].includes(e.key))return;e.preventDefault();rotation+=e.key==='ArrowLeft'?-.1:.1;requestRender();});reduced.addEventListener('change',()=>{updateScroll();requestRender();});
renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();host.classList.remove('ready');if(frameId)cancelAnimationFrame(frameId);frameId=0;});renderer.domElement.addEventListener('webglcontextrestored',()=>{host.classList.add('ready');requestRender();});
size();updateScroll();
window.HUSH_SCENE={renderer,scene,camera,requestRender};
}catch(e){console.warn('3D unavailable. Showing equipment illustration.',e.message);}
}
