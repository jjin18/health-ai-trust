/* Dependency-free WebGL viewer for the CadQuery concept mesh. */
(()=>{
const host=document.querySelector('.product-render');if(!host||!window.HUSH_MODEL)return;
const canvas=document.createElement('canvas');canvas.className='cad-canvas';canvas.setAttribute('aria-label','Interactive 3D Hush in-ear hearing aid. Drag to rotate.');canvas.setAttribute('role','img');canvas.tabIndex=0;
const gl=canvas.getContext('webgl',{alpha:true,antialias:true});if(!gl)return;
const vs=`attribute vec3 p;attribute vec3 n;uniform mat3 rot;uniform float aspect;varying vec3 normal;varying vec3 position;void main(){vec3 v=rot*p;normal=rot*n;position=v;float perspective=75.0/(75.0-v.z);gl_Position=vec4(v.x/23.0/aspect*perspective,v.y/23.0*perspective,-v.z/80.0,1.0);}`;
const fs=`precision mediump float;varying vec3 normal;varying vec3 position;uniform vec3 color;uniform float time;void main(){vec3 n=normalize(normal);vec3 light=normalize(vec3(sin(time*.25)*1.5,1.2,2.0));vec3 view=vec3(0.,0.,1.);float diffuse=max(dot(n,light),0.);float spec=pow(max(dot(n,normalize(light+view)),0.),48.);float rim=pow(1.-abs(dot(n,view)),3.);float strip=pow(max(0.,1.-abs(dot(n,normalize(vec3(-1.,.4,1.))))),14.);vec3 result=color*(.48+.48*diffuse)+vec3(1.,.92,.78)*spec*.85+vec3(.3,.24,.18)*rim+strip*.1;gl_FragColor=vec4(result,1.);}`;
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
try{
const prog=gl.createProgram();gl.attachShader(prog,shader(gl.VERTEX_SHADER,vs));gl.attachShader(prog,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(prog);if(!gl.getProgramParameter(prog,gl.LINK_STATUS))return;gl.useProgram(prog);
const meshes=window.HUSH_MODEL.map(m=>{
if(/shell|Silicone|Receiver|outlet/.test(m.name)){
 const sums=new Map();const keys=[];
 for(let i=0;i<m.positions.length;i+=3){const key=m.positions.slice(i,i+3).join(',');keys.push(key);const n=sums.get(key)||[0,0,0];for(let k=0;k<3;k++)n[k]+=m.normals[i+k];sums.set(key,n);}
 m.normals=m.positions.map((_,i)=>{const n=sums.get(keys[Math.floor(i/3)]);return n[i%3]/(Math.hypot(...n)||1);});
}
function buffer(data){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.STATIC_DRAW);return b;}return{p:buffer(m.positions),n:buffer(m.normals),color:m.color,count:m.positions.length/3};});
const loc={p:gl.getAttribLocation(prog,'p'),n:gl.getAttribLocation(prog,'n'),rot:gl.getUniformLocation(prog,'rot'),aspect:gl.getUniformLocation(prog,'aspect'),color:gl.getUniformLocation(prog,'color'),time:gl.getUniformLocation(prog,'time')};
host.append(canvas);host.classList.add('has-cad');gl.enable(gl.DEPTH_TEST);
let yaw=-.55,pitch=2.5,dragging=false,lastX=0,lastY=0,frame=0;const reduced=matchMedia('(prefers-reduced-motion: reduce)');
function render(t=0){frame=0;const size=host.getBoundingClientRect();const dpr=Math.min(devicePixelRatio,2);const w=Math.max(1,Math.round(size.width*dpr)),h=Math.max(1,Math.round(size.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}gl.viewport(0,0,w,h);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);const a=yaw+(reduced.matches?0:Math.sin(t*.00022)*.1),b=pitch;const c=Math.cos(a),s=Math.sin(a),d=Math.cos(b),e=Math.sin(b);gl.uniformMatrix3fv(loc.rot,false,new Float32Array([c,s*e,-s*d,0,d,e,s,-c*e,c*d]));gl.uniform1f(loc.aspect,w/h);gl.uniform1f(loc.time,reduced.matches?0:t*.001);for(const m of meshes){for(const k of ['p','n']){gl.bindBuffer(gl.ARRAY_BUFFER,m[k]);gl.enableVertexAttribArray(loc[k]);gl.vertexAttribPointer(loc[k],3,gl.FLOAT,false,0,0);}gl.uniform3fv(loc.color,m.color);gl.drawArrays(gl.TRIANGLES,0,m.count);}if(!reduced.matches&&visible)frame=requestAnimationFrame(render);}
let visible=true;const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible&&!frame)frame=requestAnimationFrame(render);});observer.observe(canvas);
canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);});canvas.addEventListener('pointermove',e=>{if(!dragging)return;yaw+=(e.clientX-lastX)*.01;pitch+=(e.clientY-lastY)*.01;lastX=e.clientX;lastY=e.clientY;if(!frame)frame=requestAnimationFrame(render);});for(const ev of ['pointerup','pointercancel'])canvas.addEventListener(ev,()=>dragging=false);
canvas.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();yaw+=e.key==='ArrowLeft'?-.15:e.key==='ArrowRight'?.15:0;pitch+=e.key==='ArrowUp'?-.15:e.key==='ArrowDown'?.15:0;if(!frame)frame=requestAnimationFrame(render);});
new ResizeObserver(()=>{if(!frame)frame=requestAnimationFrame(render);}).observe(host);reduced.addEventListener('change',()=>{if(!frame)frame=requestAnimationFrame(render);});render();
}catch(e){canvas.remove();host.classList.remove('has-cad');console.error('CAD preview unavailable',e);}
})();
