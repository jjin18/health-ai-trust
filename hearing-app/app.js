'use strict';
const $ = id => document.getElementById(id);
const presets = {
 everyday: {name:'Everyday',volume:60,noise:40,warmth:50,environment:'Calm & comfortable'},
 cafe: {name:'Café conversations',volume:55,noise:75,warmth:60,environment:'Focused on conversation'},
 outdoors: {name:'Out & about',volume:55,noise:55,warmth:45,environment:'Aware of your surroundings'}
};
let settings={...presets.everyday}, adjustments=0, helpful=0, busy=false, toastTimer;
let saved=Object.values(presets).map(p=>({...p}));
try {const data=JSON.parse(localStorage.getItem('hush-spaces'));if(Array.isArray(data)&&data.every(p=>typeof p.name==='string'&&['volume','noise','warmth'].every(k=>Number.isFinite(p[k])&&p[k]>=0&&p[k]<=100)))saved=data;}catch{}
function toast(text){$('toast').textContent=text;$('toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('visible'),3500);}
let fitTimer, lastSound='';
function renderEar(){
 const stage=$('ear-stage');
 stage.style.setProperty('--wave-size',String(.65+settings.volume/150));
 stage.style.setProperty('--noise-opacity',String(1-settings.noise/110));
 stage.style.setProperty('--sound-color',`hsl(${145-settings.warmth*.65} 24% 58%)`);
 const signature=[settings.volume,settings.noise,settings.warmth].join('-');
 if(lastSound && signature!==lastSound){
  $('ear-change').textContent=`Volume ${settings.volume} · Focus ${settings.noise} · Warmth ${settings.warmth}`;
  stage.classList.remove('adjusting');void stage.offsetWidth;stage.classList.add('adjusting');
  clearTimeout(fitTimer);fitTimer=setTimeout(()=>stage.classList.remove('adjusting'),2200);
 }
 lastSound=signature;
 stage.querySelector('svg').setAttribute('aria-label',`Illustrated sound settings: volume ${settings.volume} percent, noise reduction ${settings.noise} percent, warmth ${settings.warmth} percent. Simulated visualization.`);
}
function render(){renderEar();renderPhoneSettings();for(const key of ['volume','noise','warmth']){$(key).value=settings[key];$(key).style.setProperty('--value',settings[key]+'%');$(key+'-value').textContent=settings[key]+'%';}$('mode-label').textContent=settings.name==='Everyday'?'Everyday balance':settings.name;$('environment-label').textContent=settings.environment||'Personalized for you';$('orb-label').textContent=settings.name==='Everyday'?'Balanced for you':'Tuned to your moment';document.querySelectorAll('[data-preset]').forEach(b=>{const active=settings.name===presets[b.dataset.preset].name;b.classList.toggle('selected',active);b.querySelector('.space-check').textContent=active?'✓':'↗';b.setAttribute('aria-pressed',String(active));});$('adjustment-count').textContent=adjustments;$('helpful-count').textContent=helpful;$('saved-count').textContent=saved.length;$('insight-saved-count').textContent=saved.length;}
function apply(next){settings={...next};adjustments++;render();}
function message(text,role='assistant'){const node=document.createElement('div');node.className='message '+role;node.textContent=text;$('chat').append(node);$('chat').scrollTop=$('chat').scrollHeight;return node;}
function feedback(node,previous){const row=document.createElement('div');row.className='feedback';const good=document.createElement('button');good.textContent='✓ That’s better';const undo=document.createElement('button');undo.textContent='↶ Undo change';good.onclick=()=>{helpful++;render();row.replaceChildren(document.createTextNode('Preference noted for this session. ✓'));toast('Thanks. Your feedback helps shape your sound.');};undo.onclick=()=>{apply(previous);row.replaceChildren(document.createTextNode('Previous settings restored.'));toast('Back to your previous settings.');};row.append(good,undo);node.append(row);$('chat').scrollTop=$('chat').scrollHeight;}
function respond(text){if(busy)return;message(text,'user');$('suggestions').hidden=true;busy=true;const pending=message('Finding a little more comfortable…');setTimeout(()=>{const previous={...settings};const next={...settings,name:'Personal sound',environment:'Personalized for you'};const t=text.toLowerCase();let response,changed=true;const clamp=v=>Math.max(0,Math.min(100,v));
if(/sharp|tinny|bright|harsh/.test(t)){next.warmth=clamp(next.warmth+15);response='Let’s soften the edges. I’ve increased sound warmth by 15 points in your demo settings. Does your voice feel more natural?';}
else if(/background|noise|restaurant|cafe|café|clatter/.test(t)){next.noise=clamp(next.noise+20);response='A little less distraction. I’ve increased noise reduction by 20 points in your demo settings. Is the conversation still clear?';}
else if(/too loud|louder than|softer|quiet down|lower|volume down/.test(t)){next.volume=clamp(next.volume-10);response='Let’s bring things down a little. I’ve lowered demo volume by 10 points. Does that feel more comfortable?';}
else if(/quiet|louder|can.t hear|increase.*volume|volume up/.test(t)){next.volume=clamp(next.volume+10);response='I’ve increased demo volume by 10 points. Can you hear more comfortably now?';}
else if(/muffled|dull|clear|understand/.test(t)){next.warmth=clamp(next.warmth-10);next.noise=clamp(next.noise+10);response='Let’s try a little more definition. I’ve reduced warmth and increased noise reduction by 10 points in the demo. Are words easier to follow?';}
else if(/reset|balance|undo/.test(t)){Object.assign(next,presets.everyday);response='Back to everyday balance. You can always start fresh.';}
else{changed=false;response='I can help explore three things in this demo: volume, background noise, and sound warmth. Does it feel too loud, too sharp, or too noisy?';}
pending.textContent=response;if(changed){apply(next);feedback(pending,previous);}busy=false;$('chat').scrollTop=$('chat').scrollHeight;},650);}
$('chat-form').addEventListener('submit',e=>{e.preventDefault();const text=$('chat-input').value.trim();if(!text||busy)return;$('chat-input').value='';respond(text);});
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>respond(b.dataset.prompt));
document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>{apply(presets[b.dataset.preset]);toast(settings.name+' applied to demo settings.');});
for(const key of ['volume','noise','warmth']){$(key).addEventListener('input',e=>{settings[key]=Number(e.target.value);settings.name='Personal sound';settings.environment='Personalized for you';render();});$(key).addEventListener('change',()=>{adjustments++;render();});}
$('reset').onclick=()=>{apply(presets.everyday);toast('Back to everyday balance.');};
$('save-current').onclick=()=>{const name=settings.name==='Personal sound'?'My space '+(saved.filter(p=>p.name.startsWith('My space')).length+1):settings.name;const entry={...settings,name};const existing=saved.findIndex(p=>p.name===name);if(existing>=0)saved[existing]=entry;else saved.push(entry);try{localStorage.setItem('hush-spaces',JSON.stringify(saved));}catch{toast('Saved for this session. Browser storage is unavailable.');render();return;}render();toast('“'+name+'” saved to your spaces.');};
function renderSaved(){const list=$('saved-list');list.replaceChildren();saved.forEach(p=>{const row=document.createElement('article');row.className='saved-row';const content=document.createElement('div');const title=document.createElement('strong');title.textContent=p.name;const detail=document.createElement('p');detail.textContent=`Volume ${p.volume}% · Noise reduction ${p.noise}% · Warmth ${p.warmth}%`;content.append(title,detail);const button=document.createElement('button');button.textContent='Apply space';button.onclick=()=>{apply(p);showView('home');toast(p.name+' applied to demo settings.');};row.append(content,button);list.append(row);});}
function showView(view){['home','saved','insights'].forEach(v=>$(v+'-view').hidden=v!==view);document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));$('page-label').textContent={home:'My sound',saved:'Saved spaces',insights:'Listening insights'}[view];if(view==='saved')renderSaved();render();}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>showView(b.dataset.view));
$('device-button').onclick=()=>$('device-dialog').showModal();$('info-button').onclick=()=>$('device-dialog').showModal();document.querySelectorAll('.dialog-close').forEach(b=>b.onclick=()=>$('device-dialog').close());$('device-dialog').addEventListener('click',e=>{if(e.target===$('device-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});$('profile').onclick=()=>toast('Jamie’s demo profile · Preferences stay in this browser.');document.querySelector('.brand').onclick=e=>{e.preventDefault();showView('home');};render();

// Replay the cinematic reveal without restarting the sound studio.
function replayReveal(){const hero=$('product-hero');const items=[hero,hero.querySelector('.product-render'),hero.querySelector('.hero-title'),hero.querySelector('.hero-bottom')];items.forEach(el=>el.style.animation='none');hero.classList.add('replaying');hero.style.setProperty('--replay','none');const style=document.createElement('style');style.textContent='.product-hero.replaying:after{animation:none}';document.head.append(style);void hero.offsetWidth;style.remove();hero.classList.remove('replaying');items.forEach(el=>el.style.animation='');hero.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});}
$('replay-button').onclick=replayReveal;
$('replay-intro').onclick=()=>{closeMenu();replayReveal();};
function closeMenu(){$('product-menu').hidden=true;$('menu-toggle').setAttribute('aria-expanded','false');}
$('menu-toggle').onclick=()=>{const open=$('product-menu').hidden;$('product-menu').hidden=!open;$('menu-toggle').setAttribute('aria-expanded',String(open));};
$('product-menu').querySelectorAll('a').forEach(a=>a.addEventListener('click',closeMenu));
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeMenu();});
$('ear-view-toggle').onclick=()=>{const active=$('ear-stage').classList.toggle('on-ear');$('ear-view-toggle').textContent=active?'View device ↗':'View on ear ↗';$('ear-view-toggle').setAttribute('aria-pressed',String(active));};

let phoneBusy=false;
function phoneMessage(text,role){const node=document.createElement('div');node.className='phone-bubble phone-'+role;node.textContent=text;$('phone-conversation').append(node);$('phone-conversation').scrollTop=$('phone-conversation').scrollHeight;return node;}
function phoneRespond(text){if(phoneBusy||busy)return;phoneBusy=true;phoneMessage(text,'user');respond(text);const bubble=phoneMessage('One little adjustment…','assistant');setTimeout(()=>{const t=text.toLowerCase();bubble.textContent=/sharp|tinny|bright|harsh/.test(t)?'A little warmer. Better?':/background|noise|noisy|cafe|café/.test(t)?'Less noise. More conversation.':/loud|softer|lower/.test(t)?'A little softer. How’s that?':/quiet|hear/.test(t)?'A little louder. Better?':'Too loud, too sharp, or too noisy?';phoneBusy=false;$('phone-conversation').scrollTop=$('phone-conversation').scrollHeight;},750);}
$('phone-form').addEventListener('submit',e=>{e.preventDefault();const t=$('phone-input').value.trim();if(!t||phoneBusy||busy)return;$('phone-input').value='';phoneRespond(t);});
document.querySelectorAll('[data-phone-prompt]').forEach(b=>b.onclick=()=>phoneRespond(b.dataset.phonePrompt));
$('phone-reset').onclick=()=>{if(phoneBusy)return;$('phone-conversation').replaceChildren();phoneMessage('How does your world sound?','assistant');apply(presets.everyday);};

// Cosmetic movement only: the concept hardware does not physically change shape.
const productRender=document.querySelector('.product-render');
const motionPreference=matchMedia('(prefers-reduced-motion: reduce)');
function resetProductMotion(){for(const [key,value] of Object.entries({'--tilt-x':'0deg','--tilt-y':'0deg','--scroll-turn':'0deg','--scroll-y':'0px','--scroll-scale':'1'}))productRender.style.setProperty(key,value);}
$('product-hero').addEventListener('pointermove',e=>{if(motionPreference.matches||e.pointerType==='touch')return;const box=$('product-hero').getBoundingClientRect();const x=(e.clientX-box.left)/box.width;const y=(e.clientY-box.top)/box.height;productRender.style.setProperty('--tilt-x',((.5-y)*9)+'deg');productRender.style.setProperty('--tilt-y',((x-.5)*18)+'deg');productRender.style.setProperty('--light-x',(x*100)+'%');productRender.style.setProperty('--light-y',(y*100)+'%');});
$('product-hero').addEventListener('pointerleave',()=>{productRender.style.setProperty('--tilt-x','0deg');productRender.style.setProperty('--tilt-y','0deg');});
let productFrame=0;
function updateProductScroll(){productFrame=0;if(motionPreference.matches)return;const rect=$('product-hero').getBoundingClientRect();const progress=Math.max(0,Math.min(1,-rect.top/rect.height));productRender.style.setProperty('--scroll-turn',(progress*15)+'deg');productRender.style.setProperty('--scroll-y',(-progress*50)+'px');productRender.style.setProperty('--scroll-scale',String(1-progress*.08));}
window.addEventListener('scroll',()=>{if(!productFrame)productFrame=requestAnimationFrame(updateProductScroll);},{passive:true});
motionPreference.addEventListener('change',()=>{if(motionPreference.matches)resetProductMotion();else updateProductScroll();});

const productVideo=$('product-video');
function playProductVideo(){if(motionPreference.matches){productVideo.pause();return;}productVideo.currentTime=0;productVideo.play().catch(()=>{});}
$('replay-button').addEventListener('click',()=>{document.querySelector('.product-render').classList.remove('using-cad');$('three-toggle').setAttribute('aria-pressed','false');playProductVideo();});
$('replay-intro').addEventListener('click',playProductVideo);
$('three-toggle').onclick=()=>{const active=document.querySelector('.product-render').classList.toggle('using-cad');$('three-toggle').setAttribute('aria-pressed',String(active));$('three-toggle').setAttribute('aria-label',active?'Switch to product video':'Switch to interactive 3D model');if(active)productVideo.pause();else playProductVideo();};
motionPreference.addEventListener('change',()=>{if(motionPreference.matches)productVideo.pause();else playProductVideo();});
if(motionPreference.matches)productVideo.pause();

function renderPhoneSettings(){for(const key of ['volume','noise','warmth']){const el=$('phone-'+key);if(el){el.value=settings[key];el.style.setProperty('--value',settings[key]+'%');$('phone-'+key+'-value').textContent=settings[key]+'%';}}document.querySelectorAll('[data-phone-space]').forEach(b=>b.setAttribute('aria-pressed',String(presets[b.dataset.phoneSpace].name===settings.name)));}
function switchPhoneTab(tab){document.querySelectorAll('[data-phone-tab]').forEach(b=>{const active=b.dataset.phoneTab===tab;b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;$('phone-'+b.dataset.phoneTab+'-panel').hidden=!active;});}
document.querySelectorAll('[data-phone-tab]').forEach((b,index)=>{b.onclick=()=>switchPhoneTab(b.dataset.phoneTab);b.onkeydown=e=>{const tabs=[...document.querySelectorAll('[data-phone-tab]')];let next;if(e.key==='ArrowRight')next=(index+1)%tabs.length;else if(e.key==='ArrowLeft')next=(index+tabs.length-1)%tabs.length;else if(e.key==='Home')next=0;else if(e.key==='End')next=tabs.length-1;else return;e.preventDefault();switchPhoneTab(tabs[next].dataset.phoneTab);tabs[next].focus();};});
for(const key of ['volume','noise','warmth']){$('phone-'+key).addEventListener('input',e=>{settings[key]=Number(e.target.value);settings.name='Personal sound';settings.environment='Personalized for you';render();});$('phone-'+key).addEventListener('change',()=>{adjustments++;render();});}
document.querySelectorAll('[data-phone-space]').forEach(b=>b.onclick=()=>{apply(presets[b.dataset.phoneSpace]);toast(settings.name+' applied to demo settings.');});
renderPhoneSettings();
