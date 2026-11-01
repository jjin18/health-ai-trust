'use strict';
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const story=document.querySelector('.connection-story');const stage=document.querySelector('.connection-sticky');const hero=document.querySelector('.editorial-hero');
let current=0,target=0,frame=0,last=0,visible=true,heroTarget=0,heroCurrent=0;
function measure(){const r=story.getBoundingClientRect();target=Math.max(0,Math.min(1,-r.top/Math.max(1,story.offsetHeight-innerHeight)));heroTarget=Math.max(0,Math.min(1,-hero.getBoundingClientRect().top/innerHeight));requestTick();}
function requestTick(){if(!frame&&!reduced.matches&&visible&&!document.hidden)frame=requestAnimationFrame(tick);}
function tick(now){frame=0;if(reduced.matches)return;const dt=last?Math.min(now-last,50):16;last=now;const alpha=1-Math.exp(-dt*.007);current+=(target-current)*alpha;heroCurrent+=(heroTarget-heroCurrent)*alpha;const spread=innerWidth<=650?14:90;stage.style.setProperty('--spread',(current*spread).toFixed(2)+'px');stage.style.setProperty('--bridge',(1+current*.8).toFixed(4));stage.style.setProperty('--signal',String((now*.00014)%1));stage.style.setProperty('--heading-drift',(-current*18).toFixed(2)+'px');hero.style.setProperty('--title-drift',(-heroCurrent*45).toFixed(2)+'px');hero.style.setProperty('--title-opacity',String(Math.max(0,1-heroCurrent*.8)));hero.style.setProperty('--hero-drift',(-heroCurrent*18).toFixed(2)+'px');if(visible)requestTick();}
addEventListener('scroll',measure,{passive:true});addEventListener('resize',measure);
const inView=new Set([hero]);
const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting)inView.add(entry.target);else inView.delete(entry.target);}visible=inView.size>0;if(visible){last=0;measure();}},{rootMargin:'100px'});observer.observe(story);observer.observe(hero);
document.addEventListener('visibilitychange',()=>{if(document.hidden){if(frame)cancelAnimationFrame(frame);frame=0;last=0;}else measure();});
reduced.addEventListener('change',()=>{if(reduced.matches){if(frame)cancelAnimationFrame(frame);frame=0;for(const key of ['--spread','--bridge','--heading-drift'])stage.style.removeProperty(key);for(const key of ['--title-drift','--title-opacity','--hero-drift'])hero.style.removeProperty(key);}else{last=0;measure();}});
let trigger=null;function openDialog(id,source){trigger=source;document.getElementById(id).showModal();}function closeDialog(dialog){dialog.close();if(trigger)trigger.focus();}
document.getElementById('contact-button').onclick=e=>openDialog('contact-dialog',e.currentTarget);document.getElementById('about-button').onclick=e=>openDialog('about-dialog',e.currentTarget);
for(const dialog of document.querySelectorAll('dialog')){dialog.querySelector('.close-dialog').onclick=()=>closeDialog(dialog);dialog.querySelector('.dialog-button').onclick=()=>closeDialog(dialog);dialog.addEventListener('click',e=>{if(e.target!==dialog)return;const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)closeDialog(dialog);});}
measure();
