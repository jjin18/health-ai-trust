'use strict';
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
const story=document.querySelector('.distance-story');
const stage=document.querySelector('.distance-stage');
let frame=0;
function updateStory(){frame=0;if(reducedMotion.matches)return;const rect=story.getBoundingClientRect();const travel=Math.max(1,story.offsetHeight-innerHeight);const progress=Math.max(0,Math.min(1,-rect.top/travel));const mobile=innerWidth<=600;stage.style.setProperty('--gap',(progress*(mobile?50:120))+'px');stage.style.setProperty('--panel-scale',String(1-progress*.04));}
addEventListener('scroll',()=>{if(!frame)frame=requestAnimationFrame(updateStory);},{passive:true});
addEventListener('resize',()=>{if(!frame)frame=requestAnimationFrame(updateStory);});
reducedMotion.addEventListener('change',()=>{if(reducedMotion.matches){stage.style.removeProperty('--gap');stage.style.removeProperty('--panel-scale');}else updateStory();});
const observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('is-visible');observer.unobserve(e.target);}}),{threshold:.15});
for(const el of document.querySelectorAll('.local-story h2,.partnership h2,.connection-symbol')){el.classList.add('reveal');observer.observe(el);}
let trigger=null;
function openDialog(id,source){trigger=source;document.getElementById(id).showModal();}
function closeDialog(dialog){dialog.close();if(trigger)trigger.focus();}
document.getElementById('contact-button').onclick=e=>openDialog('contact-dialog',e.currentTarget);
document.getElementById('about-button').onclick=e=>openDialog('about-dialog',e.currentTarget);
for(const dialog of document.querySelectorAll('dialog')){dialog.querySelector('.close-dialog').onclick=()=>closeDialog(dialog);dialog.querySelector('.dialog-button').onclick=()=>closeDialog(dialog);dialog.addEventListener('click',e=>{if(e.target!==dialog)return;const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)closeDialog(dialog);});}
updateStory();
