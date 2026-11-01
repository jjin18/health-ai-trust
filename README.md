# Hush — remote audiology for ENT

A static concept website with a real-time Three.js 3D hero: modeled headphones, reflective materials, remote workstation, procedural illustrative chart, environment lighting, shadows, scroll-driven camera, and mouse/keyboard rotation. No staged clinical photography. Equipment and graphs are illustrations, not patient data or validated equipment specifications.

## Run

`python -m http.server 3000` in this directory. The renderer is bundled and self-hosted; no CDN, API key, build or installation is needed to view the site.

## Rebuild 3D

`npm ci` followed by `npm run build`. Versions are pinned in package-lock.json. Editable geometry and animation are in assets/scene-source.js. The Three.js MIT license is retained in assets/vendor/THREE-LICENSE.txt.

The renderer pauses when off-screen, caps pixel ratio, responds to resize and reduced-motion preferences, and falls back to the equipment SVG if WebGL cannot initialize. Prepare / Connect / Review tabs and accessible dialogs explain the proposed ENT workflow.

The partnership destination remains pending; enquiries are not submitted. The contact dialog states enquiries are opening soon. Sources are pushed to hush-website; the directory contents are deployed on gh-pages.
