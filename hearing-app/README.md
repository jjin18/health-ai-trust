# Hush — consumer hearing companion prototype

Run from this directory:

```sh
python -m http.server 3000 --bind 0.0.0.0
```

Open the served app in a browser. No build or dependencies are required. Google Fonts is optional; the interface has local sans-serif fallbacks.

The prototype includes an illustrated ear and hearing aid, responsive sound-wave animation tied to volume, noise reduction, and warmth, responsive layouts, simulated conversational adjustments, three presets, manual controls, undo and feedback, locally saved spaces, and session insights. Saved spaces persist in browser localStorage; chat and counters are session-only.

This is a UI prototype: no actual hearing aid is connected, no audio is processed, and no external AI service is called. Responses use local intent matching. Real integration needs a manufacturer SDK, calibrated device-specific adjustment limits, and a backend for the chosen AI provider (e.g. Doubao). API credentials must stay on that backend.

The existing research dashboard is unchanged.

The ear visualization represents changes in sound processing, not physical movement of the hearing aid. Reduced-motion preferences disable animations.

The product landing page opens with a CSS-animated dark-to-warm reveal using an original SVG hearing-aid concept. The animation settles and retains the product preview. Replay, navigation, and a product/on-ear studio toggle are included. The desktop studio preview stays sticky within its section; mobile uses normal document flow. Reduced-motion mode shows the final frame immediately.

## GitHub Pages

The static site needs no build step. Publish the contents of this directory at the root of the `gh-pages` branch, then configure repository **Settings → Pages → Deploy from a branch → gh-pages → / (root)**.

The application uses relative asset paths and works under `/health-ai-trust/`. `.nojekyll` bypasses Jekyll processing. The source branch is `hush-website`; the existing research project stays intact.

The opening video uses the corrected single-dome design and includes replay, reduced-motion fallback, and a separate interactive CAD view. No AI API credentials or hearing-device credentials are required for this simulated prototype.
