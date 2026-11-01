# Hush — remote audiology

A minimal, warm editorial page: audiology within reach, a patient–audiologist connection that stays linked across rooms, and a partnership invitation. The page loads no 3D renderer, staged photos, external fonts, or libraries.

Run `python -m http.server 3000` in this directory. No install or build step is required.

Scroll positions are smoothed with time-based requestAnimationFrame interpolation. The signal moves continuously; motion pauses off-screen and respects reduced-motion preferences. Contact and concept dialogs support keyboard use. The partnership destination is still pending, and no enquiries are submitted.

Earlier 3D and hearing-device assets remain as source history, but are not loaded by this page. Sources are saved on hush-website; the directory contents deploy from gh-pages.
