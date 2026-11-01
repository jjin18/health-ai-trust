# Remote audiology direction

Goal: communicate audiology and the ENT practice benefit through a minimal product narrative. No fabricated patient photos, performance statistics or live service claims.

Reference categories for further review:
- TytoCare — remote examination and local examination tools: https://www.tytocare.com/
- Butterfly — portable imaging and clinician-guided workflows: https://www.butterflynetwork.com/
- SHOEBOX — audiology and testing workflows: https://www.shoebox.md/
- hearX — hearing-care technology: https://www.hearxgroup.com/
- Auris MaestroAuD — source material supplied by the user: https://aurispracticesolutions.com/maestroaud

The external reference sites could not be fetched in this environment (proxy 403). No current visual inspection or feature-by-feature comparison is claimed. The Auris workflow comes from the user's pasted website text.

Design decisions:
1. Audiometric headphone/workstation illustration makes the clinical category explicit. It is not a real device specification or patient result.
2. Prepare / Connect / Review interaction explains local support, remote clinician, and return of reviewed findings.
3. ENT-focused section explains coverage, patient continuity and result handoff. No guaranteed increases in capacity, revenue or named EMR integrations.
4. Equipment/environment suitability and clinical availability are explained in the concept dialog.
5. Contact remains a truthful coming-soon dialog pending a real contact address or booking link.

## Real-time 3D update

The opening uses actual procedural Three.js meshes, not the SVG or a pre-rendered video: metallic headphone frame, cushioned earcups, yokes and cable, a glass-fronted workstation, procedural chart texture, room environment reflections, direct lights and contact shadows. Scroll changes camera position. The chart is illustrative. No real clinician connection or live patient data is implied. SVG remains only as a WebGL fallback.
