# Consumer Health AI Trust

**Jiahui Jin · February 2026**

Capability is converging. Trust is not. Health AI models are getting more accurate, but they are not getting better at the behaviors that determine whether patients feel safe enough to act on what they hear.

I built this because there is no existing benchmark for consumer health AI trust. MedQA and others measure whether the model is right. They do not measure whether the patient trusts the answer enough to act—or whether the product fails them through overconfidence, emotional blindness, paywalls at the moment of escalation, or opaque limitations. Those failures are measurable, and they are largely unmeasured.

So I validated the gap two ways: scraped user data across four platforms (app store reviews, Reddit, HuggingFace medical dialogue datasets) and a structured benchmark across four models and five trust dimensions from the Starke et al. (2025) JMIR consensus. The same patterns showed up in both. Transparency of limitation scores below 4.0 for every model. The same five failure modes in the wild. This repo is the proof of concept—the data pipeline, the synthesis, and the dashboard that shows what you find when you look.
