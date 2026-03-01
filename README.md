# Consumer Health AI Trust — Failure Mode Atlas
**Independent research by Jiahui Jin · February 2026**

A data-driven framework identifying the 5 core failure modes that erode consumer trust in health AI companions — built to inform product decisions at the scale of Microsoft Copilot Health.

---

## Repo Structure

```
health-ai-trust/
├── scripts/
│   ├── 1_scrape_reddit.py       # Scrape Reddit for real health AI experiences
│   ├── 2_scrape_reviews.py      # Scrape App Store reviews (Ada, K Health, Buoy)
│   ├── 3_load_datasets.py       # Load open-source medical dialogue datasets
│   └── 4_synthesize.py          # Claude-powered synthesis → trust taxonomy
├── dashboard/
│   └── index.html               # Interactive failure mode explorer
├── data/                        # Raw scraped data (gitignored)
├── outputs/
│   └── trust_patterns.json      # Final synthesized patterns (generated)
├── .env.example                 # Credentials template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quickstart

**1. Install dependencies**
```bash
pip install -r requirements.txt
pip install python-dotenv
```

**2. Set up credentials**
```bash
cp .env.example .env
# Fill in your Reddit API keys (free at reddit.com/prefs/apps)
# Fill in your Anthropic API key (free credits at console.anthropic.com)
```

**3. Collect data** (run from /scripts)
```bash
cd scripts
python 1_scrape_reddit.py       # → data/reddit_health_ai.csv
python 2_scrape_reviews.py      # → data/app_store_reviews.csv
python 3_load_datasets.py       # → data/meddialog_sample.csv
                                #   data/trust_relevant_cases.csv
```

**4. Synthesize trust patterns**
```bash
python 4_synthesize.py          # → outputs/trust_patterns.json
```

**5. Open the dashboard**
```bash
open ../dashboard/index.html
```

---

## Data Sources

| Source | What it captures | Size |
|--------|-----------------|------|
| Reddit (r/AskDocs, r/ChatGPT, r/HealthIT) | Real user frustrations with health AI | ~500 posts |
| Google Play Reviews (Ada, K Health, Buoy, WebMD) | Consumer trust breakdown at scale | ~1000 reviews |
| MedDialog — HuggingFace | Real patient-doctor conversation patterns | 260K dialogues |
| AI Medical Chatbot — HuggingFace | Patient query language + doctor responses | 250K dialogues |

---

## The 5 Failure Modes

| # | Pattern | Core Failure |
|---|---------|-------------|
| 1 | Confident When It Shouldn't Be | Overconfident responses to ambiguous symptoms |
| 2 | Useless When It Matters Most | Overcautious hedging that destroys utility |
| 3 | No Memory, No Context | Isolated sessions miss longitudinal health patterns |
| 4 | Cold at the Worst Moment | Clinical tone when emotional attunement is needed |
| 5 | Fails the Edge Case | Breakdown for complex, chronic, or atypical presentations |

---

## Why It Matters

Consumer trust in health AI dropped from **52% → 44%** in one year — despite improving models. This is a product problem, not a model problem. Each failure mode maps to a concrete product intervention and eval criterion for a consumer health AI companion.

---

## Deploy the Dashboard (free)

1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) → Import repo
3. Set root directory to `dashboard/`
4. Deploy → get a public URL in 60 seconds
