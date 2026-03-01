"""
STEP 4: AI-Powered Synthesis — Derive Trust Breakdown Taxonomy
==============================================================
Feeds all your scraped data into Claude via API and derives
the 5 core trust failure patterns with real quotes.

SETUP:
pip install anthropic pandas

Set your API key: export ANTHROPIC_API_KEY=your_key_here
(Get free credits at console.anthropic.com)

Run: python 4_synthesize.py
"""

import anthropic
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

def load_data():
    """Load all scraped data, handle missing files gracefully."""
    dfs = []

    files = {
        "../data/reddit_health_ai.csv":        ("title", "body"),
        "../data/app_store_reviews.csv":       ("text", None),
        "../data/trust_relevant_cases.csv":    ("patient", "doctor"),
    }

    for fname, (col1, col2) in files.items():
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            texts = df[col1].dropna().astype(str).tolist()
            if col2 and col2 in df.columns:
                texts += df[col2].dropna().astype(str).tolist()
            dfs.extend(texts[:200])  # cap per source
            print(f"  Loaded {min(200, len(texts))} entries from {fname}")
        else:
            print(f"  Skipping {fname} (not found)")

    return dfs


def synthesize_trust_patterns(texts):
    """Use Claude to derive trust breakdown taxonomy."""

    # Sample and join texts
    sample = texts[:300]
    combined = "\n---\n".join(sample[:150])  # keep under token limit

    prompt = f"""You are a product researcher analyzing real user experiences with health AI tools.

Below are raw user experiences from Reddit posts, app store reviews, and patient-doctor conversation datasets.

Your task: Identify exactly 5 distinct "trust breakdown patterns" — the core ways health AI companions fail to earn or maintain user trust.

For each pattern provide:
1. A sharp, memorable name (3-5 words)
2. A one-sentence description of what breaks trust
3. 2-3 real verbatim quotes from the data below that illustrate it
4. Why this matters specifically for a consumer health AI companion at scale (like Microsoft Copilot)
5. One concrete product intervention that could fix it

Return ONLY a JSON array with 5 objects. No preamble. No markdown. Just the JSON.

Schema:
[
  {{
    "pattern_name": "...",
    "description": "...",
    "quotes": ["...", "...", "..."],
    "why_it_matters": "...",
    "product_intervention": "..."
  }}
]

USER DATA:
{combined}
"""

    print("Sending to Claude for synthesis...")
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()

    # Clean JSON if wrapped in backticks
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    patterns = json.loads(raw)
    return patterns


def main():
    print("Loading data...")
    texts = load_data()

    if len(texts) < 10:
        print("\nNot enough data found. Run steps 1-3 first.")
        print("Using demo mode with placeholder data...")
        # Demo patterns for testing the dashboard without real data
        patterns = demo_patterns()
    else:
        print(f"\nTotal data points: {len(texts)}")
        patterns = synthesize_trust_patterns(texts)

    # Save results
    with open("../outputs/trust_patterns.json", "w") as f:
        json.dump(patterns, f, indent=2)

    print(f"\nSynthesis complete! Found {len(patterns)} trust patterns.")
    for i, p in enumerate(patterns, 1):
        print(f"  {i}. {p['pattern_name']}")

    print("\nPatterns saved to outputs/trust_patterns.json")
    print("Now open: dashboard/index.html")


def demo_patterns():
    """Demo patterns when real data isn't available yet."""
    return [
        {
            "pattern_name": "Confident When It Shouldn't Be",
            "description": "The AI gives definitive answers to ambiguous symptoms where a human doctor would express uncertainty.",
            "quotes": [
                "It told me I definitely had anxiety when I described my chest tightness. I had a pulmonary embolism.",
                "No hedging, no 'you should see a doctor' — just told me it was probably stress.",
                "It was so confident I didn't think to question it."
            ],
            "why_it_matters": "At Copilot's scale of hundreds of millions of users, overconfident responses could delay care for a meaningful percentage of users with serious conditions.",
            "product_intervention": "Build a calibrated uncertainty layer that classifies symptom severity and enforces appropriate hedging language for ambiguous or high-stakes queries."
        },
        {
            "pattern_name": "Useless When It Matters Most",
            "description": "The AI becomes overly cautious and unhelpful precisely when users are most scared and need real guidance.",
            "quotes": [
                "Every single question I asked got 'please consult a doctor.' Why am I even using this?",
                "I needed to know if I should go to the ER at 2am and it just gave me a disclaimer.",
                "So hedged it was completely useless. I'd rather Google it."
            ],
            "why_it_matters": "Overcautious AI trains users to ignore safety warnings, eroding trust in the moments that matter — the opposite of the intended outcome.",
            "product_intervention": "Define a clear escalation spectrum: inform, guide, warn, redirect. Responses should be maximally helpful within each tier rather than defaulting to the most cautious tier."
        },
        {
            "pattern_name": "No Memory, No Context",
            "description": "The AI treats every interaction as isolated, missing longitudinal patterns that a human care provider would catch.",
            "quotes": [
                "I mentioned my symptoms three separate times over two weeks. It never connected them.",
                "It didn't remember I was diabetic even though I'd told it before.",
                "A real doctor would have put this together. The AI just kept giving me generic advice."
            ],
            "why_it_matters": "Health is longitudinal. Single-session AI companions miss the patterns that define chronic conditions and early-stage serious illness.",
            "product_intervention": "Design a persistent health context layer with user-controlled memory that surfaces relevant prior disclosures when new symptoms are described."
        },
        {
            "pattern_name": "Cold at the Worst Moment",
            "description": "The AI responds to emotionally charged health disclosures with clinical, robotic language that feels dismissive.",
            "quotes": [
                "I told it I was having thoughts of self harm and it gave me a list of bullet points.",
                "My mom just got diagnosed with cancer and it started talking about treatment statistics.",
                "It felt like talking to a form, not something that understood I was scared."
            ],
            "why_it_matters": "Emotional attunement is what separates a trusted health companion from a symptom database. Getting this wrong at scale causes real harm.",
            "product_intervention": "Build an emotional context classifier that detects distress signals and routes to a more empathetic response mode before any clinical content."
        },
        {
            "pattern_name": "Fails the Edge Case",
            "description": "The AI performs well on common presentations but breaks down on atypical symptoms, rare conditions, or complex comorbidities.",
            "quotes": [
                "My symptoms didn't match the textbook and it just kept suggesting the most common diagnosis.",
                "I have three conditions that interact with each other. It had no idea what to do with that.",
                "It was trained on healthy people. My baseline is different and it couldn't adjust."
            ],
            "why_it_matters": "The patients who most need good health AI — those with complex, chronic, or rare conditions — are exactly the ones current systems fail most often.",
            "product_intervention": "Develop explicit complexity flags that trigger more cautious, specialist-referral responses when multiple conditions or atypical presentations are detected."
        }
    ]


if __name__ == "__main__":
    main()
