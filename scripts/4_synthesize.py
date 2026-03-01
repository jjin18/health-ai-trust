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

def load_data_with_sources():
    """Load scraped data with source URL/index for each item. Returns (items, source_map)."""
    items = []
    source_map = {}  # id -> {url, label}

    # Reddit: has url per post
    reddit_path = "../data/reddit_health_ai.csv"
    if os.path.exists(reddit_path):
        df = pd.read_csv(reddit_path)
        url_col = "url" if "url" in df.columns else None
        start = len(items)
        for i, row in df.head(150).iterrows():
            text = (str(row.get("title", "")) + " " + str(row.get("body", ""))).strip()[:800]
            if not text:
                continue
            sid = f"R{len(items)+1}"
            url = str(row[url_col]) if url_col and pd.notna(row.get(url_col)) else ""
            items.append({"id": sid, "text": text, "url": url, "label": "Reddit"})
            source_map[sid] = {"url": url, "label": "Reddit"}
        print(f"  Loaded {len(items) - start} from Reddit")

    # App store reviews: link to app's Play Store page
    APP_NAME_TO_ID = {
        "WebMD": "com.webmd.android", "Ada Health": "com.adahealth.app", "K Health": "com.khealth.android",
        "Buoy Health": "com.buoyhealth.android", "Babylon Health": "com.babylon.patientapp", "Youper AI": "ai.youper.android",
        "Symptomate": "com.symptomate.mobile", "HealthTap": "com.healthtap.android", "Sharecare": "com.sharecare.android",
        "Infermedica": "com.infermedica.app", "Livongo Health": "com.livongo.health", "Omada Health": "com.omadahealth.omada",
        "Noom": "com.noom.coach", "Teladoc": "com.teladoc.members", "MDLive": "com.mdlive.mobile", "Doctor On Demand": "com.doctorondemand.android",
    }
    app_path = "../data/app_store_reviews.csv"
    if os.path.exists(app_path):
        df = pd.read_csv(app_path)
        has_url = "app_store_url" in df.columns
        has_app_id = "app_id" in df.columns
        start = len(items)
        for i, row in df.head(200).iterrows():
            text = str(row.get("text", ""))[:600]
            if not text or text == "nan":
                continue
            sid = f"A{len(items)+1}"
            app_name = str(row.get("app", "App"))
            if has_url and pd.notna(row.get("app_store_url")):
                url = str(row["app_store_url"])
            elif has_app_id and pd.notna(row.get("app_id")):
                url = f"https://play.google.com/store/apps/details?id={row['app_id']}"
            else:
                app_id = APP_NAME_TO_ID.get(app_name, "")
                url = f"https://play.google.com/store/apps/details?id={app_id}" if app_id else "https://play.google.com/store/apps"
            label = f"{app_name} (reviews)"
            items.append({"id": sid, "text": text, "url": url, "label": label})
            source_map[sid] = {"url": url, "label": label}
        print(f"  Loaded {len(items) - start} from app reviews")

    # HuggingFace trust-relevant cases: link to dataset + case index
    hf_path = "../data/trust_relevant_cases.csv"
    HF_DATASET_URL = "https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot"
    if os.path.exists(hf_path):
        df = pd.read_csv(hf_path)
        has_index = "case_index" in df.columns
        has_ds_url = "dataset_url" in df.columns
        start = len(items)
        for i, row in df.head(150).iterrows():
            patient = str(row.get("patient", ""))[:600]
            doctor = str(row.get("doctor", ""))[:400]
            text = patient + " " + doctor
            if not text.strip() or text == "nan nan":
                continue
            sid = f"H{len(items)+1}"
            idx = row["case_index"] if has_index else int(i)
            url = str(row["dataset_url"]) if has_ds_url else HF_DATASET_URL
            label = f"AI Medical Chatbot dataset (case #{idx})"
            items.append({"id": sid, "text": text, "url": url, "label": label})
            source_map[sid] = {"url": url, "label": label}
        print(f"  Loaded {len(items) - start} from HuggingFace")

    return items, source_map


def synthesize_trust_patterns(items, source_map):
    """Use Claude to derive trust breakdown taxonomy; return patterns with quoted sources."""

    combined = "\n---\n".join([f"[{x['id']}] {x['text']}" for x in items[:120]])

    prompt = f"""You are a product researcher analyzing real user experiences with health AI tools.

Below are raw user experiences. Each block is prefixed with a source ID in brackets (e.g. [R1], [A2], [H3]) — Reddit, app store reviews, or HuggingFace AI Medical Chatbot dataset.

Your task: Identify exactly 5 distinct "trust breakdown patterns" — the core ways health AI companions fail to earn or maintain user trust. Use these exact pattern names:
1. "The Locked Door Loop"
2. "Symptom Checker Dead End"
3. "Dangerous Confidence Gap"
4. "Cold at the Worst Moment"
5. "Monetization Erodes Credibility"

For each pattern provide:
1. pattern_name (one of the 5 names above)
2. description (one sentence)
3. quotes: array of 1–3 objects with "quote" (verbatim substring from the data) and "source_ref" (the ID in brackets, e.g. "A3" or "H7")
4. why_it_matters
5. product_intervention

Use ONLY verbatim quotes from the data below. source_ref must be one of the IDs that appear in the data. When multiple sources exist (e.g. different apps like WebMD, Teladoc, MDLive, Symptomate, or HuggingFace dataset), prefer spreading quotes across sources—do not use only one app for all patterns.

Return ONLY a JSON array with 5 objects. No preamble. No markdown. Just the JSON.

Schema:
[
  {{
    "pattern_name": "The Locked Door Loop",
    "description": "...",
    "quotes": [{{ "quote": "exact text from data", "source_ref": "A2" }}],
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
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    patterns = json.loads(raw)

    # Attach source_url and source_label to each quote
    for p in patterns:
        linked = []
        for q in p.get("quotes", []):
            if isinstance(q, dict):
                ref = q.get("source_ref", "")
                info = source_map.get(ref, {})
                linked.append({
                    "text": q.get("quote", ""),
                    "source_url": info.get("url", ""),
                    "source_label": info.get("label", ref),
                })
            else:
                linked.append({"text": str(q), "source_url": "", "source_label": ""})
        p["quotes"] = linked

    return patterns


def main():
    print("Loading data with sources...")
    items, source_map = load_data_with_sources()

    if len(items) < 10:
        print("\nNot enough data found. Run steps 1-3 first.")
        print("Using demo mode with placeholder data...")
        patterns = demo_patterns()
    else:
        print(f"\nTotal data points: {len(items)}")
        patterns = synthesize_trust_patterns(items, source_map)

    # Save results
    with open("../outputs/trust_patterns.json", "w") as f:
        json.dump(patterns, f, indent=2)
    try:
        with open("../dashboard/trust_patterns.json", "w") as f:
            json.dump(patterns, f, indent=2)
    except Exception:
        pass

    print(f"\nSynthesis complete! Found {len(patterns)} trust patterns.")
    for i, p in enumerate(patterns, 1):
        print(f"  {i}. {p['pattern_name']}")

    print("\nPatterns saved to outputs/trust_patterns.json and dashboard/trust_patterns.json")
    print("Now open: dashboard/index.html")


def demo_patterns():
    """Demo patterns when real data isn't available yet. Quotes have no source links."""
    return [
        {
            "pattern_name": "The Locked Door Loop",
            "description": "Paywalls at the moment of peak clinical need.",
            "quotes": [{"text": "Locked me out when I needed help the most.", "source_url": "", "source_label": "Demo"}],
            "why_it_matters": "If the front door is broken, users conclude the company is incompetent.",
            "product_intervention": "Zero-auth guest mode for core health features."
        },
        {
            "pattern_name": "Symptom Checker Dead End",
            "description": "Core health assessment fails or returns unhelpful jargon.",
            "quotes": [{"text": "More useless than Googling. At least Google gives me something to go on.", "source_url": "", "source_label": "Demo"}],
            "why_it_matters": "Users lose trust in the AI's ability to understand them.",
            "product_intervention": "Conversational symptom input with clear error messages."
        },
        {
            "pattern_name": "Dangerous Confidence Gap",
            "description": "AI presents health assessments with apparent authority but gets critical details wrong.",
            "quotes": [{"text": "It told me it was definitely anxiety. I had a pulmonary embolism.", "source_url": "", "source_label": "Demo"}],
            "why_it_matters": "Overconfident responses could delay care for serious conditions.",
            "product_intervention": "Calibrated uncertainty and hedging for high-stakes queries."
        },
        {
            "pattern_name": "Cold at the Worst Moment",
            "description": "AI responds to emotionally charged disclosures with clinical, robotic language.",
            "quotes": [{"text": "It gave me a Wikipedia article when I needed someone to talk to.", "source_url": "", "source_label": "Demo"}],
            "why_it_matters": "Emotional attunement separates a trusted companion from a database.",
            "product_intervention": "Emotional context classifier and empathetic response mode."
        },
        {
            "pattern_name": "Monetization Erodes Credibility",
            "description": "Features users depend on are abruptly removed or gated.",
            "quotes": [{"text": "Can't build a health habit on a tool that keeps changing on me.", "source_url": "", "source_label": "Demo"}],
            "why_it_matters": "Longitudinal trust collapses when the platform is not reliable.",
            "product_intervention": "Health feature stability promise and sunset notices."
        }
    ]


if __name__ == "__main__":
    main()
