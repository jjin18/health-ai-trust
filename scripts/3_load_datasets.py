"""
STEP 3: Load & Sample Open Source Medical Dialogue Datasets
============================================================
Downloads free datasets from HuggingFace and samples
conversations that reveal trust-relevant patterns.

SETUP:
pip install datasets pandas transformers

Run: python 3_load_datasets.py
"""

from datasets import load_dataset
import pandas as pd
import random

random.seed(42)

# ── Dataset 1: MedDialog (260K real patient-doctor conversations) ──
print("Loading MedDialog...")
try:
    med_dialog = load_dataset("UCSD26/medical_dialog", "en", split="train", streaming=True)

    samples = []
    for i, item in enumerate(med_dialog):
        if i >= 2000:  # sample 2000
            break
        turns = item.get("dialogue_turns", [])
        if not turns:
            continue

        # Extract patient messages (these reveal what people actually ask)
        patient_turns = [t["utterance"] for t in turns if t.get("speaker") == "Patient"]
        doctor_turns  = [t["utterance"] for t in turns if t.get("speaker") != "Patient"]

        samples.append({
            "dataset":        "MedDialog",
            "patient_query":  " | ".join(patient_turns[:2]),
            "doctor_response": " | ".join(doctor_turns[:2]),
            "num_turns":      len(turns),
        })

    df_meddialog = pd.DataFrame(samples)
    df_meddialog.to_csv("../data/meddialog_sample.csv", index=False)
    print(f"  Saved {len(df_meddialog)} MedDialog samples")

except Exception as e:
    print(f"  Error loading MedDialog: {e}")
    df_meddialog = pd.DataFrame()


# ── Dataset 2: AI Medical Chatbot (250K patient-doctor dialogues) ──
print("Loading AI Medical Chatbot dataset...")
try:
    chatbot_ds = load_dataset("ruslanmv/ai-medical-chatbot", split="train", streaming=True)

    samples = []
    for i, item in enumerate(chatbot_ds):
        if i >= 2000:
            break
        samples.append({
            "dataset":   "AI-Medical-Chatbot",
            "patient":   item.get("Patient", ""),
            "doctor":    item.get("Doctor", ""),
            "condition": item.get("Description", ""),
        })

    df_chatbot = pd.DataFrame(samples)
    df_chatbot.to_csv("../data/ai_medical_chatbot_sample.csv", index=False)
    print(f"  Saved {len(df_chatbot)} AI Medical Chatbot samples")

except Exception as e:
    print(f"  Error loading chatbot dataset: {e}")
    df_chatbot = pd.DataFrame()


# ── Combine and flag trust-relevant patterns ──────────────────────
print("\nFlagging trust-relevant patterns...")

TRUST_KEYWORDS = [
    "don't know", "uncertain", "consult", "see a doctor", "emergency",
    "call 911", "serious", "not sure", "might be", "could be",
    "recommend", "worried", "concern", "urgent", "immediately",
    "chest pain", "suicide", "depression", "anxiety", "mental health",
]

def flag_trust_relevant(text):
    if not isinstance(text, str):
        return False
    return any(kw in text.lower() for kw in TRUST_KEYWORDS)

if not df_chatbot.empty:
    df_chatbot["trust_relevant"] = df_chatbot["patient"].apply(flag_trust_relevant)
    trust_cases = df_chatbot[df_chatbot["trust_relevant"]].copy()
    trust_cases["case_index"] = range(len(trust_cases))
    trust_cases["dataset_url"] = "https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot"
    trust_cases.to_csv("../data/trust_relevant_cases.csv", index=False)
    print(f"  Found {len(trust_cases)} trust-relevant cases from chatbot dataset")

print("\nAll datasets saved. Now run 4_synthesize.py")
