import pandas as pd
import spacy
from spacy.util import filter_spans
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from negspacy.negation import Negex
import os
from pathlib import Path

base_path = Path().resolve()
chunks = 4  # how many slices to create

def extract_entities(text, nlp, ner_pipe, keep_labels):
    text = text.lower()
    entities = ner_pipe(text)
    doc      = nlp(text)

    spans = []
    for ent in entities:
        if ent["entity_group"] not in keep_labels:
            continue
        span = doc.char_span(
            ent["start"],
            ent["end"],
            label=ent["entity_group"],
            alignment_mode="expand",
        )
        if span is not None:
            spans.append(span)

    # keep longest then drop overlaps
    spans = sorted(spans, key=lambda s: s.end_char - s.start_char, reverse=True)
    spans = filter_spans(spans)
    doc.ents = spans

    # re run NegEx
    nlp.get_pipe("negex")(doc)

    out = []
    for ent in doc.ents:
        flag = "false" if ent._.negex else "true"
        token = ent.text.replace(" ", "_")
        out.append(f"{token}_{flag}")
    return out


# Load models once
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("negex", last=True)

model_name = "d4data/biomedical-ner-all"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipe   = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0  # or -1 for CPU
)

keep_labels = {"Diagnostic_procedure", "Biological_structure", "Sign_symptom"}

# Read CSV once
df = pd.read_csv(
    os.path.join(base_path, "data/raw/NOTEEVENTS.csv"),
    usecols=["ROW_ID","SUBJECT_ID","HADM_ID","TEXT"],
    dtype=str,
)

N    = len(df)
size = N // chunks + 1

# Output folder
out_dir = base_path / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

# Loop over all chunks
for chunk_index in range(1, chunks + 1):
    start = (chunk_index - 1) * size
    end   = min(start + size, N)
    if start >= end:
        continue

    sub = df.iloc[start:end].copy()

    # Extract
    sub["extracted"] = sub["TEXT"].apply(
        lambda t: extract_entities(t, nlp, ner_pipe, keep_labels)
    )

    # Dump one file per chunk, includes SUBJECT_ID and HADM_ID
    out_path = out_dir / f"NOTEEVENTS.chunk{chunk_index}.csv"
    sub.to_csv(out_path, index=False)
    print(f"Wrote {len(sub)} rows to {out_path}")