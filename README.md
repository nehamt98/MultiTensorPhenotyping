# Multitensor Phenotyping on MIMIC-III

This repository builds three views from MIMIC-III and learns unsupervised phenotypes:

- Medications
- Labs
- Clinical notes (NER with negation handling)

Each view is turned into a tensor, then cleaned and aligned to the same set of patient–diagnosis pairs. Models run on the cleaned data:

- LDA baseline on a combined feature space
- Granite (single view) on medications only
- Granite (multi view) on medications, labs, and notes with shared patient and diagnosis factors

All scripts are standalone, and write their outputs under `data/`.


## Environment

Conda environment name: `spacy310`

```bash
# from the repo root
conda env create -f environment.yml
conda activate spacy310

# spaCy model needed for notes
python -m spacy download en_core_web_sm
```

Hardware notes:
- Notes extraction and the multi view Granite run are the heaviest steps.
- A GPU helps. If you are on CPU, lower the topic/rank in the scripts or subset rows.

---

## Data expected in `data/raw/`

Place the MIMIC-III CSVs you plan to use here. The pipeline expects:

- `DIAGNOSES_ICD.csv`
- `PRESCRIPTIONS.csv`
- `LABEVENTS.csv`
- `NOTEEVENTS.csv`

---

## Quick start

### 1) Build tensors

Each script runs independently and writes to `data/tensors/`.

```bash
# medications tensor: (SUBJECT_ID, ICD9_CODE, DRUG, count)
python src/build_tensors/build_meds.py

# labs tensor: (SUBJECT_ID, ICD9_CODE, LAB_CONCEPT, count)
python src/build_tensors/build_labs.py

# notes concept extraction (biomedical NER + NegEx). Heavy.
python src/build_tensors/extract_notes.py

# build notes tensor from extracted chunks:
# (SUBJECT_ID, ICD9_CODE, token, count)
python src/build_tensors/build_notes.py
```

You should see:
- `data/tensors/meds_tensor.csv`
- `data/tensors/labs_tensor.csv`
- `data/tensors/notes_tensor.csv`

### 2) Clean and align tensors

Filters extremes, trims token/lab frequencies, aligns all views to the same patient–diagnosis pairs, and keeps a top-K set used by all models.

```bash
python src/models/clean_tensors.py
```

Outputs:
- `data/processed_tensors/meds_cleaned.csv`
- `data/processed_tensors/labs_cleaned.csv`
- `data/processed_tensors/notes_cleaned.csv`

### 3) Run models

All model scripts are standalone and write to their own folders.

```bash
# LDA baseline
python src/models/baseline_lda.py        # -> data/lda_out/

# Granite single view (medications only)
python src/models/granite_single.py      # -> data/granite_out/

# Granite multi view (meds + labs + notes)
python src/models/granite_multiple.py    # -> data/granite3_out/
```

Each folder will include factor matrices (patients, diagnoses, and view-specific concepts), lambda weights per topic/component, top tokens, and simple fit summaries.

---
