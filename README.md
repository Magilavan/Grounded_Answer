# Grounded Answer — Policy-Grounded RAG Assistant

A policy-grounded Retrieval-Augmented Generation (RAG) assistant built for the **Brite Spark 2026 Challenge**. The system provides deterministic, fully cited policy answers derived from the Calder County Household Support Program policy manual and its amendments (including *Amendment No. 2026-01.md*).

---

## 🌟 Key Features

* **Hybrid Retrieval (Vector + BM25)**: Combines dense semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`) with sparse lexical BM25 keyword matching fused via Reciprocal Rank Fusion (RRF) and custom heuristic reranking.
* **Temporal Policy Awareness (Day 2)**: Dynamically resolves policy versions based on query date context (pre-amendment vs. post-amendment vs. transitional provisions). Suppresses legacy clauses during contradiction evaluation.
* **Evidence Verification & Gap Detection**: Distinguishes topic relevance from true answer support. Refuses queries where provisions state topics are "addressed separately" without establishing actionable rules (apparent gap detection).
* **Citation Validation**: Enforces strict citation verification against retrieved supporting evidence before returning any answer.
* **Deterministic Fallback & LLM Phrasing**: Operates fully offline without external API dependencies using deterministic output, or optionally utilizes OpenAI-compatible APIs (e.g., Groq) for answer rephrasing.

---

## 📋 Prerequisites

* **Python**: `3.11` or higher
* **OS**: Windows, macOS, or Linux

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Magilavan/Grounded_Answer.git
cd Grounded_Answer
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

to enable optional LLM phrasing (e.g. Groq API):
```env
LLM_API_KEY=put_your_groq_api_key_here
LLM_ENABLED=true
LLM_MODEL=llama-3.3-70b-versatile
```


---

## ⚙️ Ingestion & Indexing

Before running queries, build the retrieval index (Vector + BM25):

```bash
python scripts/ingest.py
```

This ingests `data/policy-manual.md` and `data/Amendment No. 2026-01.md`, creating the index in `data/index/`.

---

## 💻 Running the Assistant

### Interactive CLI

Launch the interactive prompt to ask policy questions:

```bash
python -m app.cli
```

* Type your policy question at the prompt `> `.
* Type `verbose` to toggle detailed retrieval, date extraction, and score breakdown.
* Type `exit` to quit.

**Example CLI Usage:**
```text
Question:
> What is the monthly earnings disregard for a determination made on 15 March 2026?

Decision: ANSWER

Answer:
The monthly earnings disregard is $175.

Source: §6.4.1, Amendment No. 2026-01 §5.1
```

---

## 🧪 Testing & Evaluation

### Run Test Suite
To execute all 40 automated unit and integration tests:

```bash
python -m pytest -v
```

### Run Evaluation Suite
To run the evaluation harness against `evaluation/questions.json`:

```bash
python evaluation/evaluate.py
```

---

## 📁 Project Architecture

```text
├── app/
│   ├── citations/          # Citation extraction and validation
│   ├── generation/         # Answer text construction & optional LLM integration
│   ├── ingestion/          # Policy manual & amendment parsing/chunking
│   ├── reasoning/          # Evidence assessment, contradiction detection, & temporal resolution
│   ├── retrieval/          # BM25, Vector search, RRF fusion, & reranking
│   ├── cli.py              # Interactive command-line interface
│   ├── config.py           # Settings and configuration management
│   └── pipeline.py         # End-to-end question answering pipeline
├── data/
│   ├── policy-manual.md    # Consolidated policy manual
│   └── Amendment No. 2026-01.md # Day-2 policy amendment
├── evaluation/             # Evaluation dataset & benchmark runner
├── scripts/                # Indexing & setup scripts
├── tests/                  # Pytest validation test suite
├── .env.example            # Sample configuration file
├── DECISIONS.md            # Technical rationale and design log
└── README.md               # Project documentation
```
