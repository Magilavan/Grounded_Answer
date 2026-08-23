# AI Usage

This entire project — architecture, code, tests, evaluation set, and
documentation — was written by Claude (Anthropic), operating as an
autonomous coding agent with a sandboxed Linux environment, in direct
response to the Brite Spark 2026 Problem 1 specification and the supplied
`policy-manual.md`.

## What AI was used for

- **Reading and analysing the source documents.** The `.docx` problem
  specification and `policy-manual.md` were parsed and read in full before
  any code was written, specifically to locate the manual's intentional
  contradiction and apparent gap by close reading, not by being told
  where they were.
- **Architecture design**, including the decision to deviate from the
  brief's suggested ChromaDB/sentence-transformers/cross-encoder stack in
  favor of a fully offline-capable TF-IDF/BM25/heuristic-reranker stack
  (see `DECISIONS.md`), made because the development environment had no
  network access and the project's own requirement — "clone into a clean
  environment and run using the README instructions alone" — argued
  against a dependency chain that needs to download model weights.
- **All code**: ingestion/chunking, hybrid retrieval, evidence
  verification, contradiction detection, the answer/refuse/conflict
  decision layer, generation, citation validation, the CLI, and the test
  suite.
- **Debugging via actually running the system.** Every fix documented in
  `DECISIONS.md` — the reranker scoring the wrong clause on the
  full-time-student question, the "Calder County" false-positive, the
  citation validator breaking a correct answer, the contradiction
  detector's false positives across the corpus, the threshold
  recalibration after the IDF-weighting fix — was found by running real
  questions through the actual pipeline and inspecting the output, not
  predicted in advance. Several of these required more than one iteration
  to fix correctly.
- **The evaluation question set** was written after reading the manual's
  actual clause content (not invented), designed to cover the ten
  required categories, and then run against the live system to get the
  results in `evaluation/RESULTS.md` — including the two questions that
  honestly fail, which were kept in the set rather than replaced with
  easier questions.
- **Documentation**: README.md, DECISIONS.md, and this file.

## What was not done

- No test results, evaluation numbers, or example outputs in this
  repository were hand-written or adjusted after the fact.
  `evaluation/RESULTS.md` is the literal output of running
  `python evaluation/evaluate.py` against the code as committed.
- No clause numbers, policy rules, or manual content were invented. Every
  citation traces back to actual parsed clauses in `data/policy-manual.md`.

## Models used

- **Answer generation (optional):** any OpenAI-compatible chat completions
  API, defaulting to Groq (`llama-3.3-70b-versatile`) — see `.env.example`.
  This is optional and off by default in the sense that the system
  produces fully correct, grounded answers without it; when configured,
  it only rephrases already-verified evidence and its output is
  citation-validated before being shown. This path could not be
  exercised against a live network in the development sandbox (no network
  access); it was verified instead with a mocked HTTP response
  confirming the correct endpoint, auth header, request shape, and — most
  importantly — that a hallucinated citation in the mocked response is
  correctly rejected and falls back to the deterministic answer.
- **Embeddings:** none (deliberately) — TF-IDF via scikit-learn. See
  `DECISIONS.md`, section 4.
- **Reranking:** none (deliberately) — a heuristic, IDF-weighted lexical
  scorer. See `DECISIONS.md`, section 6.
- **The agent building this project:** Claude (Anthropic), via an
  agentic coding session with bash/file-editing tools and no internet
  access.
