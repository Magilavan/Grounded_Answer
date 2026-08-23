# Design Decisions

This document is meant to be read alongside the code, not instead of it.
It records *why* things are the way they are, including the mistakes
found and fixed along the way — the brief asks for honesty here, and the
most useful honesty is showing what didn't work first.

## 1. Architecture decisions

Retrieval, evidence verification, decision, generation, and citation
validation are five separate modules with no component making a decision
that belongs to another (`app/pipeline.py` wires them together in one
short function). The brief calls out a "day two requirements change" as
a design constraint; the concrete test applied while building was: could
a specific requirement change be made by editing one file? A new
relevance threshold — `config.py`. A different LLM provider — `config.py`
+ `generator.py`'s single `_call_llm`. A stricter contradiction rule —
`contradiction_detector.py` only. A different refusal wording —
`refusal.py` only. This was actually exercised, not just designed for: the
reranker, evidence thresholds, and contradiction detector were each
revised multiple times during development (see sections 6-9 below)
without touching retrieval, generation, or the CLI.

## 2. Why hybrid retrieval (BM25 + vector) was selected

Pure semantic search misses exact clause references, dollar figures, and
day counts — a question naming "§4.3.2" or "$4,000" needs an exact match,
which cosine similarity over dense-ish vectors doesn't reliably prioritise.
Pure keyword search misses paraphrases with no shared vocabulary. Neither
alone covers the range of question styles the evaluation set is required
to include (direct, paraphrased, exact-terminology, numeric). Both are
kept and fused rather than picked between.

## 3. Why BM25 (hand-rolled, not `rank-bm25`)

No network access was available in the development sandbox to install
`rank-bm25`, and — more importantly for the shipped project — a clean
clone shouldn't need it either: BM25 is one well-specified formula, and
implementing it directly (`app/retrieval/bm25_search.py`) removes a
dependency for about 40 lines of code. Standard `k1=1.5`, `b=0.75`.

## 4. Why vector search is TF-IDF/cosine, not sentence-transformers

This is the biggest deviation from the brief's suggested stack, so it
gets the most explanation.

**What was tried:** the brief recommends ChromaDB + sentence-transformers
+ a cross-encoder reranker. The development environment for this project
had no network access at all — `pip install sentence-transformers`,
`pip install chromadb`, and even `pip install rank-bm25` all failed with
no matching distribution, and no model weights could be downloaded even
if the packages installed. This isn't a hypothetical concern for the
delivered project either: "clone into a clean environment and run using
the README instructions alone" (brief, top) is a real constraint, and a
dependency chain that silently requires an internet connection at
`pip install` time (to fetch `torch`) and again at first run (to download
model weights from the Hugging Face Hub) is a fragile thing to hand
someone as "clone and run."

**What was built instead:** scikit-learn's `TfidfVectorizer`
(word-level unigrams + bigrams) with cosine similarity, wrapped behind an
`EmbeddingBackend` interface (`app/retrieval/embeddings.py`) so a neural
backend is a same-shape swap later, not a rewrite. For a ~150-clause,
~20-page manual, TF-IDF is not a compromise on capability so much as a
reasonable match for the corpus size — the risk TF-IDF genuinely carries
(missing a paraphrase with near-zero shared vocabulary) is real and is
visible in the evaluation results (Q09), not hidden.

**A secondary, genuine advantage**: TF-IDF match decisions are directly
inspectable (which words overlapped, weighted by which corpus
frequencies), which mattered while debugging the reranker (see section 6)
— a wrong ranking from a neural embedding would have been much harder to
diagnose than "these two clauses share generic template vocabulary and
the distinctive word didn't get enough weight."

## 5. Why Reciprocal Rank Fusion

RRF combines the two retrievers using rank position, not raw score —
which matters because TF-IDF cosine similarity and BM25 scores live on
different, incomparable scales, and normalising-and-averaging them
requires assumptions about score distribution that are easy to get wrong
on a small corpus. RRF's only parameter (`k=60`, the standard default)
controls how much weight lower ranks get; not tuned further since the
reranking stage downstream does the real precision work.

## 6. Reranking — including a bug found and fixed twice

The brief recommends a cross-encoder. This project uses a heuristic
scorer (`app/retrieval/reranker.py`) for the same reason as section 4 —
no ability to fetch model weights — behind a `Reranker` interface so a
real cross-encoder is a same-shape swap.

The heuristic went through two real, evaluation-driven fixes, both worth
recording because they show the actual failure modes of naive lexical
reranking, which is exactly what the brief's "apparent gap" section warns
about:

**Fix 1 — plain overlap wasn't enough.** The first version scored
candidates by the fraction of question content-words found in the
passage. Testing against a full-time-student question, it ranked §7.3.2
("needs figure increased $140/month for a dependent child under 2") above
§7.1.3 (the clause that actually mentions full-time students), because
§7.3.2 shared more *words* with the question ("needs", "figure",
"household", "includes") — just not the word that mattered ("student").
**Fix:** weight each matched token by its corpus IDF, computed natively
from the clause corpus using the same stemming rules the reranker itself
applies (an earlier version borrowed IDF from the TF-IDF vectorizer's
raw, unstemmed vocabulary, which caused a second, subtler mismatch when
stemming was added — see the module docstring in `reranker.py` for the
exact history).

**Fix 2 — a proper noun distorted IDF.** A question containing "Calder
County" — the program's own jurisdiction name — was scoring an unrelated
eligibility clause as highly relevant, purely because both mentioned
"Calder County". In this small, ~150-clause corpus, "calder" and "county"
happen to appear in only 5-8% of clauses (most clauses just say "the
Department" without restating the county name), so plain IDF ranked them
as *statistically* distinctive, even though they're semantically inert —
comparable to a company's internal search engine treating its own company
name as a rare, meaningful search term. **Fix:** a short, explicitly
justified list of jurisdiction/manual-identity terms (`calder`, `county`,
`manual`) held out of the weighting, the same way grammatical stopwords
are held out.

**Also added:** a floor requiring at least two shared content words before
overlap counts at all (a single incidental match — e.g. "capital" in "the
capital of France" hitting a clause about "capital expenditure" — was
otherwise enough to clear the relevance threshold), and light,
hand-written suffix stripping (not a full stemmer) so that "definition"
and a section titled "Definitions" are recognised as the same idea.

Both fixes are now regression tests in `tests/test_retrieval.py`.

## 7. Evidence verification strategy

This is the layer the brief identifies as most important, and it's the
one built last and iterated on most. The core mechanism: any clause whose
text hands the substantive answer off to another clause or section (via
"see §X.X", "under §X.X", or "addressed separately") is checked for
whether that hand-off actually resolves — does the target clause's
content genuinely overlap with the question, or does it just exist? This
is generic (implemented as a pattern, not keyed to specific clause IDs)
and it's what catches the manual's actual apparent gap: §7.1.3 defers to
§5.4 (care allowances, not students), and §3.2.3/§5.2.3 defer to no named
clause at all.

**A second real bug, found via evaluation, is worth recording.** After
lowering the support-score threshold to fix under-answering on legitimate
direct questions (section 8 below), the full-time-student question started
answering again — not from §7.1.3 (correctly disqualified by the deferral
check) but from §1.4.6, the *definition* of "full-time student", which
happens to score reasonably well against the question purely because it
shares the phrase "full-time student" without addressing the calculation
being asked about at all. This is the same "topically relevant, doesn't
establish the answer" trap the deferral check exists for, just in a form
the deferral check doesn't catch (there's no cross-reference to resolve —
it's simply the wrong clause for the question). **Fix:** when the single
highest-relevance candidate for a question is the one disqualified by an
unresolved deferral, that's treated as a strong signal the question has
hit this failure mode, and any remaining candidate must be almost as
relevant (≥85% of the disqualified clause's relevance) to be accepted as
independent support — a distant runner-up doesn't get to quietly become
the answer. This is now `tests/test_refusal.py::test_apparent_gap_question_is_refused_not_answered`.

## 8. Refusal threshold — and why it moved

Two thresholds gate every decision: `MIN_RELEVANCE_SCORE` (does this
clause even get considered) and `MIN_SUPPORT_SCORE` (does it clear the
bar to actually back an answer). The support threshold started at 0.34
(chosen before the IDF-weighting fix in section 6 existed) and had to be
lowered to 0.20 afterward — IDF-weighting produces a different score
distribution than plain overlap did, and several genuinely-answerable
direct questions (e.g. the household resource limit, §2.4.1) were being
wrongly refused as "ambiguous" at the old threshold. Lowering it re-opened
the door to the false-positive in section 7, which is what motivated that
fix. The two changes are connected: neither threshold should be read as
tuned in isolation from the other safeguards around it.

A second safeguard, added once the threshold was lowered: when more than
one clause clears the support bar, any clause scoring below 70% of the
strongest one is dropped from the final answer, so a dominant single-clause
answer (e.g. §2.4.1 for the resource limit) doesn't get diluted by
weakly-related clauses that happen to also clear a floor that has to stay
low enough to admit genuine multi-clause answers.

**This did not fully solve the multi-clause case** — see the honest
failure on Q05 in `evaluation/RESULTS.md`: a question deliberately
requiring two clauses (§8.3.1 and §8.3.3) only retrieves one, because the
70% margin (needed to keep single-clause answers clean) also excludes the
second, more weakly-scored clause of the genuine multi-clause pair. This
is a real, unresolved tension between "don't dilute a clear answer" and
"don't miss a legitimate second clause," and it's flagged as the first
thing to improve (section 15).

## 9. Contradiction handling — tuned against real false positives

The first version of the contradiction detector was pure generic overlap:
two clauses with high content-word Jaccard similarity that both stated a
number with the same unit (days, weeks, per cent), and disagreed. Run
across the *whole* 148-clause corpus as a sanity check (not just the
question that motivated it), this produced 13-18 false positives — pairs
of clauses governing completely different administrative deadlines
(application review, appeals, panel hearings) that happened to share
generic phrasing like "within N days" and "determination". Lowering the
threshold enough to catch the manual's real contradiction (§4.3.2 vs.
§9.1.4, whose lexical overlap sits at 0.21 — just under the safe generic
threshold of 0.22) made the false-positive problem much worse, not
better.

**Fix:** a second, higher-precision detection path was added first:
looking specifically for a clause that explicitly *asserts* what another
clause requires ("...required under §4.3", "...permitted under §X.Y")
with a number that doesn't match what the referenced clause actually
states. This is a much stronger, checkable signal than generic topical
overlap, and it's exactly the pattern the manual's real contradiction
uses. The generic overlap path is kept as a fallback for a contradiction
that isn't phrased as an explicit cross-reference, but its threshold was
raised to 0.40 — high enough that a corpus-wide sweep (`tests/
test_contradictions.py::test_contradiction_count_is_small_and_precise`)
finds *only* the one genuine contradiction, twice (once per numeric
mention in §4.3.2).

A further guard excludes any numeric claim introduced by an explicit
modifier phrase ("extended to", "increased to", "in place of") — this is
what stops §3.2.1 (28 days) / §3.2.2 (deliberately extended to 90 days)
from being flagged as a contradiction. That pair is a signalled,
intentional extension of the same period, not a silent inconsistency, and
the manual states it that way explicitly.

The pipeline also expands the contradiction-checking candidate pool to
include any clause explicitly cross-referenced by a retrieved clause
(resolving both full clause references and section-level references),
because otherwise a contradiction where only one side is lexically
similar enough to the question to be independently retrieved would never
be detected.

## 10. Citation strategy

Citations are never generated freely — they come only from retrieval
metadata (`clause.citation`, set once at parse time from the manual's own
numbering). The deterministic answer path is correct by construction
(built directly from the same `supporting_clauses` list used for its
citations) and is not re-validated by scanning its own rendered text —
see the "Citation validation" entry below for why that specific
safety-net check was removed after it broke a real, correct answer.

## 11. Citation validation strategy

`app/citations/validator.py` checks two independent things for any
citation: does the clause exist in the manual at all (catches invention),
and was it actually part of the evidence used for *this* answer (catches
a real-but-irrelevant clause number being cited — tested explicitly in
`tests/test_citations.py::test_real_clause_not_in_evidence_is_rejected`).

**A real bug, found and fixed:** an earlier version ran this same
full-text scan on the deterministic answer path too, "for safety". It
broke a correct answer: §1.4.1 (the definition of "applicant"), quoted
verbatim, happens to mention "§2.1.2" in passing as part of the manual's
own wording — not a claim the answer was making. The validator flagged it
as an unverified citation and the system refused a question it should
have answered. Full-text citation scanning now only runs on LLM-produced
text, where an invented citation is a real and distinct risk; the
deterministic path is trusted by construction instead. See the comment
above `generate_response`'s ANSWER branch in `generator.py` for the exact
reasoning kept in place to stop this regressing.

## 12. What was not implemented

- A neural embedding model and cross-encoder reranker (see sections 4 and
  6) — deliberately substituted, with a clean interface for swapping one
  in.
- ChromaDB — substituted with a local, persisted TF-IDF vector store
  (`app/retrieval/vector_store.py`) that plays the same architectural
  role (build once, query many times, vectors + metadata together).
- A FastAPI HTTP layer. The brief explicitly says a CLI is sufficient and
  not to spend time on a frontend; that guidance was followed literally.
- Query rewriting / spelling correction / synonym expansion — the
  paraphrase failure in the evaluation set (Q09) is exactly the case this
  would help with; see section 15.
- A distinct `OUT_OF_SCOPE` refusal *reason* in practice, even though the
  category exists in `RefusalReason`. Out-of-scope questions currently
  surface as `INSUFFICIENT_EVIDENCE` (nothing scored above the relevance
  floor) rather than a separately-detected "this isn't even the right
  domain" signal. The user-facing behavior is correct (refuse, with
  escalation guidance); the internal reason code is less specific than it
  could be.

## 13. What was intentionally cut

- Multi-turn conversation / follow-up question handling. Each question is
  answered independently; there's no session memory of prior questions.
  Out of scope for what the brief asks for, and adding it would touch the
  evidence-verification layer in ways not worth the risk this late.
- A more complete `Contradiction`-pair deduplication UI (the generator
  does deduplicate identical clause pairs, but doesn't try to merge
  multiple distinct contradictions between the same two clauses into one
  combined narrative).

## 14. Known limitations

- **TF-IDF cannot bridge a paraphrase with near-zero shared vocabulary.**
  Documented honestly as an evaluation failure (Q09 in
  `evaluation/RESULTS.md`): a question about "help with everyday tasks
  like bathing or dressing" doesn't retrieve §7.3.1 ("activities of daily
  living"), because there's no lexical bridge between the two phrasings
  at all. A neural embedding model would likely close this gap; see
  section 15.
- **The multi-clause margin trade-off** (section 8): the 70%-of-top-score
  cutoff that keeps single-clause answers clean can also cut a legitimate
  second clause from a genuine multi-clause answer (Q05 in
  `evaluation/RESULTS.md`).
- **The heuristic reranker is tuned, not learned.** Its weights (0.7
  overlap / 0.2 number bonus / 0.4 clause bonus, the 85% and 70% margins
  in evidence.py, the 0.40 contradiction-overlap threshold) were set by
  testing against the specific hard cases this manual contains, not
  learned from a labelled dataset. They generalise reasonably within this
  manual (verified with the corpus-wide false-positive sweeps described
  above) but are not guaranteed to transfer to a differently-styled
  manual without retuning.
- **The stemmer is a handful of suffix rules, not a real stemmer.** It
  fixes the specific cases found (definition/Definitions,
  determination/determinations) but will miss others and could in
  principle over-stem an unanticipated word.
- **No OUT_OF_SCOPE detection** distinct from INSUFFICIENT_EVIDENCE — see
  section 12.

## 15. What would be improved first

In order:

1. **Fix the Q05 multi-clause trade-off.** The right fix is probably not
   a better margin threshold but a different question to the evidence
   layer: "does clause B state a *consequence* that clause A's rule
   leads to" is a different relationship than "clause B is another
   independent fact needed to answer the question," and conflating them
   under one relevance-margin check is the root issue.
2. **Close the paraphrase gap (Q09)** with either a small, curated
   synonym-expansion table for known policy terms (cheap, no network
   dependency) or a real sentence-embedding backend behind the existing
   `EmbeddingBackend` interface (more general, reintroduces the
   installation/network trade-off from section 4 — worth it once the
   project is no longer required to install from a fully offline clean
   clone).
3. **Give OUT_OF_SCOPE its own detection path** rather than folding it
   into INSUFFICIENT_EVIDENCE, so logs and evaluation can distinguish "no
   clause was relevant" from "this domain isn't covered at all."
4. **Replace the hand-tuned reranker weights with something learned**, if
   a labelled query/relevance dataset for this manual (or a similar one)
   is ever built.

## 16. How the architecture handles requirement changes

Concretely, the kinds of "day two" changes the brief anticipates map to:

- New refusal category or wording → `app/reasoning/refusal.py` only.
- Different relevance/support thresholds → `.env` / `app/config.py`, no
  code change.
- Swap the LLM provider → `.env` (`LLM_API_BASE`, `LLM_MODEL`); the
  request/response shape in `generator.py` already speaks the
  OpenAI-compatible chat-completions format, so any compatible provider
  works without touching that file either.
- Swap in a real embedding model or vector database → implement
  `EmbeddingBackend` (embeddings.py) and/or replace `VectorStore`
  (vector_store.py); nothing above `vector_search.py` needs to change.
- New contradiction pattern → `app/reasoning/contradiction_detector.py`
  only.
- A policy manual update → re-run `python scripts/ingest.py`; the
  chunker's clause-marker regex is the only thing that would need
  adjusting if the manual's own numbering format changed.
