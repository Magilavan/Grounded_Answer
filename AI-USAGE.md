# AI Usage

## Overview

AI tools were used as development assistance during the implementation of
this project.

The overall solution approach, system architecture, technology decisions,
RAG pipeline design, evidence-grounding strategy, temporal policy approach,
evaluation strategy, and major engineering decisions were planned and
directed by the project author.

AI assistance was primarily used for code implementation, debugging,
refactoring, test support, and documentation drafting.

The project author reviewed, tested, modified, and validated the resulting
implementation.

---

## What AI Was Used For

AI assistance was used mainly in the following areas:

- Writing and modifying Python code for components defined by the project
  architecture.
- Implementing retrieval, reasoning, generation, citation, and CLI
  functionality based on the planned design.
- Assisting with debugging when tests or runtime behavior exposed issues.
- Suggesting code-level fixes and refactoring existing implementations.
- Assisting with test implementation and updates.
- Assisting with documentation and README preparation.
- Helping inspect runtime errors and test failures during development.

AI-generated code was not accepted blindly. The implementation was run
locally, tested against the project requirements, and modified when the
observed behavior did not match the expected behavior.

---

## What Was Planned and Designed by the Project Author

The project author was responsible for the overall technical direction
and solution design, including:

- Understanding and interpreting the Brite Spark Problem 1 requirements.
- Designing the overall RAG architecture.
- Deciding the separation between retrieval, reasoning, evidence
  verification, temporal policy resolution, and answer generation.
- Deciding how the system should distinguish between ANSWER, REFUSE,
  and CONFLICT outcomes.
- Designing the evidence-grounding approach.
- Designing the temporal policy handling required for the Day-2 amendment.
- Deciding how relevant dates such as determination dates, change dates,
  and claim periods should affect policy applicability.
- Designing the approach for handling transitional provisions.
- Deciding that unsupported policy questions should be refused rather
  than answered using general model knowledge.
- Defining the evaluation strategy and the scenarios that needed to be
  tested.
- Reviewing the behavior of the complete system and deciding which
  implementation changes were required.

The architecture and engineering decisions were made first, with AI used
primarily to assist with implementation.

---

## Development and Debugging

The implementation was repeatedly run against the automated test suite
and real policy questions.

When failures occurred, the failure output was examined to determine
whether the issue was related to retrieval, evidence verification,
temporal reasoning, refusal behavior, generation, or another component.

AI assistance was then used to help implement code-level changes where
appropriate.

The final implementation was validated through the project's automated
tests and manual CLI testing.

---

## Evaluation

The evaluation questions were designed around the actual policy content
and the required challenge behaviors.

The evaluation focused on areas including:

- Direct grounded answers
- Evidence-supported answers
- Refusal when evidence is insufficient
- Missing date context
- Pre-amendment policy rules
- Post-amendment policy rules
- Transitional provisions
- Claim periods spanning an amendment boundary
- Citation correctness
- Contradiction handling

Evaluation results were obtained by running the implemented system rather
than manually changing results to improve the reported outcome.

---

## Models Used

### Answer Generation

The project can optionally use an OpenAI-compatible chat-completions API
for answer generation.

The LLM is used only after the retrieval and evidence-processing stages.
The policy corpus remains the source of truth.

The system is designed so that unsupported information should not be
introduced by the generation model.

### Embeddings

The project uses TF-IDF-based representations rather than a separately
downloaded embedding model.

### Reranking

The project uses a lexical/heuristic reranking approach.

---

## What AI Was Not Responsible For

AI was not treated as the source of truth for policy decisions.

The following were not delegated to an LLM as authoritative decisions:

- Which policy rule applies to a date.
- Whether evidence is sufficient.
- Whether a question should be refused.
- Whether a policy conflict exists.
- Which policy clause is authoritative.
- Whether a citation is valid.

These decisions are implemented in the project's retrieval and reasoning
pipeline and are validated through tests.

---

## Human Responsibility

The project author is responsible for the submitted implementation.

AI assistance does not replace understanding of the code or architecture.
The project author reviewed and tested the implementation and is
responsible for explaining the design, implementation choices, and
behavior of the system during evaluation.
