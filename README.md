# Wake-Sleep Model Society (WSMS)

Code and results for the paper "A Two-Signal Theory of Memory Consolidation in Language Models: Trust Layers, Source Monitoring and a Wake-Sleep Model Society".

A language model reads a simulated 30-day stream of documents and user remarks. Every night it writes what it learned into a LoRA adapter ("sleep"). This repository measures the false memories that this produces and tests a write rule that removes most of them:

- A memory is written only if two different models both judge that the source states the answer.
- A document answer is written as a fact. A user remark is written with its source inside the memory: `According to the user (unconfirmed): ...`.
- A remark that contradicts a stored fact, or what both models already know, is not written.
- Every night also rehearses refusals and the frozen model's own answers to general questions.

## Results

All values are percentages after 30 nights, mean ± standard deviation over three streams (seeds 1 to 3).

**Qwen2.5-1.5B-Instruct (helper: Qwen3-4B-Instruct)**

| Policy | Recall | False assertions | Control facts | Hallucination | MMLU |
|---|---|---|---|---|---|
| Before sleep | – | – | 100.0 | 67.3 | 57.3 |
| Naive sleep (`naive`) | 84.9 ± 2.5 | 75.1 ± 2.1 | 72.5 ± 3.4 | 100.0 ± 0.0 | 42.3 |
| Document-only (`doc_only`) | 57.8 ± 4.3 | 22.6 ± 1.3 | 62.7 ± 24.5 | 100.0 ± 0.0 | 46.4 |
| Oracle (`oracle`) | 86.1 ± 1.4 | 2.5 ± 0.4 | 92.2 ± 3.4 | 100.0 ± 0.0 | 49.6 |
| WSMS (`wsms`) | 82.8 ± 2.4 | 2.5 ± 0.8 | 86.3 ± 14.8 | 62.0 ± 7.2 | 51.6 |
| WSMS, no claim tag (`wsms_notag`) | 67.8 ± 2.8 | 9.6 ± 1.4 | 92.2 ± 13.6 | 71.3 ± 13.3 | 53.8 |
| WSMS, no helper model (`wsms_solo`) | 74.5 ± 2.9 | 6.8 ± 2.5 | 29.4 ± 11.8 | 64.0 ± 8.7 | 50.4 |
| WSMS, no support signal (`wsms_nosup`) | 83.3 ± 3.4 | 22.5 ± 1.5 | 17.6 ± 10.2 | 67.3 ± 9.2 | 49.8 |
| WSMS, no conflict veto (`wsms_noveto`) | 80.9 ± 4.8 | 2.5 ± 1.1 | 27.5 ± 27.2 | 70.0 ± 19.3 | 51.0 |
| WSMS, no rehearsal (`wsms_nohomeo`) | 83.6 ± 1.3 | 3.8 ± 1.1 | 7.8 ± 9.0 | 100.0 ± 0.0 | 39.9 |

**Qwen3-4B-Instruct (helper: Phi-4-mini-instruct)**

| Policy | Recall | False assertions | Control facts | Hallucination | MMLU |
|---|---|---|---|---|---|
| Before sleep | – | – | 100.0 | 6.0 | 74.0 |
| Naive sleep (`naive`) | 79.6 ± 2.0 | 42.9 ± 0.5 | 100.0 ± 0.0 | 92.0 ± 8.7 | 44.3 |
| Document-only (`doc_only`) | 52.6 ± 1.6 | 2.5 ± 0.4 | 33.3 ± 12.2 | 84.0 ± 3.5 | 60.9 |
| WSMS (`wsms`) | 80.0 ± 1.1 | 0.6 ± 1.0 | 96.1 ± 3.4 | 59.3 ± 24.1 | 66.9 |
| WSMS, refusal dose 0.40 (`wsms_r40`) | 78.8 ± 1.5 | 0.6 ± 0.2 | 96.1 ± 3.4 | 22.0 ± 5.3 | 67.3 |

- **Recall**: true facts for which the model gives the true value, with or without the user-claim tag.
- **False assertions**: opportunities for a false memory in which the model states the false value as a plain fact.
- **Control facts**: real-world facts that were never targeted and are still answered correctly.
- **Hallucination**: questions about 50 people who never appear in the stream that get an answer instead of a refusal.
- **MMLU**: accuracy on 300 questions.

![Recall and false assertions over 30 nights](results/figures/fig2_curves.png)

## Files

| File | Purpose |
|---|---|
| `world.py` | Generator of the stream: invented people, documents, true and false user remarks, questions and answer scoring. |
| `engine.py` | Model wrapper: answering, yes/no judgement, LoRA training for one night, MMLU. |
| `wake.py` | Awake phase of one stream with the frozen models (reader answers, support judgements, closed-book answers, anchor answers). |
| `run.py` | 30 nights of consolidation for one policy and one stream. |
| `metrics.py`, `aggregate.py` | Metrics of one run; `summary.json` and the result table of one model. |
| `gate_theory.py` | Measured error rates of the two gate signals. |
| `figures.py` | Figures. |
| `run_all.sh` | Every run of the paper. |
| `results/<model>/<policy>_s<seed>.json` | One file per run: arguments, the facts of the stream, the gate decision for every candidate memory, answer codes after every third night, final answers. |
| `results/<model>/summary.json`, `results/table_<model>.txt` | Aggregated metrics. |
| `results/wake/` | Awake-phase caches and anchor answers used by the runs. |
| `results/theory_*.json` | Output of `gate_theory.py`. |

Answer codes in the run files: 0 other, 1 true, 2 false, 3 refusal, 4 / 5 / 6 true / false / other given as a user claim.

## Reproduce

```bash
pip install -r requirements.txt
./run_all.sh                                   # all runs; a run whose result file exists is skipped
python aggregate.py results/qwen2.5-1.5b
python aggregate.py results/qwen3-4b
python gate_theory.py q15 results/theory_q15.json 1 2 3
python gate_theory.py q4 results/theory_q4.json 1 2 3
python figures.py results results/figures
```

The result files of all reported runs are included, so every command after `run_all.sh` works without running a model. To run the experiments again, point `WSMS_ROOT` to an empty folder: `WSMS_ROOT=rerun ./run_all.sh`.

## Models and data

- Sleeping model Qwen/Qwen2.5-1.5B-Instruct with helper Qwen/Qwen3-4B-Instruct-2507, and sleeping model Qwen/Qwen3-4B-Instruct-2507 with helper microsoft/Phi-4-mini-instruct.
- The stream is synthetic. All people, employers and towns are invented by `world.py`.
- Anchor questions for rehearsal come from WebQuestions (`stanfordnlp/web_questions`). General ability is measured on MMLU (`cais/mmlu`).
