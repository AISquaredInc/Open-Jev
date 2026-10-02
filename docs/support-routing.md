# Support routing with a review queue

The workflow takes human-written support text or a CSV, proposes a BANKING77
intent with real local Open-Jev probabilities, applies a frozen confidence
policy, lets a reviewer correct uncertain rows, and exports a final CSV.
Unreviewed uncertain rows remain unresolved; it never performs banking actions.

The English interface is available from the existing model server at
`http://127.0.0.1:8791/examples/support-routing/`. It contains 12 natural
official-training examples and a label index. It calls the server that served
the page; it has no simulated model or fallback oracle. The displayed model
identity and probability vector come from the response. Without an imported
calibration lock, every predicted row waits for human review.

## Prepare and freeze the evaluation

```bash
python -m scripts.evaluate_support_routing prepare \
  --output data/support-routing-20261002 \
  --calibration-rows 96 --test-rows 256 \
  --demo-dir examples/support-routing
```

The script downloads the two BANKING77 CSVs at fixed upstream revision
`57ec275d8078af65b7731c2a98be812d844a6d6b` and verifies their known SHA-256s.
The original dataset has 10,003 train and 3,080 test utterances across 77
human-labeled intents. Attribution: Casanueva et al. (2020), *Efficient Intent
Detection with Dual Sentence Encoders*, PolyAI BANKING77, [CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data).

Normalized official-test texts are reserved before deduplicating train.
Calibration takes 96 deterministic whole utterances from official train;
the remaining 9,851 training utterances build 77 intent documents for BM25.
The official test sample is independently shuffled with the saved seed and
never enters retrieval training or threshold fitting. Inputs and gold are
stored separately, and every input file is frozen in `manifest.json`.

Candidate selection uses the text alone: BM25 selects eight labels from the
77-label catalog, followed by an explicit human-review candidate. Gold never
inserts the correct candidate. This means routing failures can come from
retrieval or from the model; candidate recall and retrieval top-1 accuracy
make that distinction visible. This is a nine-candidate retrieval-and-routing
pipeline, not a direct 77-way classifier score.

## Measure a released checkpoint

Start the actual 2B, 9B or 27B server using the usual pinned local checkpoint
command. Then, from a second terminal:

```bash
python -m scripts.evaluate_support_routing evaluate \
  --dataset data/support-routing-20261002 \
  --endpoint http://127.0.0.1:8791/v1/systemone \
  --output runs/support-routing-2b-20261002
```

The runner measures calibration first, chooses the highest empirical coverage
that reaches 90% accepted accuracy with at least ten accepted calibration
rows, and writes `decisions.lock.json` **before collecting test predictions or
opening test gold**. If no threshold meets the policy, it sends every row for
review. It never retunes on test. Calibration and test must use the same model,
checkpoint hash, base revision, temperature, method, code revision and context
limit. Identity drift stops the run.

`summary.json` reports routing accuracy, candidate recall, accepted coverage,
error among accepted decisions, review rate, failed requests and HTTP wall
time. Constant-majority and retrieval-top-1 baselines use the same test rows.
All failures remain in the denominator; raw probability vectors and requests'
hashes are kept in the prediction journals. The threshold is an empirical
calibration selection, not a statistical guarantee. With 256 test rows,
accepted-error estimates may remain uncertain.

Import the measured `decisions.lock.json` into the browser before routing its
queue. A lock from another model is rejected. All measurements in a claim
must name the actual checkpoint size; offline fixture tests are not model
performance.

## Finish human review

The browser provides a catalog selector for every row and requires a reviewer
name for human corrections. Export either the review queue or the final CSV.
The CLI supports the same last step: edit `reviewed_label` and `reviewer` in
the saved review queue, then run:

```bash
python -m scripts.evaluate_support_routing finalize \
  --queue runs/support-routing-2b-20261002/review-queue.csv \
  --index data/support-routing-20261002/index.json \
  --output runs/support-routing-2b-20261002/final.csv
```

The export records whether the final label came from the model, from a named
human, or remains unresolved. It does not infer a review label from gold.
The actual human-review completion count must come from saved reviews; merely
testing the correction mechanism with fixtures is not a completed human pilot.

## Limits

Released checkpoints may already have trained on BANKING77 official train.
This is a supervised intent-routing evaluation, not an unseen-task zero-shot
claim. This campaign's calibration and retrieval subsets are separate from
each other, but they were not necessarily absent from older checkpoint
training. BANKING77 supplies no official out-of-scope requests, so review
rates are abstention and failure rates, not a measured OOS detection score.
The natural dataset represents a benchmark, not new production users.

```bash
python -m unittest tests.test_support_routing
```
