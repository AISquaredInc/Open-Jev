# Frontier controls v4

This is an original synthetic development mixture for six failure conditions:
deadline boundaries, latest-account state, direct authorization and revocation,
negation, numeric candidate selection, and irrelevant department policies.
All labels are recomputed by executable Python oracles. These are synthetic
controls, not real customer conversations.

The previously published 27B public JevBench result informed the priorities:
`temporal_numeric` had 13 errors among 15 decisions, and `long_policy` had eight
among 19. The analysis preserves the checkpoint and aggregate checksum in
[development-error-families.json](../reports/frontier-controls-v4/development-error-families.json).
Individual original responses were not available in the local saved aggregate;
the analyzer does not invent per-example explanations. JevBench's 231 public
decisions are now development feedback, not an untouched test. No public
benchmark question, scenario or answer text, and no sealed material, enters the
generator.

## Build and audit

```bash
python -m scripts.build_frontier_controls_v4 \
  --output data/frontier-controls-v4-20261002 --groups-per-family 100
python -m scripts.analyze_jevbench_errors \
  --report reports/new27b-jevbench-20260922/report.json \
  --output reports/frontier-controls-v4/development-error-families.json
python -m unittest tests.test_frontier_controls_v4
```

Both commands refuse to overwrite their outputs. The generator needs only the
standard library. The default produces 3,360 records: 2,400 train and 240 each
for calibration, validation, test and OOD. Every four counterfactual variants
of a scenario remain in the same split; the split is never placed in the model
state. Every record follows the existing training schema and includes original
CC0-1.0 provenance, its generator revision, and a group identifier. The manifest
pins each split and the generator by SHA-256.

Three variants use dynamic Choice candidates, and one asks whether a proposed
outcome holds as a Noul. Option order is shuffled independently of the label.
The independent `audit` recomputes the answer from state facts and rejects even
a valid one-hot target if it differs from the oracle.

The OOD split uses separately named template versions, rewritten question
prefixes, new entity identities and, where applicable, new time windows and
departments. It tests controlled variations within these six workflows. It
does not establish transfer to wholly new domains or human-written requests.

## Evaluate before claiming improvement

Use the untouched synthetic test/OOD files once after choosing training and
calibration settings on their respective files. Preserve the unmodified
checkpoint baseline on the same records. The continued-training runner selects
256 train, 64 calibration and 128 each test/OOD rows by whole counterfactual
group, balancing families without looking at labels:

```bash
python -m scripts.train_frontier_controls_v4 \
  --checkpoint /path/to/released-2b-checkpoint \
  --data data/frontier-controls-v4-20261002 \
  --output runs/frontier-controls-v4-2b-20261002 \
  --steps 64 --train-rows 256 --calibration-rows 64 --eval-rows 128
python -m unittest tests.test_train_frontier_controls_v4
```

It starts from `DecisionModel.load`, explicitly enables the existing LoRA A/B
and decision-head parameters, and keeps the backbone frozen. It uses the
released temperature for the original baseline, fits the new temperature only
on independent calibration rows, and locks it before trained test/OOD inference.
Checkpoint selection is the fixed final step; test and validation do not choose
training settings or checkpoints. Split hashes, selected IDs, checkpoint files,
runner hash, code commit and trainable parameter names are recorded. A fresh
load must reproduce probabilities within 0.005 on saved Choice/Noul/OOD probes.

Report the selected IDs and input hashes, per-family accuracy, Noul Brier/ECE,
and the original-checkpoint versus adapted-checkpoint difference. Improvement
on this mixture does not imply improvement on JevBench or BANKING77.

There is no claim of completed fine-tuning or measured improvement in this
document. Numeric selection and deadline arithmetic are stress controls;
production workflows should let code compute exact arithmetic and retain the
model for the semantic decisions.
