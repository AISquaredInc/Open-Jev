# Original temporal-window controls v5

The completed 2B v4 pilot left numeric candidate accuracy at 30.0% on its
synthetic test and reduced it from 35.0% to 30.0% on controlled OOD. Timeline
OOD accuracy also fell from 58.3% to 54.2%. On the separate frozen public
development sample, temporal/numeric Choice stayed 0/3 and Noul stayed 0/3.
These are small development samples, not general population estimates.

An audit of all 560 original v4 timeline rows found no request before delivery
or exactly at delivery; all delivery timestamps used UTC. Their four-row
groups covered the upper deadline and exception route. V5 fills this specific
lower-bound and time-representation gap. V4 data, reports, checkpoints and
release weights remain frozen.

## Authored controls and independent validation

Each original scenario has four counterfactuals: one second before, exactly
at, and one second after an inclusive boundary, followed by a request after
the deadline with an approved exception. Groups alternate between the delivery
boundary and deadline boundary. Three rows are Choice and one is Noul; the
verification row rotates among the four positions. Choice option order is
shuffled, and Noul true/false proposals are balanced over complete eight-group
blocks. Case IDs are opaque, and hidden split, variant and target metadata
does not enter model prompts.

Delivery and request strings use different explicit fixed UTC offsets. The
controls include month and year crossings. Controlled OOD uses different
offsets and durations, reworded policy text, and dates around February 29
in four leap years. Seeded year/minute/second variation keeps the temporal
facts distinct across groups; changing only case IDs cannot pass the audit.
It remains the same closed-interval workflow, not a new domain. These
are fixed-offset instants; daylight-saving transitions and business calendars
are outside the scope.

The generator computes labels using aware `datetime` interval comparisons.
The independent auditor does not import the generator or its oracle. It parses
the visible timestamp fields, converts validated calendar components into
integer epoch seconds, subtracts explicit offsets and checks
`0 <= request_seconds - delivery_seconds <= hours * 3600`. An approved
exception yields review. Noul proposals are read from the visible question
and checked against metadata. The audit validates the two authored policy
wordings, not arbitrary natural-language policy semantics.

Hand-calculated tests verify equivalent instants across offsets, inclusive
lower and upper boundaries, the second immediately outside each boundary,
leap-day crossing, and exception precedence. Corrupted labels, mismatched
visible proposals, incomplete counterfactual groups and changed split hashes
are rejected.

## Frozen small preparation, October 2, 2026

```bash
python -m scripts.build_temporal_windows_v5 \
  --output data/temporal-windows-v5-20261002-r1
python -m scripts.audit_temporal_windows_v5 \
  --dataset data/temporal-windows-v5-20261002-r1 \
  --output reports/temporal-windows-v5-20261002/audit.json
python -m unittest tests.test_temporal_windows_v5
```

The default freezes 256 rows in 64 complete groups: 128 train rows and 32 each
for calibration, validation, test and OOD. There are 192 Choice and 64 Noul
rows. Every split covers both boundary sides, different displayed offsets,
month/year crossings and exception overrides. OOD also covers leap day.
Groups, cases and model inputs are disjoint across splits. Dataset creation
and report creation refuse to overwrite their paths. The CLI audit output's
parent directory must already exist; choose fresh output paths when replaying
the committed preparation.

The [copied manifest](../reports/temporal-windows-v5-20261002/manifest.json)
records exact generator settings and hashes for all five split files. The
[independent audit](../reports/temporal-windows-v5-20261002/audit.json) pins
the manifest, generator and validator hashes and records boundary/offset
coverage. The raw original CC0 data stays in the ignored `data/` directory
and can be rebuilt from the pinned source and settings. No public benchmark
question, scenario, answer, or sealed material enters either implementation.

## Next evaluation decision

No model has trained on or been evaluated against this preparation. It carries
no accuracy or JevBench improvement claim. The original numeric balance
controls also lack negative/zero balances and ledger-order variation; v5 does
not fix that separate gap.

Before the next pilot, predeclare a fixed training schedule and a separately
frozen comparison with the released checkpoint, retaining unaffected families
as regression controls. Fit temperature only on calibration, use validation
for any development choices, and keep the new test/OOD independent of those
choices. Report Choice accuracy, Noul coverage/error under the intended
thresholds, and performance by boundary and offset representation. Account
for calibration changes separately when interpreting probability metrics.
Treat the already-inspected frozen public64 sample as development feedback;
it cannot become a new blind test. Production code should continue to compute
exact dates and amounts directly.

The [predeclared pilot plan](../reports/temporal-windows-v5-20261002/pilot-plan.json)
freezes one 64-step adaptation from the released 2B, 128 training rows, 32
calibration rows and 32 each Test/OOD, before model evaluation. It also requires
the earlier frozen v4 groups as regression controls and both weight states at
both temperatures. A completed plan alone is not a measured model result.
