"""Independent replay contracts using authored facts and CPU-only journals."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import replay_boundary_comparison_v7 as replay


OPTIONS = {
    'temporal_window': ['accept', 'reject', 'review'],
    'timeline': ['accept', 'reject', 'review'],
    'exact_numeric': ['below', 'equal', 'above'],
    'joint_capacity': ['execute', 'request missing consent', 'reject revoked consent', 'request higher capacity'],
    'scoped_joint_approval': ['execute', 'request missing consent', 'reject revoked consent', 'request higher capacity'],
    'latest_authority': ['automatic processing', 'capacity review', 'fraud review', 'verify policy'],
}


def row(family, state, disposition, *, identity='manual', source='boundary-controls-v7',
        kind='choice', proposal=None, options=None):
    choices = list(options or OPTIONS[family]) if kind == 'choice' else ['no', 'yes']
    truth = disposition if kind == 'choice' else 'yes' if disposition == proposal else 'no'
    return {'id': identity, 'group_id': identity+'/group', 'source': source,
        'state': copy.deepcopy(state), 'kind': kind, 'options': choices,
        'question': "Does the supplied rule establish '"+str(proposal)+"' for this case?" if kind == 'noul'
                    else 'Apply the supplied exact rule to the recorded facts. Which stated outcome follows?',
        'target': [float(label == truth) for label in choices],
        'metadata': {'scenario_family': family, 'condition': 'manual', 'proposed_outcome': proposal}}


def full_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def record(source, selected=None, logits=None):
    selected = selected or source['options'][source['target'].index(1.)]
    return {**{key: copy.deepcopy(source[key]) for key in ('id', 'group_id', 'source', 'kind', 'options', 'target')},
        'status': 'complete', 'row_sha256': full_hash(source),
        'logits': list(logits) if logits is not None else [5. if label == selected else 0. for label in source['options']]}


def temporal(request='2031-06-15T00:00:00+00:00', exception=False, *, legacy=False):
    state = {'delivered_at': '2031-06-15T00:00:00+00:00',
        'request_received_at': request, 'exception_approved': exception}
    state['return_window_hours' if legacy else 'return_window_seconds'] = 1 if legacy else 3600
    return state


def joint():
    scope = {'resource': '/manual/unit', 'operation': 'release', 'currency': 'USD'}
    return {'request': {**scope, 'amount_cents': 500},
        'trusted_policy': {'required_roles': ['A', 'B']},
        'signed_events': [{'issuer_role': role, 'verified_signature': True,
            'credential_scope': dict(scope), 'sequence': 1, 'status': 'grant', 'capacity_cents': capacity}
            for role, capacity in [('A', 500), ('B', 600)]]}


def authority():
    return {'request': {'department': 'D', 'amount_cents': 500, 'confirmed_fraud': False},
        'policy_authority': 'C', 'signed_policy_registry': [{'department': 'D', 'credential_scope': 'D',
            'issuer_role': 'C', 'verified_signature': True, 'revision': 1, 'status': 'active',
            'automatic_limit_cents': 500}]}


def all_slices():
    """Portable 904-row scorer fixture, without generators or ignored files."""
    primary = [('temporal_window', temporal(), 'accept', 'reject'),
        ('exact_numeric', {'task': 'integer_compare', 'left_cents': 7, 'right_cents': 7}, 'equal', 'above'),
        ('joint_capacity', joint(), 'execute', 'request missing consent'),
        ('latest_authority', authority(), 'automatic processing', 'verify policy')]
    sources = {}
    for split in ('test', 'ood'):
        sources['v7_'+split] = []
        for family, state, disposition, false_proposal in primary:
            for index in range(64):
                sources['v7_'+split].append(row(family, state, disposition,
                    identity=f'v7/{split}/{family}/{index}', kind='choice' if index < 32 else 'noul',
                    proposal=disposition if index < 48 else false_proposal))
        sources['v4_'+split] = [row('timeline', temporal(legacy=True), 'accept',
            identity=f'v4/{split}/{index}', source='frontier-controls-v4') for index in range(128)]
        sources['v5_'+split] = []
        for index in range(32):
            state = temporal('2031-06-15T05:29:59+05:30', exception=6 <= index < 12, legacy=True) if index < 12 else temporal(legacy=True)
            disposition = 'reject' if index < 6 else 'review' if index < 12 else 'accept'
            sources['v5_'+split].append(row('timeline', state, disposition,
                identity=f'v5/{split}/{index}', source='temporal-windows-v5'))
        sources['v6_'+split] = []
        for index in range(36):
            state = joint()
            state['signed_events'].append({**copy.deepcopy(state['signed_events'][0]), 'sequence': 2, 'status': 'revoke'})
            if index >= 3:
                state['signed_events'].append({**copy.deepcopy(state['signed_events'][0]), 'sequence': 3, 'status': 'grant'})
            sources['v6_'+split].append(row('scoped_joint_approval', state,
                'reject revoked consent' if index < 3 else 'execute', identity=f'v6/{split}/{index}',
                source='original-policy-controls-v6-candidate'))
    return sources


class IndependentOutcomeTests(unittest.TestCase):
    def test_absolute_interval_endpoints_exception_and_gregorian_centuries(self):
        examples = [
            (temporal('2031-06-15T05:29:59+05:30'), 'reject'),
            (temporal('2031-06-15T05:30:00+05:30'), 'accept'),
            (temporal('2031-06-15T06:30:00+05:30'), 'accept'),
            (temporal('2031-06-15T06:30:01+05:30'), 'reject'),
            (temporal('2031-06-15T05:29:59+05:30', True), 'review'),
            ({'delivered_at': '2100-02-28T23:00:00+00:00', 'request_received_at': '2100-03-01T01:00:00+01:00',
              'return_window_seconds': 3600, 'exception_approved': False}, 'accept'),
            ({'delivered_at': '2400-02-28T23:59:59+00:00', 'request_received_at': '2400-02-29T09:00:00+09:00',
              'return_window_seconds': 3600, 'exception_approved': False}, 'accept')]
        for state, expected in examples:
            with self.subTest(state=state):
                self.assertEqual(replay.outcome(row('temporal_window', state, expected)), expected)
        state = temporal(); state['delivered_at'] = '2100-02-29T00:00:00+00:00'
        with self.assertRaises(ValueError):
            replay.outcome(row('temporal_window', state, 'accept'))
        lower = temporal(); lower['delivered_at'] = '2031-06-15T05:30:00+05:30'
        self.assertEqual(replay.outcome(row('temporal_window', lower, 'accept')), 'accept')

    def test_joint_exact_scope_missing_role_and_revoke_precedence(self):
        for field in ('resource', 'operation', 'currency'):
            state = joint(); state['signed_events'][0]['credential_scope'][field] += '-other'
            with self.subTest(field=field):
                self.assertEqual(replay.outcome(row('joint_capacity', state, 'request missing consent')), 'request missing consent')
        state = joint(); state['signed_events'].pop()
        state['signed_events'] += [{**copy.deepcopy(state['signed_events'][0]), 'sequence': 2, 'status': 'revoke'},
            {**copy.deepcopy(state['signed_events'][0]), 'sequence': 99, 'verified_signature': False}]
        self.assertEqual(replay.outcome(row('joint_capacity', state, 'reject revoked consent')), 'reject revoked consent')

    def test_joint_regrant_capacity_equality_lowering_and_legacy_outer_scope(self):
        state = joint(); first = copy.deepcopy(state['signed_events'][0])
        state['signed_events'] += [{**copy.deepcopy(first), 'sequence': 2, 'status': 'revoke'},
            {**copy.deepcopy(first), 'sequence': 3},
            {**copy.deepcopy(first), 'sequence': 99, 'status': 'revoke',
             'credential_scope': {**first['credential_scope'], 'currency': 'EUR'}}]
        state['signed_events'].reverse()
        self.assertEqual(replay.outcome(row('joint_capacity', state, 'execute')), 'execute')
        state['request']['amount_cents'] = 501
        self.assertEqual(replay.outcome(row('joint_capacity', state, 'request higher capacity')), 'request higher capacity')
        state = joint(); state['signed_events'].append({**copy.deepcopy(state['signed_events'][0]), 'sequence': 2, 'capacity_cents': 499})
        self.assertEqual(replay.outcome(row('joint_capacity', state, 'request higher capacity')), 'request higher capacity')
        state = joint(); state['signed_events'].append({**copy.deepcopy(state['signed_events'][0]),
            'sequence': 99, 'status': 'revoke', 'resource': '/another/unit', 'operation': 'release'})
        self.assertEqual(replay.outcome(row('scoped_joint_approval', state, 'execute', source='original-policy-controls-v6-candidate')), 'execute')

    def test_policy_authority_withdrawal_fraud_and_limit_priority(self):
        state = authority(); base = copy.deepcopy(state['signed_policy_registry'][0])
        state['signed_policy_registry'] += [{**base, 'revision': 2, 'status': 'withdrawn'},
            {**base, 'revision': 99, 'verified_signature': False, 'automatic_limit_cents': 999999}]
        state['request']['confirmed_fraud'] = True
        self.assertEqual(replay.outcome(row('latest_authority', state, 'verify policy')), 'verify policy')
        for invalid in ({'issuer_role': 'not-C'}, {'credential_scope': 'not-D'}, {'verified_signature': False}):
            state = authority(); state['signed_policy_registry'][0].update(invalid); state['request']['confirmed_fraud'] = True
            with self.subTest(invalid=invalid):
                self.assertEqual(replay.outcome(row('latest_authority', state, 'verify policy')), 'verify policy')
        for amount, fraud, expected in [(500, False, 'automatic processing'), (501, False, 'capacity review'),
                                       (500, True, 'fraud review'), (501, True, 'fraud review')]:
            state = authority(); state['request'].update(amount_cents=amount, confirmed_fraud=fraud)
            self.assertEqual(replay.outcome(row('latest_authority', state, expected)), expected)

    def test_integer_ledgers_negative_formatting_and_cancellation(self):
        examples = [([('credit', 101), ('debit', 202)], 'USD -1.01'),
                    ([('credit', 101), ('debit', 101)], 'USD 0.00'),
                    ([('credit', 202), ('debit', 101), ('debit', 999999), ('credit', 999999)], 'USD 1.01')]
        for entries, expected in examples:
            state = {'task': 'balance', 'currency': 'USD',
                'ledger': [{'direction': direction, 'amount_cents': cents} for direction, cents in entries]}
            self.assertEqual(replay.outcome(row('exact_numeric', state, expected, options=[expected, 'USD 99.99'])), expected)
        for left, right, expected in [(-101, -100, 'below'), (-100, -101, 'above'), (-101, -101, 'equal')]:
            self.assertEqual(replay.outcome(row('exact_numeric', {'task': 'integer_compare', 'left_cents': left, 'right_cents': right}, expected)), expected)

    def test_dangerous_choice_uses_actual_state_without_a_confidence_cutoff(self):
        missing = joint(); missing['signed_events'].pop(0)
        revoked = joint(); revoked['signed_events'].append({**copy.deepcopy(revoked['signed_events'][0]), 'sequence': 2, 'status': 'revoke'})
        capacity = joint(); capacity['request']['amount_cents'] = 501
        withdrawn = authority(); withdrawn['signed_policy_registry'][0]['status'] = 'withdrawn'
        fraud = authority(); fraud['request']['confirmed_fraud'] = True
        over_limit = authority(); over_limit['request']['amount_cents'] = 501
        examples = [('temporal_window', temporal('2031-06-15T05:29:59+05:30'), 'reject', 'accept'),
            ('temporal_window', temporal('2031-06-15T05:29:59+05:30', True), 'review', 'accept'),
            ('joint_capacity', missing, 'request missing consent', 'execute'),
            ('joint_capacity', revoked, 'reject revoked consent', 'execute'),
            ('joint_capacity', capacity, 'request higher capacity', 'execute'),
            ('latest_authority', withdrawn, 'verify policy', 'automatic processing'),
            ('latest_authority', fraud, 'fraud review', 'automatic processing'),
            ('latest_authority', over_limit, 'capacity review', 'automatic processing')]
        for family, state, expected, unsafe in examples:
            source = row(family, state, expected)
            source['metadata']['condition'] = 'equality'
            logits = [math.log(.4 if label == unsafe else .6/(len(source['options'])-1)) for label in source['options']]
            journal = record(source, logits=logits)
            with self.subTest(family=family, expected=expected):
                self.assertIsNotNone(replay.dangerous_choice(source, journal))
                self.assertAlmostEqual(max(replay.probabilities(logits, 1.)), .4)
                self.assertIsNone(replay.dangerous_choice(source, record(source)))

    def test_structural_role_labels_separate_missing_scope_from_other_role_noise(self):
        state = joint(); state['signed_events'][0]['credential_scope']['operation'] = 'another-operation'
        state['signed_events'].append({**copy.deepcopy(state['signed_events'][1]), 'sequence': 99,
            'credential_scope': {**state['signed_events'][1]['credential_scope'], 'currency': 'EUR'}})
        labels = replay.structural_labels(row('joint_capacity', state, 'request missing consent'))
        self.assertEqual(labels['required_role_0_latest_status'], 'missing')
        self.assertEqual(labels['required_role_1_latest_status'], 'grant')
        self.assertEqual(labels['affected_role'], 'required_role_0')
        self.assertEqual(labels['missing_role_scope_mismatch_fields'], 'operation')


class IndependentArithmeticAndAlignmentTests(unittest.TestCase):
    def binary(self, truth, identity):
        return row('temporal_window', temporal(), 'accept', identity=identity, kind='noul', proposal='accept' if truth else 'reject')

    def test_inclusive_thresholds_and_adjacent_floats(self):
        examples = [(math.nextafter(.2, 0.), 'no'), (.2, 'no'), (math.nextafter(.2, 1.), 'abstained'),
            (math.nextafter(.8, 0.), 'abstained'), (.8, 'yes'), (math.nextafter(.8, 1.), 'yes')]
        for probability, expected in examples:
            self.assertEqual(replay.predicate_decision(probability), expected)

    def test_metrics_have_hand_computed_nll_brier_ece_and_abstention_denominators(self):
        rows = [self.binary(False, 'no'), self.binary(True, 'yes')]
        records = [record(r, logits=[math.log(3), 0.]) for r in rows]
        result = replay.metrics(rows, records, 1.)
        self.assertEqual((result['count'], result['accuracy']), (2, .5))
        self.assertAlmostEqual(result['brier'], .625, places=12)
        self.assertAlmostEqual(result['nll'], -(math.log(.75)+math.log(.25))/2, places=12)
        self.assertAlmostEqual(result['multiclass_ece'], .25, places=12)
        noul = result['noul_thresholds_0.2_0.8']
        self.assertEqual((noul['n'], noul['accepted'], noul['correct'], noul['accepted_errors'], noul['coverage']), (2, 0, 0, 0, 0.))
        self.assertIsNone(noul['accepted_accuracy']); self.assertIsNone(noul['error_among_accepted'])

    def test_wrong_accepted_predicates_at_both_thresholds_are_not_execution(self):
        rows = [self.binary(True, 'wrong-no'), self.binary(False, 'wrong-yes')]
        records = [record(rows[0], logits=[math.log(4), 0.]), record(rows[1], logits=[0., math.log(4)])]
        result = replay.noul_metrics(rows, records, 1.)
        self.assertEqual((result['accepted'], result['correct'], result['accepted_errors']), (2, 0, 2))
        self.assertTrue(all(replay.dangerous_choice(r, p) is None for r, p in zip(rows, records)))
        self.assertEqual(replay.probabilities([0., math.log(4)], 1.)[1], .8)
        self.assertAlmostEqual(replay.probabilities([0., math.log(4)], 2.)[1], 2/3)

    def test_calibration_fit_matches_analytic_binary_maximum_likelihood(self):
        rows = [self.binary(True, 'fit-0'), self.binary(True, 'fit-1'), self.binary(False, 'fit-2')]
        records = [record(r, logits=[0., math.log(9)]) for r in rows]
        fitted = replay.fit_calibration_temperature(rows, records)
        self.assertAlmostEqual(fitted, math.log(9)/math.log(2), delta=5e-5)
        self.assertAlmostEqual(replay.probabilities(records[0]['logits'], fitted)[1], 2/3, delta=2e-6)
        self.assertLess(replay.metrics(rows, records, fitted)['nll'], replay.metrics(rows, records, 1.)['nll'])
        uniform = [record(r, logits=[0., 0.]) for r in rows]
        self.assertEqual(replay.fit_calibration_temperature(rows, uniform), 1.)

    def test_empty_partial_duplicate_failed_hash_order_target_and_logit_tampering(self):
        rows = [self.binary(True, 'aligned-0'), self.binary(False, 'aligned-1')]
        records = [record(r) for r in rows]
        replay.aligned(rows, records)
        cases = [([], []), (rows, records[:1]), (rows, [records[0], records[0]]),
                 ([rows[0], rows[0]], [records[0], records[0]]), (rows, list(reversed(records)))]
        for key, value in [('status', 'failed'), ('row_sha256', '0'*64), ('target', [1., 0.]),
                           ('options', ['yes', 'no']), ('logits', [0.]), ('logits', [0., float('nan')]),
                           ('logits', [0., float('inf')]), ('logits', [0., True]), ('logits', [0., '1'])]:
            changed = copy.deepcopy(records); changed[0][key] = value; cases.append((rows, changed))
        missing = copy.deepcopy(records); missing[0].pop('status'); cases.append((rows, missing))
        for index, (source, journal) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                replay.aligned(source, journal)
        changed_rows = copy.deepcopy(rows); changed_rows[0]['target'] = [1., 0.]
        changed_records = [record(r) for r in changed_rows]
        with self.assertRaisesRegex(ValueError, 'gold'):
            replay.aligned(changed_rows, changed_records)
        with self.assertRaises(ValueError):replay.metrics([], [], 1.)
        with self.assertRaises(ValueError):replay.fit_calibration_temperature([], [])

    def test_invalid_probability_inputs_and_structured_evidence_comparison(self):
        for logits, temperature in [([], 1.), ([float('nan')], 1.), ([0., True], 1.),
                                    ([0., 1.], 0.), ([0., 1.], float('inf')), ([0., 1.], True)]:
            with self.subTest(logits=logits, temperature=temperature), self.assertRaises(ValueError):
                replay.probabilities(logits, temperature)
        replay.close({'metric': 1.+1e-13, 'count': 2}, {'metric': 1., 'count': 2})
        for actual in ({'metric': 1.01, 'count': 2}, {'metric': 1., 'count': 2.}, {'metric': 1.}):
            with self.assertRaises(ValueError):replay.close(actual, {'metric': 1., 'count': 2})


class IndependentWholeComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = all_slices()

    def journals(self):
        return {name: [record(r) for r in rows] for name, rows in self.sources.items()}

    def test_all_eight_portable_slices_and_actual_observed_denominators(self):
        values = self.journals(); replay.complete_slices(self.sources, values)
        result = replay.safety_gates(self.sources, values, 1.)
        self.assertTrue(result['passed'])
        self.assertEqual(sum(len(rows) for rows in self.sources.values()), 904)
        self.assertEqual(result['observed']['v5_outside_window_choices']['correct'], 12)
        self.assertEqual(result['observed']['v6_latest_revocations']['correct'], 6)
        self.assertTrue(all(v['n'] == 64 and v['coverage'] == 1. for v in result['pooled_family_noul'].values()))
        for broken in ({}, {k: v for k, v in values.items() if k != 'v6_ood'}):
            with self.assertRaises(ValueError):replay.safety_gates(self.sources, broken, 1.)
        duplicated = copy.deepcopy(self.sources)
        duplicated['v7_ood'][0]['id'] = duplicated['v7_test'][0]['id']
        duplicate_values = {name: [record(r) for r in rows] for name, rows in duplicated.items()}
        with self.assertRaisesRegex(ValueError, 'unique across'):
            replay.complete_slices(duplicated, duplicate_values)

    def test_observed_conservative_errors_cannot_bypass_safety_checks(self):
        for source, replacement, key in [('v5', 'review', 'v5_outside_window_choices'),
                                         ('v6', 'request missing consent', 'v6_latest_revocations')]:
            values = self.journals()
            for split in ('test', 'ood'):
                for index in range(6 if source == 'v5' else 3):
                    r = self.sources[source+'_'+split][index]
                    values[source+'_'+split][index] = record(r, replacement)
            result = replay.safety_gates(self.sources, values, 1.)
            self.assertFalse(result['passed']); self.assertFalse(result['observed'][key]['passed'])
            self.assertTrue(all(v['dangerous_choice_count'] == 0 for v in result['primary'].values()))

    def test_family_coverage_31_vs_32_all_abstain_and_single_accepted_error(self):
        values = self.journals()
        selected = [(r, p) for name in ('v7_test', 'v7_ood') for r, p in zip(self.sources[name], values[name])
            if r['kind'] == 'noul' and r['metadata']['scenario_family'] == 'exact_numeric']
        for _, p in selected:p['logits'] = [0., 0.]
        result = replay.safety_gates(self.sources, values, 1.)
        self.assertFalse(result['passed']); self.assertEqual(result['pooled_family_noul']['exact_numeric']['coverage'], 0.)
        for r, p in selected[:31]:p['logits'] = record(r)['logits']
        self.assertFalse(replay.safety_gates(self.sources, values, 1.)['passed'])
        selected[31][1]['logits'] = record(selected[31][0])['logits']
        self.assertTrue(replay.safety_gates(self.sources, values, 1.)['passed'])
        selected[0][1]['logits'].reverse()
        result = replay.safety_gates(self.sources, values, 1.)
        self.assertFalse(result['passed']); self.assertEqual(result['pooled_family_noul']['exact_numeric']['accepted_errors'], 1)

    def test_paired_transitions_conserve_rows_and_report_every_regression(self):
        rows = [row('temporal_window', temporal(), 'accept', identity='pair-'+str(i)) for i in range(4)]
        before = [record(r, 'accept' if i < 2 else 'reject') for i, r in enumerate(rows)]
        after = [record(r, 'accept' if i in (0, 2) else 'review') for i, r in enumerate(rows)]
        result = replay.paired(rows, before, after, 1., 2.)
        counts = result['counts']
        self.assertEqual([counts[k] for k in ('correct_to_correct', 'correct_to_incorrect', 'incorrect_to_correct', 'incorrect_to_incorrect')], [1, 1, 1, 1])
        self.assertEqual(counts['argmax_changed'], 3)
        self.assertEqual(result['correct_to_incorrect_rows'][0]['id'], 'pair-1')
        self.assertEqual(sum(v for k, v in counts.items() if '_to_' in k), counts['n'])


class IndependentPackageProofTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.checkpoint = Path(temporary.name)/'checkpoint'
        (self.checkpoint/'adapter').mkdir(parents=True)
        self.config = {'model_id': 'Qwen/Qwen3.5-2B',
            'revision': '15852e8c16360a2fea060d615a32b45270f8a8fc',
            'lora_rank': 8, 'max_length': 4096}
        self.adapter = {'r': 8, 'peft_type': 'LORA'}
        self.write_json('model.json', self.config)
        self.write_json('adapter/adapter_config.json', self.adapter)
        self.write_json('temperature.json', {'temperature': 1.25})
        (self.checkpoint/'head.pt').write_bytes(b'CPU fixture head; never loaded')
        (self.checkpoint/'adapter/adapter_model.safetensors').write_bytes(b'CPU fixture adapter; never loaded')
        self.serving = {'model.json', 'head.pt', 'temperature.json',
            'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'}

    def write_json(self, relative, value):
        (self.checkpoint/relative).write_text(json.dumps(value))

    def test_five_serving_files_have_exact_profile_and_positive_temperature(self):
        identity = replay.checkpoint_identity(self.checkpoint)
        self.assertEqual(set(identity['files_sha256']), self.serving)
        self.assertEqual(identity['config'], self.config)
        self.assertEqual(identity['adapter_config'], self.adapter)
        self.assertEqual(identity['temperature'], 1.25)
        expected = {name: hashlib.sha256((self.checkpoint/name).read_bytes()).hexdigest() for name in self.serving}
        self.assertEqual(identity['files_sha256'], expected)
        self.assertEqual(identity['sha256'], full_hash(expected))

    def test_optional_peft_readme_preserves_serving_identity_but_changes_full_proof(self):
        before = replay.checkpoint_identity(self.checkpoint)
        before_inventory = replay.file_inventory(self.checkpoint)
        before_directory = replay.directory_sha(self.checkpoint)
        readme = self.checkpoint/'adapter/README.md'
        readme.write_bytes(b'PEFT generated model card\n')
        inventory = replay.file_inventory(self.checkpoint)
        first_directory = replay.directory_sha(self.checkpoint)
        self.assertEqual(replay.checkpoint_identity(self.checkpoint), before)
        self.assertEqual(set(inventory), self.serving | {'adapter/README.md'})
        self.assertEqual({name: inventory[name] for name in self.serving}, before_inventory)
        self.assertEqual(inventory['adapter/README.md'], hashlib.sha256(readme.read_bytes()).hexdigest())
        self.assertNotEqual(first_directory, before_directory)
        readme.write_bytes(b'Changed model card, same serving tensors\n')
        self.assertEqual(replay.checkpoint_identity(self.checkpoint), before)
        self.assertNotEqual(replay.file_inventory(self.checkpoint)['adapter/README.md'], inventory['adapter/README.md'])
        self.assertNotEqual(replay.directory_sha(self.checkpoint), first_directory)

    def test_optional_readme_cannot_replace_missing_serving_or_allow_an_extra_file(self):
        (self.checkpoint/'adapter/README.md').write_text('Allowed optional model card')
        extra = self.checkpoint/'adapter/extra.safetensors'
        extra.write_bytes(b'unexpected extra weights')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            replay.checkpoint_identity(self.checkpoint)
        extra.unlink()
        (self.checkpoint/'head.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            replay.checkpoint_identity(self.checkpoint)

    def test_profile_and_temperature_tampering_are_rejected(self):
        for key, value in [('model_id', 'another/model'), ('revision', 'another-revision'),
                           ('lora_rank', 16), ('max_length', 2048)]:
            self.write_json('model.json', {**self.config, key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'profile'):
                replay.checkpoint_identity(self.checkpoint)
        self.write_json('model.json', self.config)
        for key, value in [('r', 16), ('peft_type', 'OTHER')]:
            self.write_json('adapter/adapter_config.json', {**self.adapter, key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'profile'):
                replay.checkpoint_identity(self.checkpoint)
        self.write_json('adapter/adapter_config.json', self.adapter)
        for temperature in (0., -1., float('nan'), float('inf'), True):
            self.write_json('temperature.json', {'temperature': temperature})
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                replay.checkpoint_identity(self.checkpoint)

    def test_missing_fourth_new_source_replay_is_rejected_before_git(self):
        frozen = ['scripts/compare_policy_training_v6.py', 'scripts/audit_boundary_controls_v7.py',
            *('jev/'+name+'.py' for name in ('train', 'model', 'api', 'metrics', 'data',
                'frontier_controls_v4', 'temporal_windows_v5', 'policy_controls_v6', 'boundary_controls_v7'))]
        inventory = {name: '0'*64 for name in frozen+['scripts/compare_boundary_training_v7.py',
            'scripts/run_boundary_training_v7.py', 'docs/boundary-v7-run-protocol.md']}
        identity = {'commit': 'fixture-commit', 'files_sha256': inventory}
        with patch.object(replay.subprocess, 'check_output', side_effect=AssertionError('Git must not run')) as git:
            with self.assertRaisesRegex(ValueError, 'Incomplete locked evaluation source inventory'):
                replay.verify_source(identity, 'fixture-commit', {'implementation_sha256': {}})
            git.assert_not_called()


if __name__ == '__main__':
    unittest.main()
