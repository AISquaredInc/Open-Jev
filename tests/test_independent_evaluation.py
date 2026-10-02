"""Submission provenance checks; synthetic file fixtures are not model weights."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_independent_evaluation as submission
from scripts.evaluate_openjev_provider import validate_identity_config


class IndependentEvaluationTests(unittest.TestCase):
    def test_invalid_revision_cannot_write_or_inject_shell_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            for revision in ("main", "a" * 39, "$(touch injected)", "a" * 40 + "\n"):
                with self.assertRaises(ValueError):
                    submission.prepare(output, revision)
                self.assertFalse(output.exists())
            for length in (0, -1, True):
                with self.assertRaises(ValueError):
                    submission.prepare(output, "a" * 40, max_length=length)
                self.assertFalse(output.exists())

    def test_bundle_is_pinned_not_evaluated_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            manifest = submission.prepare(output, "a" * 40)
            validate_identity_config(manifest["identity"])
            self.assertIsNone(manifest["independent_evaluation"]["results"])
            self.assertEqual(manifest["status"], "prepared_not_evaluated")
            self.assertEqual(manifest["public_reproduction"]["denominators"]["total"], 231)
            self.assertEqual(manifest["request_settings"]["saved_training_max_length"], 4096)
            self.assertTrue(manifest["request_settings"]["evaluation_max_length_override"])
            hashes = json.loads((output / "files.json").read_text())
            for name, checksum in hashes.items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), checksum)
            for script in ("install-and-serve.sh", "reproduce-public.sh"):
                subprocess.run(["bash", "-n", str(output / script)], check=True)
            before = (output / "submission.json").read_bytes()
            with self.assertRaises(FileExistsError):
                submission.prepare(output, "b" * 40)
            self.assertEqual((output / "submission.json").read_bytes(), before)

    def test_verifier_rejects_wrong_bytes_without_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            submission.prepare(output, "a" * 40)
            checkpoint = output / "model/package/checkpoint"
            checkpoint.mkdir(parents=True)
            (checkpoint / "head.pt").write_bytes(b"corrupted opaque fixture")
            with self.assertRaisesRegex(SystemExit, "checksum differs"):
                runpy.run_path(str(output / "verify_and_record.py"))
            self.assertFalse((output / "runtime.json").exists())

    def test_verified_fixture_still_does_not_claim_inference_or_sealed_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            manifest = submission.prepare(output, "a" * 40, max_length=4096)
            checkpoint = output / "model/package/checkpoint"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.json").write_text(json.dumps({
                "model_id": submission.BASE_MODEL, "revision": submission.BASE_REVISION}))
            (checkpoint / "temperature.json").write_text(json.dumps({
                "temperature": submission.TEMPERATURE}))
            digest = hashlib.sha256()
            for file in sorted(checkpoint.rglob("*")):
                if file.is_file():
                    digest.update(file.relative_to(checkpoint).as_posix().encode() + b"\0")
                    digest.update(file.read_bytes())
            # This positive fixture uses its own checksum, never the release checksum.
            manifest["identity"]["checkpoint_sha256"] = digest.hexdigest()
            (output / "submission.json").write_text(json.dumps(manifest))
            def command(args, **kwargs):
                return "a" * 40 + "\n" if "rev-parse" in args else ""
            with patch("subprocess.check_output", side_effect=command):
                runpy.run_path(str(output / "verify_and_record.py"))
            report = json.loads((output / "runtime.json").read_text())
            self.assertTrue(report["checkpoint_verified"])
            self.assertFalse(report["inference_performed"])
            self.assertEqual(report["hardware_test_status"], "not_measured")
            self.assertEqual(report["sealed_result_status"], "pending_independent_evaluator")


if __name__ == "__main__":
    unittest.main()
