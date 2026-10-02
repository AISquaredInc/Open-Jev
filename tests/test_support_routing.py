from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from scripts.evaluate_support_routing import (UNKNOWN, finalize, fit_threshold, make_index,
                                               measure, metrics, request_for, shortlist, write_csv)


class SupportRoutingTests(unittest.TestCase):
    def setUp(self):
        self.index = make_index([{"text": "my card has not arrived", "category": "card_arrival"},
                                 {"text": "cash withdrawal from ATM", "category": "cash_withdrawal"},
                                 {"text": "I forgot the card PIN", "category": "pin_issue"}])

    def test_shortlist_never_reads_gold_and_retrieval_failure_stays_visible(self):
        self.assertEqual(shortlist("where is my card", self.index, 2)[0], "card_arrival")
        request = request_for("cash from ATM", self.index, 2)
        self.assertEqual(len(request["questions"]["route"]["criteria"]), 3)
        self.assertNotIn("target", json.dumps(request))
        results = [{"id": "1", "success": True, "prediction": "card_arrival", "top_probability": .95,
                    "candidate_labels": ["card_arrival", UNKNOWN], "retrieval_prediction": "card_arrival", "wall_ms": 4}]
        stats = metrics(results, {"1": "cash_withdrawal"}, .9, "cash_withdrawal")
        self.assertEqual(stats["candidate_recall"], 0)
        self.assertEqual(stats["routing_accuracy"], 0)
        self.assertEqual(stats["error_among_accepted"], 1)

    def test_threshold_fit_abstains_when_no_calibration_policy_is_met(self):
        results = [{"id": str(i), "success": True, "prediction": "right" if i < 3 else "wrong", "top_probability": .8 if i < 3 else .3} for i in range(6)]
        gold = {str(i): "right" for i in range(6)}
        self.assertEqual(fit_threshold(results, gold, .9, 3), .8)
        self.assertGreater(fit_threshold(results, gold, .9, 4), 1)
        results[0]["prediction"] = UNKNOWN
        self.assertGreater(fit_threshold(results, gold, .9, 3), 1)

    def test_real_http_schema_probabilities_and_identity_drift(self):
        responses = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                labels = list(request["questions"]["route"]["criteria"])
                probs = {label: .9 if i == 0 else .1 / (len(labels) - 1) for i, label in enumerate(labels)}
                responses.append(request)
                answer = {"model": "fixture-2B", "metadata": {"checkpoint_sha256": str(len(responses)),
                          "base_revision": "a" * 40, "temperature": 1.0, "method": "test_fixture",
                          "code_commit": "b" * 40, "max_length": 1024},
                          "answers": {"route": {"choice": labels[0], "probabilities": probs}}}
                payload = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    measure([{"id": "1", "text": "card"}, {"id": "2", "text": "cash"}], self.index,
                            f"http://127.0.0.1:{server.server_port}/v1/systemone", Path(tmp) / "results.jsonl", 2)
                records = [json.loads(x) for x in (Path(tmp) / "results.jsonl").read_text().splitlines()]
                self.assertTrue(records[0]["success"])
                self.assertFalse(records[1]["success"])
                self.assertGreater(records[1]["wall_ms"], 0)
                self.assertEqual(len(responses), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_review_queue_keeps_unresolved_rows_and_requires_human_attribution(self):
        rows = [{"id": "1", "text": "card", "status": "accepted", "proposed_label": "card_arrival", "top_probability": .95, "reviewed_label": "", "reviewer": ""},
                {"id": "2", "text": "cash", "status": "review", "proposed_label": UNKNOWN, "top_probability": .2, "reviewed_label": "", "reviewer": ""}]
        fields = list(rows[0])
        with tempfile.TemporaryDirectory() as tmp:
            queue, output = Path(tmp) / "queue.csv", Path(tmp) / "final.csv"
            write_csv(queue, rows, fields)
            self.assertEqual(finalize(queue, self.index, output)["unresolved"], 1)
            rows[1]["reviewed_label"] = "cash_withdrawal"
            write_csv(queue, rows, fields)
            with self.assertRaisesRegex(ValueError, "reviewer"):
                finalize(queue, self.index, output)
            rows[1]["reviewer"] = "fixture reviewer"
            write_csv(queue, rows, fields)
            result = finalize(queue, self.index, output)
            self.assertEqual(result["human_reviewed"], 1)
            self.assertEqual(result["unresolved"], 0)


if __name__ == "__main__":
    unittest.main()
