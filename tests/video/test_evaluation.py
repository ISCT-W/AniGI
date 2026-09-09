"""Evaluation tests never call a model and do not invent model judgments."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from anigen.video.evaluation import EvaluationError, _cases, _digest, _file_digest, prepare, score


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gpt-agent-score-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / "manifest.json"
        self.judgments_path = self.root / "judgments.json"
        self.manifest = {"schema_version": 1, "cases": []}
        for split in ("tune", "holdout"):
            spec = {"duration_ms": 5000, "check": "motion", "requirement": "Keep moving"}
            for member in ("control", "variant"):
                identity = f"{split}_{member}"
                media = self.root / (identity + ".mp4")
                # This scorer only binds bytes, so its unit fixtures do not pretend to be decodable video.
                media.write_bytes(identity.encode())
                self.manifest["cases"].append({"case_id": identity, "pair_id": split, "member": member,
                    "split": split, "control_case_id": f"{split}_control", "media_path": media.name,
                    "media_sha256": _file_digest(media), "spec": spec, "spec_sha256": _digest(spec),
                    "ground_truth": {"expected_accept": member == "control", "severity": "none" if member == "control" else "severe",
                                     "defect_intervals_ms": [] if member == "control" else [[2000, 3000]]}})
        self.document = {"schema_version": 1, "judgments": []}

    def run_score(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.judgments_path.write_text(json.dumps(self.document))
        return score(self.manifest_path, self.judgments_path)

    def judge(self, identity, decision, arm="B", **extra):
        case = next(c for c in self.manifest["cases"] if c["case_id"] == identity)
        row = {"case_id": identity, "arm": arm, "media_sha256": case["media_sha256"],
               "spec_sha256": case["spec_sha256"], "decision": decision, **extra}
        self.document["judgments"].append(row)
        return row

    def test_empty_judgments_are_missing_not_passed_and_zero_denominator_is_null(self):
        result = self.run_score()["results"]
        for split in ("tune", "holdout"):
            for arm in "ABCD":
                metrics = result[split][arm]
                self.assertEqual(metrics["coverage"]["rate"], 0)
                self.assertIsNone(metrics["severe_false_accept"]["rate"])
                self.assertIsNone(metrics["false_reject"]["rate"])
                self.assertIsNone(metrics["unknown"]["rate"])
                self.assertEqual(metrics["unresolved_including_missing"]["rate"], 1)
                self.assertIsNone(metrics["usage"]["cost_usd"]["total"])
                self.assertEqual(metrics["usage"]["cost_usd"]["missing_judgments"], 2)

    def test_known_errors_unknown_missing_and_fixed_splits_remain_separate(self):
        self.judge("tune_control", "reject")
        self.judge("tune_variant", "accept")
        self.judge("holdout_control", "unknown")
        result = self.run_score()["results"]
        self.assertEqual(result["tune"]["B"]["severe_false_accept"]["rate"], 1)
        self.assertEqual(result["tune"]["B"]["false_reject"]["rate"], 1)
        self.assertEqual(result["holdout"]["B"]["unknown"]["rate"], 1)
        self.assertIsNone(result["holdout"]["B"]["severe_false_accept"]["rate"])
        self.assertEqual(result["holdout"]["B"]["missing_case_ids"], ["holdout_variant"])
        self.assertEqual(result["holdout"]["B"]["unresolved_including_missing"]["rate"], 1)

    def test_localization_and_partial_costs_are_not_confused_with_totals(self):
        self.judge("tune_control", "accept", usage={"calls": 1, "cost_usd": .02, "latency_ms": 300})
        self.judge("tune_variant", "reject", finding_intervals_ms=[[2100, 3200]],
                   usage={"calls": 2, "cost_usd": None, "latency_ms": 500})
        metrics = self.run_score()["results"]["tune"]["B"]
        self.assertEqual(metrics["localization"]["mean_boundary_absolute_error_ms"], 150)
        self.assertEqual(metrics["usage"]["calls"]["total"], 3)
        self.assertEqual(metrics["usage"]["latency_ms"]["total"], 800)
        self.assertEqual(metrics["usage"]["cost_usd"]["known_sum"], .02)
        self.assertIsNone(metrics["usage"]["cost_usd"]["total"])
        self.assertEqual(metrics["usage"]["cost_usd"]["unknown_judgments"], 1)
        self.assertFalse(metrics["usage"]["cost_usd"]["complete"])

    def test_needs_review_is_unknown_and_reject_without_location_is_reported(self):
        self.judge("tune_control", "needs_review")
        self.judge("tune_variant", "reject")
        metrics = self.run_score()["results"]["tune"]["B"]
        self.assertEqual(metrics["unknown"]["rate"], .5)
        self.assertEqual(metrics["false_reject"]["count"], 0)
        self.assertEqual(metrics["localization"]["unlocalized_rejected_cases"], 1)
        self.assertIsNone(metrics["localization"]["mean_boundary_absolute_error_ms"])

    def test_media_spec_and_duplicate_judgments_are_bound(self):
        row = self.judge("tune_control", "accept")
        row["media_sha256"] = "stale"
        with self.assertRaisesRegex(EvaluationError, "hash"):
            self.run_score()
        row["media_sha256"] = self.manifest["cases"][0]["media_sha256"]
        self.document["judgments"].append(copy.deepcopy(row))
        with self.assertRaisesRegex(EvaluationError, "Duplicate"):
            self.run_score()
        self.document["judgments"].pop()
        (self.root / "tune_control.mp4").write_bytes(b"replacement")
        with self.assertRaisesRegex(EvaluationError, "media"):
            self.run_score()

    def test_control_pairs_and_declared_split_cannot_cross_splits(self):
        row = self.judge("tune_control", "accept", split="holdout")
        with self.assertRaisesRegex(EvaluationError, "split"):
            self.run_score()
        row.pop("split")
        self.manifest["cases"][1]["split"] = "holdout"
        with self.assertRaisesRegex(EvaluationError, "splits"):
            self.run_score()

    def test_pair_spec_mutation_and_unknown_decisions_are_rejected(self):
        self.judge("tune_control", "PASS")
        with self.assertRaisesRegex(EvaluationError, "Decision"):
            self.run_score()
        self.document["judgments"] = []
        self.manifest["cases"][1]["spec"] = {"duration_ms": 1000, "check": "motion"}
        self.manifest["cases"][1]["spec_sha256"] = _digest(self.manifest["cases"][1]["spec"])
        with self.assertRaisesRegex(EvaluationError, "specification"):
            self.run_score()

    def test_invalid_time_and_usage_fail_closed(self):
        row = self.judge("tune_variant", "reject", finding_intervals_ms=[[3000, 5100]])
        with self.assertRaisesRegex(EvaluationError, "interval"):
            self.run_score()
        row["finding_intervals_ms"] = [[2000, 3000]]
        for usage in ({"calls": 1.5}, {"calls": True}, {"cost_usd": -1}, {"latency_ms": float("nan")}):
            row["usage"] = usage
            with self.assertRaises(EvaluationError):
                self.run_score()

    def test_malformed_nested_fields_are_validation_errors(self):
        self.document["judgments"] = [{"case_id": [], "arm": "B"}]
        with self.assertRaises(EvaluationError):
            self.run_score()
        self.document["judgments"] = []
        self.manifest["cases"][0]["ground_truth"] = None
        with self.assertRaises(EvaluationError):
            self.run_score()

    def test_ablation_claim_requires_same_declared_model_prompt_and_other_parameters(self):
        configs = {"B": {"model": "fixed-model", "prompt_sha256": "prompt", "fps": 1,
                         "audio": True, "local_review": False, "other_parameters": {"temperature": 0}}}
        configs["C"] = {**configs["B"], "local_review": True}
        configs["D"] = {**configs["B"], "fps": 4}
        self.document["arm_configs"] = configs
        result = self.run_score()["comparability"]
        self.assertEqual(result["B_vs_C"]["status"], "declared_comparable")
        self.assertEqual(result["B_vs_D"]["status"], "declared_comparable")
        self.assertFalse(result["B_vs_D"]["verified_execution"])
        configs["D"]["model"] = "another-model"
        self.assertEqual(self.run_score()["comparability"]["B_vs_D"]["status"], "confounded")
        configs["C"].pop("prompt_sha256")
        self.assertEqual(self.run_score()["comparability"]["B_vs_C"]["status"], "unknown")

    def test_recipe_has_18_single_factor_pairs_and_never_overwrites_directory(self):
        recipes = _cases()
        self.assertEqual(len(recipes), 18)
        self.assertEqual(len({p["pair_id"] for p in recipes}), 18)
        self.assertEqual(sum(p["variant_acceptable"] for p in recipes), 3)
        for pair in recipes:
            self.assertTrue(pair["changed_factor"])
            self.assertEqual(sum(pair[f"control_{m}"] != pair[f"variant_{m}"] for m in ("video", "audio")), 1)
        with self.assertRaisesRegex(EvaluationError, "new output"):
            prepare(self.root)


if __name__ == "__main__":
    unittest.main()
