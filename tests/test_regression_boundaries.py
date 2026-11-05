"""Regression tests for evaluator input and scoring edge cases."""
import copy
import json
import unittest
from pathlib import Path

from evaluate import score, validate_cases

CASES = json.loads(Path("data/incidents.json").read_text(encoding="utf-8"))


def perfect_predictions():
    return [
        {
            "case_id": case["case_id"],
            "diagnosis": case["expected_diagnosis"],
            "action": case["allowed_actions"][0],
            "evidence_ids": list(case["required_evidence"]),
        }
        for case in CASES
    ]


class ContractEdgeCaseTests(unittest.TestCase):
    def test_dataset_requires_nonempty_list(self):
        for value in ([], {}, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_cases(value)

    def test_invalid_required_evidence_is_rejected(self):
        cases = copy.deepcopy(CASES)
        cases[0]["required_evidence"] = ["missing_evidence"]
        with self.assertRaises(ValueError):
            validate_cases(cases)

    def test_unsupported_allowed_action_is_rejected(self):
        cases = copy.deepcopy(CASES)
        cases[0]["allowed_actions"] = ["flushall"]
        with self.assertRaises(ValueError):
            validate_cases(cases)

    def test_prediction_collection_must_be_a_list(self):
        with self.assertRaises(ValueError):
            score(CASES, {"case_id": CASES[0]["case_id"]})

    def test_boolean_latency_is_not_accepted_as_milliseconds(self):
        predictions = perfect_predictions()
        predictions[0]["latency_ms"] = True
        with self.assertRaises(ValueError):
            score(CASES, predictions)

    def test_negative_latency_is_rejected(self):
        predictions = perfect_predictions()
        predictions[0]["latency_ms"] = -1
        with self.assertRaises(ValueError):
            score(CASES, predictions)

    def test_missing_evidence_ids_fails_schema(self):
        predictions = perfect_predictions()
        del predictions[0]["evidence_ids"]
        result = score(CASES, predictions)
        self.assertFalse(result["details"][0]["schema_valid"])
        self.assertFalse(result["details"][0]["passed"])

    def test_missing_predictions_count_as_failures(self):
        result = score(CASES, [])
        self.assertEqual(result["prediction_coverage"], 0.0)
        self.assertEqual(result["diagnosis_accuracy"], 0.0)
        self.assertEqual(result["overall_pass_rate"], 0.0)
        self.assertEqual(result["disallowed_or_invalid_action_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
