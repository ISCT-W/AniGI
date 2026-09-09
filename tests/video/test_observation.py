"""Contract tests: observations cannot promote themselves to production verdicts."""
import copy
import unittest

from anigen.video import observation as av


def response():
    return {"events": [{"start_s": 0, "end_s": 5, "description": "Square moves right", "evidence": "visible position changes"}],
            "dialogue_segments": [], "audio_events": [], "findings": [], "uncertainties": []}


class ObservationTests(unittest.TestCase):
    def test_director_resolution_contiguous_evidence_union(self):
        artifact = {"request": {"source_offset_s": 2}, "response": {"events": [
            {"start_s": 0, "end_s": 2}, {"start_s": 2, "end_s": 5}]}}
        chosen = {"observation_1": artifact}
        resolution = {"reason": "Continuous supporting observations", "evidence_refs": [
            "observation_1/events/1", "observation_1/events/0"]}
        finding = {"start_s": 0, "end_s": 5}
        av.director_resolution(resolution, chosen, "action", artifact, finding)
        artifact["response"]["events"][1]["start_s"] = 2.01
        with self.assertRaisesRegex(ValueError, "does not cover"):
            av.director_resolution(resolution, chosen, "action", artifact, finding)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            av.director_resolution(resolution, chosen, "action", artifact,
                                   {"start_s": 6, "end_s": 6})

    def test_actual_audio_required_and_times_bounded(self):
        data = response()
        self.assertEqual(av.validate_response(data, duration_s=5, has_audio=False), data)
        data["audio_events"] = copy.deepcopy(data["events"])
        with self.assertRaisesRegex(ValueError, "without audio"):
            av.validate_response(data, duration_s=5, has_audio=False)
        data["audio_events"] = []
        for value in (5.01, float("nan"), True):
            data["events"][0]["end_s"] = value
            with self.assertRaises(ValueError):
                av.validate_response(data, duration_s=5, has_audio=True)

    def test_response_cannot_self_authorize_or_invent_checks(self):
        data = response()
        data["accept"] = True
        with self.assertRaises(ValueError):
            av.validate_response(data, duration_s=5, has_audio=True)
        del data["accept"]
        data["findings"] = [{**data["events"][0], "check": "动作", "status": "observed"}]
        with self.assertRaisesRegex(ValueError, "unknown finding"):
            av.validate_response(data, duration_s=5, has_audio=True)

    def test_mapping_preserves_model_estimate_and_original_time(self):
        data = response()
        mapped = av.mapped_response(data, 2.25)
        self.assertEqual(mapped["events"][0]["source_end_s"], 7.25)
        self.assertEqual(mapped["events"][0]["time_precision"], "model_estimate")
        self.assertNotIn("source_end_s", data["events"][0])

    def test_policy_requires_positive_finite_cumulative_limits(self):
        policy = {"user_instruction": "Synthetic policy fixture, no real authorization", "allow_task_spec": True,
                  "allow_generated_media": False, "imported_target_ids": [], "reference_transfer_ids": [],
                  "max_calls": 4, "max_media_seconds": 20}
        self.assertEqual(av.validate_policy(copy.deepcopy(policy))["reserve_final_calls"], 1)
        for key, value in (("max_calls", True), ("max_media_seconds", float("inf")), ("max_output_tokens", 4097)):
            with self.assertRaises(ValueError):
                av.validate_policy({**policy, key: value})


if __name__ == "__main__":
    unittest.main()
