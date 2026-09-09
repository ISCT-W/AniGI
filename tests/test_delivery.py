from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from anigen import delivery
from anigen.config import use_workspace
from anigen.image.task_store import StoreError, TaskStore
from anigen.workspace import TaskError, create_task
from anigen.workspace import refresh_index
from tests.image.test_task_store import PNG


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = create_task(self.root, "image", "synthetic", "gpt", "Synthetic brief")
        self.store = TaskStore(self.task / "image")
        self.reference = self.root / "fixture.png"
        self.reference.write_bytes(PNG)

    def candidate(self, verdict="pass"):
        with use_workspace(self.root):
            round_id = self.store.prepare("Synthetic", "Synthetic", "gpt", "fixture")
            self.store.reserve(round_id, True)
            self.store.finish(round_id, "succeeded", [self.reference], "Synthetic fixture result")
            self.store.review(round_id, "output-01.png", "Synthetic visual review", verdict,
                              blockers=[] if verdict == "pass" else ["Synthetic defect"])
            return round_id

    def test_failed_candidate_not_exported(self):
        round_id = self.candidate("fail")
        with self.assertRaises(StoreError):
            delivery.export(self.task, round_id, "output-01.png")
        self.assertFalse((self.task / "final_output").exists())

    def test_exact_approved_image_export_is_repeatable_and_user_verdict_separate(self):
        round_id = self.candidate()
        output = delivery.export(self.task, round_id, "output-01.png")
        self.assertEqual(output.parent, self.task / "final_output")
        self.assertEqual(output.read_bytes(), PNG)
        self.assertEqual(delivery.export(self.task, round_id, "output-01.png"), output)
        self.assertIn("pending", (self.task / "delivery.json").read_text())
        delivery.accept(self.task, output.name, "accepted", "User approved this exact fixture")
        self.assertIn("User: accepted", (self.task / "final_output/acceptance.md").read_text())

    def test_revoked_or_changed_candidate_cannot_be_user_accepted(self):
        round_id = self.candidate()
        output = delivery.export(self.task, round_id, "output-01.png")
        self.store.review(round_id, "output-01.png", "Synthetic later issue", "fail", ["Synthetic defect"])
        with self.assertRaises(StoreError):
            delivery.accept(self.task, output.name, "accepted", "New user response")
        delivery.accept(self.task, output.name, "changes_requested", "Please repair the fixture")

    def test_video_task_cannot_export_a_keyframe_as_its_final(self):
        task = create_task(self.root, "video", "video fixture", "gpt", "Synthetic brief")
        with self.assertRaises(TaskError):
            delivery.export(task, "001", "output-01.png")
        self.assertFalse((task / "final_output").exists())

    def test_tampered_delivery_cannot_be_accepted(self):
        output = delivery.export(self.task, self.candidate(), "output-01.png")
        output.write_bytes(b"changed synthetic content")
        with self.assertRaises(TaskError):
            delivery.accept(self.task, output.name, "accepted", "User response")

    def test_review_revoked_during_copy_leaves_no_final_artifact(self):
        round_id = self.candidate()
        original = delivery.shutil.copyfileobj
        def revoke(incoming, outgoing):
            original(incoming, outgoing)
            self.store.review(round_id, "output-01.png", "Synthetic revoked review", "fail", ["Defect"])
        with patch.object(delivery.shutil, "copyfileobj", side_effect=revoke):
            with self.assertRaises((StoreError, TaskError)):
                delivery.export(self.task, round_id, "output-01.png")
        self.assertFalse(any((self.task / "final_output").glob("*.png")))

    def test_revocation_removes_current_delivery_but_keeps_history(self):
        round_id = self.candidate()
        output = delivery.export(self.task, round_id, "output-01.png")
        delivery.accept(self.task, output.name, "accepted", "Synthetic acceptance")
        self.store.review(round_id, "output-01.png", "Later defect", "fail", ["Defect"])
        refresh_index(self.task)
        import json
        manifest = json.loads((self.task / "delivery.json").read_text())
        self.assertIsNone(manifest.get("current_version"))
        self.assertFalse(manifest["versions"][0]["approval_valid"])
        self.assertIn("historical", (self.task / "README.md").read_text())
        self.assertTrue(output.is_file())

    def test_production_status_retains_validity_with_workspace_local_scope(self):
        # A synthetic scope is sufficient; there are no credentials or model calls.
        (self.root / ".env").write_text('REFERENCE_SCOPE_ID="synthetic-scope"\n')
        task = create_task(self.root, "image", "production fixture", "gpt", "Synthetic brief",
                           mode="generation", authorization="Synthetic fixture authorization; no service calls")
        with use_workspace(self.root):
            store = TaskStore(task / "image")
            round_id = store.prepare("Synthetic", "Synthetic", "gpt", "fixture")
            store.reserve(round_id, True)
            store.finish(round_id, "succeeded", [self.reference], "Synthetic fixture, no service call")
            store.review(round_id, "output-01.png", "Synthetic review", "pass")
        output = delivery.export(task, round_id, "output-01.png")
        refresh_index(task)
        import json
        manifest = json.loads((task / "delivery.json").read_text())
        self.assertTrue(manifest["versions"][0]["approval_valid"])
        self.assertEqual(manifest["current_version"], output.name)
