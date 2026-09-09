"""Project isolation stays enforced after removing deployment identifiers."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from anigen.image.config import reference_project
from anigen.image.config import gpt_key, gpt_image_model
from anigen.image.task_store import StoreError, TaskStore


class ProjectConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_offline_never_reads_real_project_configuration(self):
        with patch("anigen.image.config.read_settings", side_effect=AssertionError("no config read")):
            task = TaskStore.create(self.root / "offline", "Synthetic test", "Offline")
            self.assertEqual(task.snapshot()["project_id"], "offline-fixture")

    def test_generation_requires_local_project_before_creating_files(self):
        with patch("anigen.image.config.read_settings", return_value={}):
            with self.assertRaises(StoreError):
                TaskStore.create(self.root / "missing", "Synthetic test", "Missing",
                                 mode="generation", authorization="Fixture authorization; no API calls")
        self.assertFalse((self.root / "missing").exists())

    def test_configured_project_is_bound_and_different_project_is_rejected(self):
        with patch("anigen.image.config.read_settings", return_value={"REFERENCE_SCOPE_ID": "project-a"}):
            task = TaskStore.create(self.root / "valid", "Synthetic test", "Configured",
                                    mode="generation", authorization="Fixture authorization; no API calls")
            self.assertEqual(task.snapshot()["project_id"], "project-a")
            with self.assertRaises(StoreError):
                TaskStore.create(self.root / "other", "Synthetic test", "Other",
                                 mode="generation", authorization="Fixture", project_id="project-b")
        self.assertFalse((self.root / "other").exists())
        before = (task.path / "state.json").read_bytes()
        with patch("anigen.image.config.read_settings", return_value={"REFERENCE_SCOPE_ID": "project-b"}):
            with self.assertRaises(StoreError):
                task.recover()
        self.assertEqual((task.path / "state.json").read_bytes(), before)
        self.assertEqual(json.loads(before)["project_id"], "project-a")

    def test_project_reads_selected_setting_with_environment_precedence(self):
        config = self.root / ".env"
        config.write_text('REFERENCE_SCOPE_ID=project-a\nUNRELATED="unterminated\n')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(reference_project("generation", config), "project-a")
        with patch.dict(os.environ, {"REFERENCE_SCOPE_ID": "project-b"}, clear=True):
            self.assertEqual(reference_project("generation", config), "project-b")

    def test_invalid_project_does_not_appear_in_error(self):
        with patch("anigen.image.config.read_settings", return_value={"REFERENCE_SCOPE_ID": "https://private.invalid/?token=value"}):
            with self.assertRaises(StoreError) as caught:
                reference_project("generation")
        self.assertNotIn("private.invalid", str(caught.exception))

    def test_public_key_and_model_aliases_are_supported(self):
        config = self.root / ".env"
        config.write_text("OPENAI_API_KEY=synthetic-key\nOPENAI_IMAGE_MODEL=gpt-image-2\n")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gpt_key(config), "synthetic-key")
            self.assertEqual(gpt_image_model(env_file=config), "gpt-image-2")

    def test_conflicting_alias_values_are_rejected_without_disclosure(self):
        config = self.root / ".env"
        for fields, reader in (("GPT_API_KEY=synthetic-first\nOPENAI_API_KEY=synthetic-second\n", gpt_key),
                               ("GPT_IMAGE_MODEL=synthetic-first\nOPENAI_IMAGE_MODEL=synthetic-second\n", gpt_image_model)):
            config.write_text(fields)
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(StoreError) as caught:
                reader(env_file=config)
            self.assertNotIn("synthetic-first", str(caught.exception))
            self.assertNotIn("synthetic-second", str(caught.exception))

    def test_workspace_scoping_does_not_read_working_directory_dotenv(self):
        from anigen.config import use_workspace
        config = self.root / ".env"
        config.write_text("REFERENCE_SCOPE_ID=synthetic-scope\n")
        with patch.dict(os.environ, {}, clear=True), use_workspace(self.root):
            self.assertEqual(reference_project("generation"), "synthetic-scope")
            self.assertEqual(reference_project("offline"), "offline-fixture")

    def test_reference_scope_alias_is_accepted_without_changing_stored_format(self):
        task = TaskStore.create(self.root / "scope-alias", "Synthetic brief", "Synthetic",
                                reference_scope_id="offline-fixture")
        self.assertEqual(task.snapshot()["project_id"], "offline-fixture")
        with self.assertRaises(StoreError):
            TaskStore.create(self.root / "conflict", "Synthetic brief", "Synthetic",
                             project_id="offline-fixture", reference_scope_id="other-synthetic")


if __name__ == "__main__":
    unittest.main()
