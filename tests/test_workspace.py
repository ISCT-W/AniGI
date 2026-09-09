from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from anigen.config import use_workspace
from anigen.workspace import TaskError, artifact_path, create_task, description_slug, load_task, refresh_index
from anigen.image.task_store import TaskStore


class WorkspaceTests(unittest.TestCase):
    def test_final_purpose_names_and_no_video_image_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            for purpose, backend in (("image", "gpt"), ("image", "gemini"), ("video", "gpt"), ("video", "gemini")):
                task = create_task(tmp, purpose, "双人 * 场景 / 试验", backend, "Synthetic brief")
                path, root, record = load_task(task)
                self.assertEqual(path.name.count("*"), 2)
                self.assertEqual(path.parent.name, "figs" if purpose == "image" else "videos")
                self.assertEqual(path.name.split("*")[-1], backend + ("-minimax" if purpose == "video" else ""))
                self.assertEqual(record["description"], "双人-场景-试验")
                self.assertEqual((path / "image/state.json").exists(), purpose == "image")
                self.assertFalse((path / "final_output").exists())

    def test_same_clock_and_description_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
            def create(_):
                return create_task(tmp, "video", "same", "gpt", "Synthetic", now=stamp)
            with ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(create, range(4)))
            self.assertEqual(len(set(paths)), 4)
            self.assertTrue(all(path.name.count("*") == 2 for path in paths))
            self.assertEqual(len({load_task(path)[2]["video_run_id"] for path in paths}), 4)

    def test_description_safety_and_fallback(self):
        self.assertEqual(description_slug("../\\*?\n"), "task")
        self.assertEqual(len(description_slug("字" * 90)), 48)
        self.assertNotIn("\x00", description_slug("hello\x00world"))

    def test_resume_uses_existing_budget_and_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = create_task(tmp, "image", "resume", "gpt", "Synthetic brief")
            with use_workspace(tmp):
                store = TaskStore(path / "image")
                first = store.prepare("Synthetic prompt", "Synthetic reference", "gpt", "fixture")
                store.reserve(first, True)
                store.finish(first, "failed", note="Synthetic sent failure")
                refresh_index(path)
                self.assertEqual(store.recover()["remaining"], 5)
                self.assertEqual(load_task(path)[0], path)
                self.assertEqual(len(list((Path(tmp) / "generation/figs").iterdir())), 1)

    def test_renamed_task_and_changed_brief_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = create_task(tmp, "video", "check", "gpt", "Synthetic")
            (task / "brief.md").write_text("changed")
            with self.assertRaises(TaskError):
                load_task(task)
            other = create_task(tmp, "video", "check", "gemini", "Synthetic")
            moved = other.with_name(other.name.replace("*check*", "*different*"))
            other.rename(moved)
            with self.assertRaises(TaskError):
                load_task(moved)

    def test_symlink_generation_or_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            (root / "generation").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(TaskError):
                create_task(root, "video", "test", "gpt", "Synthetic")
            (root / "generation").unlink()
            task = create_task(root, "video", "test", "gpt", "Synthetic")
            (task / "escape").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(TaskError):
                artifact_path(task, "escape/file")
            with self.assertRaises(TaskError):
                artifact_path(task, "../other/file")

    def test_index_and_lock_symlinks_cannot_modify_external_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = create_task(tmp, "video", "test", "gpt", "Synthetic")
            outside = Path(tmp) / "external-marker"
            outside.write_text("untouched")
            (task / "README.md").unlink()
            (task / "README.md").symlink_to(outside)
            with self.assertRaises(TaskError):
                refresh_index(task)
            self.assertEqual(outside.read_text(), "untouched")

    def test_replaced_image_stage_cannot_reset_budget_or_change_brief(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = create_task(tmp, "image", "first", "gpt", "First brief")
            (task / "image").rename(task / "old-image")
            TaskStore.create(task / "image", "Second brief", "other", initial_backend="gpt")
            with self.assertRaisesRegex(TaskError, "Image stage"):
                load_task(task)
