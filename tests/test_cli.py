from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from anigen.cli import image_action, main, video_action
from anigen.workspace import TaskError, create_task, load_task
from tests.video.test_directing import plan


class CLITests(unittest.TestCase):
    def test_image_commands_use_named_task_and_explicit_first_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brief = root / "brief.md"
            prompt = root / "prompt.md"
            reference = root / "reference.md"
            for path in (brief, prompt, reference):
                path.write_text("Synthetic offline text")
            # An offline explicit-backend task must not parse this private file.
            (root / ".env").write_text('GPT_API_KEY="unterminated synthetic value\n')
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["--workspace", tmp, "new", "image", "--description", "离线测试", "--backend", "gpt", "--brief-file", str(brief)]), 0)
            task = Path(output.getvalue().strip())
            args = ["--prompt-file", str(prompt), "--reference-file", str(reference), "--model", "fixture"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(image_action(task, "prepare", [*args, "--backend", "gemini"]), 2)
                self.assertEqual(image_action(task, "prepare", [*args, "--backend", "gpt"]), 0)
                self.assertEqual(main(["status", "--task", str(task)]), 0)
            state = json.loads((task / "image/state.json").read_text())
            self.assertEqual(len(state["attempts"]), 1)
            self.assertEqual(state["selected_backend"], "gpt")

    def test_video_init_and_scope_binding_under_literal_star_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = create_task(tmp, "video", "video fixture", "gemini", "Synthetic brief")
            record = load_task(task)[2]
            data = plan()
            data["reference_scope_id"] = record["reference_scope_id"]
            data["source_scope"] = "synthetic"
            state = video_action(task, "init", data)
            self.assertEqual(state["id"], record["video_run_id"])
            self.assertEqual(state["plan"]["task_id"], record["id"])
            self.assertTrue((task / "video" / record["video_run_id"] / "state.json").is_file())
            self.assertEqual(load_task(task)[0], task)
            self.assertFalse((Path(tmp) / "generation/figs").exists())
            with self.assertRaises(TaskError):
                video_action(task, "submit", {}, live=True)

    def test_standalone_image_actions_cannot_write_video_keyframe(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = create_task(tmp, "video", "fixture", "gpt", "Synthetic")
            with self.assertRaises(TaskError):
                image_action(task, "recover", [])
            self.assertFalse((task / "image").exists())
