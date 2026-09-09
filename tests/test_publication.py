"""Synthetic publication-boundary tests; never read user configuration."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.public_release import ReleaseError, _read_candidate, candidates, export, findings, main, snapshot


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve() / "source"
        self.root.mkdir()
        self.addCleanup(self.temp.cleanup)

    def write(self, name, text="synthetic fixture\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def manifest(self, names):
        return self.write("public-files.txt", "\n".join(names) + "\n")

    def test_only_explicit_files_export_and_configuration_never_read(self):
        self.write("src/app.py", "value = 1\n")
        self.write(".env", "DO_NOT_READ=private\n")
        self.write("generation/figs/private/state.json")
        self.write("src/not_reviewed.py")
        self.manifest(["src/app.py"])
        original = Path.read_bytes

        def guarded(path):
            self.assertNotEqual(path.name, ".env")
            return original(path)

        output = self.root.parent / "export"
        with patch.object(Path, "read_bytes", guarded):
            self.assertEqual(export(self.root, output), 1)
        self.assertEqual([p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()], ["src/app.py"])

    def test_paths_cannot_escape_or_expand(self):
        for name in ["../outside.py", "/outside.py", "src/../outside.py", "src//file.py", "src/*.py", "src\\file.py"]:
            with self.subTest(name=name):
                self.manifest([name])
                with self.assertRaises(ReleaseError):
                    candidates(self.root)

    def test_protected_files_rejected_even_when_explicitly_listed(self):
        for name in [".env", ".env.example", "AGENTS.md", ".private/note.md", "generation/figs/a.py", "video/runs/a.py", "state.json", "a.png"]:
            with self.subTest(name=name):
                self.write(name)
                self.manifest([name])
                with self.assertRaises(ReleaseError):
                    candidates(self.root)

    def test_symlinks_and_directory_symlinks_rejected(self):
        self.write("safe.py")
        (self.root / "link.py").symlink_to(self.root / "safe.py")
        self.manifest(["link.py"])
        with self.assertRaises(ReleaseError):
            candidates(self.root)
        (self.root / "linkdir").symlink_to(self.root, target_is_directory=True)
        self.manifest(["linkdir/safe.py"])
        with self.assertRaises(ReleaseError):
            candidates(self.root)

    def test_binary_and_duplicate_entries_are_rejected(self):
        file = self.write("a.py")
        file.write_bytes(b"fake\x00binary")
        self.manifest(["a.py"])
        with self.assertRaises(ReleaseError):
            snapshot(self.root)
        file.write_text("fixture")
        self.manifest(["a.py", "a.py"])
        with self.assertRaises(ReleaseError):
            candidates(self.root)

    def test_safe_read_rejects_link_replaced_after_manifest_validation(self):
        source = self.write("source.py")
        self.write("private.py", "fixture secret")
        self.manifest(["source.py"])
        candidates(self.root)
        source.unlink()
        source.symlink_to(self.root / "private.py")
        with self.assertRaises(ReleaseError):
            _read_candidate(self.root, "source.py")

    def test_secrets_report_location_without_matching_value(self):
        synthetic_secret = "sk-" + "a" * 32
        path = self.write("sample.py", "\nvalue = '" + synthetic_secret + "'\n")
        result = findings([path])
        self.assertEqual(result, [(path, 2, ["credential"])])
        self.assertNotIn(synthetic_secret, str(result))
        self.manifest(["sample.py"])
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["--root", str(self.root)]), 1)
        self.assertIn("sample.py:2: credential", out.getvalue())
        self.assertNotIn(synthetic_secret, out.getvalue() + err.getvalue())

    def test_private_semantic_patterns_are_local_and_block_export(self):
        phrase = "INTERNAL_" + "FIXTURE_CAPABILITY"
        rules = self.write(".private/publication-rules.json", json.dumps({"patterns": [{"category": "private_semantics", "regex": phrase}]}))
        self.write("docs/guide.md", phrase)
        self.manifest(["docs/guide.md"])
        files, issues = snapshot(self.root, rules=rules)
        self.assertEqual(issues, [("docs/guide.md", 1, ["private_semantics"])])
        output = self.root.parent / "blocked"
        with self.assertRaises(ReleaseError):
            export(self.root, output, rules=rules)
        self.assertFalse(output.exists())
        self.assertNotIn(phrase, str(issues))
        self.assertEqual(len(files), 1)

    def test_sensitive_filename_is_redacted(self):
        filename = "sk-" + "z" * 32 + ".py"
        self.write(filename)
        self.manifest([filename])
        _, issues = snapshot(self.root)
        self.assertEqual(issues, [("candidate-1", 0, ["credential", "filename"])])
        self.assertNotIn(filename, str(issues))

    def test_existing_output_not_overwritten(self):
        self.write("safe.py")
        self.manifest(["safe.py"])
        output = self.root.parent / "existing"
        output.mkdir()
        (output / "user.txt").write_text("preserve")
        with self.assertRaises(ReleaseError):
            export(self.root, output)
        self.assertEqual((output / "user.txt").read_text(), "preserve")

    def test_export_contains_exact_scanned_bytes(self):
        self.write("safe.py", "fixture = '中文'\n")
        self.manifest(["safe.py"])
        output = self.root.parent / "snapshot"
        self.assertEqual(export(self.root, output), 1)
        self.assertEqual((output / "safe.py").read_bytes(), (self.root / "safe.py").read_bytes())


if __name__ == "__main__":
    unittest.main()
