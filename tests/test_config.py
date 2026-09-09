from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from anigen.config import ConfigError, read_settings, use_workspace


class ConfigTests(unittest.TestCase):
    def test_unrelated_malformed_field_is_not_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings"
            path.write_text('UNRELATED="unterminated\nGPT_IMAGE_MODEL="gpt-image-test"\n')
            self.assertEqual(read_settings(["OPENAI_IMAGE_MODEL"], path, environ={}), {"OPENAI_IMAGE_MODEL": "gpt-image-test"})

    def test_alias_conflicts_never_echo_values(self):
        with self.assertRaises(ConfigError) as raised:
            read_settings(["GPT_API_KEY"], None, environ={"GPT_API_KEY": "synthetic-first", "OPENAI_API_KEY": "synthetic-second"})
        self.assertNotIn("synthetic-first", str(raised.exception))
        self.assertNotIn("synthetic-second", str(raised.exception))

    def test_environment_overrides_same_field_but_not_conflicting_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings"
            path.write_text('GPT_IMAGE_MODEL="old"\n')
            self.assertEqual(read_settings(["GPT_IMAGE_MODEL"], path, environ={"GPT_IMAGE_MODEL": "new"}), {"GPT_IMAGE_MODEL": "new"})
            with self.assertRaises(ConfigError):
                read_settings(["GPT_IMAGE_MODEL"], path, environ={"OPENAI_IMAGE_MODEL": "new"})

    def test_context_isolated_between_workers_and_no_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / name for name in ("a", "b")]
            for root in roots:
                root.mkdir()
                (root / ".env").write_text(f'REFERENCE_SCOPE_ID="{root.name}"\n')
            def read(root):
                with use_workspace(root):
                    return read_settings(["REFERENCE_SCOPE_ID"], environ={})
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(list(pool.map(read, roots)), [{"REFERENCE_SCOPE_ID": "a"}, {"REFERENCE_SCOPE_ID": "b"}])
            self.assertEqual(read_settings(["REFERENCE_SCOPE_ID"], None, environ={}), {})

    def test_private_aliases_are_field_names_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".private").mkdir()
            (root / ".private/config-aliases.json").write_text(json.dumps({"REFERENCE_SCOPE_ID": ["LOCAL_SCOPE"]}))
            (root / ".env").write_text('LOCAL_SCOPE="synthetic-scope"\n')
            self.assertEqual(read_settings(["REFERENCE_SCOPE_ID"], workspace=root, environ={}), {"REFERENCE_SCOPE_ID": "synthetic-scope"})
            self.assertEqual(read_settings(["REFERENCE_SCOPE_ID"], None, workspace=root, environ={}), {})

    def test_no_workspace_means_no_default_file_read(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("unexpected file read")):
            self.assertEqual(read_settings(["GPT_API_KEY"], environ={}), {})

    def test_repeated_conflicting_field_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings"
            path.write_text('GPT_IMAGE_MODEL="first"\nGPT_IMAGE_MODEL="second"\n')
            with self.assertRaisesRegex(ConfigError, "Conflicting repeated"):
                read_settings(["GPT_IMAGE_MODEL"], path, environ={})
