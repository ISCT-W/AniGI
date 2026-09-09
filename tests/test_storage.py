"""Synthetic retention boundary tests; never connect to providers."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from anigen.storage import compact, archive_approval
from anigen.workspace import TaskError, create_task
from anigen.image.task_store import TaskStore


class StorageTests(unittest.TestCase):
    def test_active_task_cannot_be_compacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = create_task(tmp, 'video', 'fixture', 'gpt', 'Synthetic')
            metadata = json.loads((task/'task.json').read_text())
            run_id = metadata['video_run_id']
            run = task/'video'/run_id
            run.mkdir(parents=True)
            (run/'state.json').write_text(json.dumps({'id': run_id, 'plan': {'task_id': metadata['id'], 'reference_scope_id': metadata['reference_scope_id']}, 'attempts': [{'status': 'submitted'}]}))
            (task/'keyframe').mkdir()
            with self.assertRaisesRegex(TaskError, 'unfinished'):
                compact(task)
            self.assertFalse((task/'.archive.json').exists())

    def test_shared_image_inputs_are_reused_without_changing_input_order(self):
        from tests.image.test_task_store import PNG
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root/'sample.png'
            image.write_bytes(PNG)
            store = TaskStore.create(root/'image', 'Synthetic', 'fixture')
            a = store.prepare('one', 'reference', 'gpt', 'fixture', inputs=[('reference', image)])
            b = store.prepare('two', 'reference', 'gpt', 'fixture', inputs=[('base', image), ('reference', image)])
            state = store.snapshot()
            first, second = state['attempts']
            self.assertEqual(first['inputs'][0]['path'], second['inputs'][0]['path'])
            self.assertEqual([r['role'] for r in second['inputs']], ['base', 'reference'])
            self.assertEqual(len(list((store.path/'references/assets').iterdir())), 1)
            self.assertFalse((store.path/'rounds'/a/'inputs').exists())
