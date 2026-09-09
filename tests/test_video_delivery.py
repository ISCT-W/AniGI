"""Managed video creation, review, delivery and withdrawal using synthetic media."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from anigen import delivery
from anigen.cli import video_action
from anigen.video import directing, keyframes, observation, runtime
from anigen.workspace import create_task, refresh_index
from tests.video.test_directing import plan
from tests.video.test_keyframes import FakeImage, FakeUpload
from tests.video.test_observation_runtime import FakeObserver
from tests.video.test_runtime import FakeProvider


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class ManagedVideoDeliveryTests(unittest.TestCase):
    def test_reviewed_full_video_exports_then_withdraws_without_removing_history(self):
        self._exercise_delivery()

    def test_compaction_preserves_approval_and_detects_tampering(self):
        self._exercise_delivery(compact=True)

    def _exercise_delivery(self, compact=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'original.png'
            movie = root / 'synthetic.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=red:s=1536x864',
                            '-frames:v','1',str(original)], check=True, capture_output=True)
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=red:s=160x90:r=10:d=5',
                            '-c:v','libx264','-pix_fmt','yuv420p',str(movie)], check=True, capture_output=True)
            task = create_task(root, 'video', 'Synthetic end to end', backend='gpt', brief='Synthetic test only')
            image_backend, upload = FakeImage(original), FakeUpload()
            provider, observer = FakeProvider(movie), FakeObserver()
            provider.production = observer.production = False
            def call(action, data=None, *, live=False):
                return video_action(task, action, data, live=live, provider=provider, observer=observer,
                                    image_provider=image_backend, uploader=upload)
            script = plan(); script['reference_scope_id'] = 'offline-fixture'
            call('init', script)
            call('evidence', {'id':'ref', 'reference_scope_id':'offline-fixture', 'kind':'user_reference',
                             'verification':'verified', 'verification_basis':'direct_observation',
                             'verification_method':'synthetic_fixture', 'observation':'Synthetic red pixels',
                             'locator':'fixture:original', 'claim':'Synthetic fixture identity',
                             'observation_path':str(original), 'observation_sha256':runtime.digest(original),
                             'observation_mime_type':'image/png'})
            call('keyframe-authorize', {'user_instruction':'Synthetic test only', 'transfer_basis':'Fake providers only',
                                       'max_generations':6, 'max_uploads':5, 'allow_generated_edits':True,
                                       'allow_fal_upload':True, 'reference_transfer_ids':['ref']})
            call('keyframe-prepare', {'prompt':'Synthetic opening frame', 'model':'gpt-image-2',
                                     'design':{'composition':'Synthetic scene', 'starting_action':'Fixed toy',
                                               'motion_feasibility':'Continue synthetic camera',
                                               'reference_roles':{'ref':'Original fixture authority'}}})
            state = call('keyframe-generate', {'keyframe_id':'keyframe_0001'}, live=True)
            item = state['keyframes']['keyframe_0001']
            frame_report = {'review_authority':'codex_director', 'reviewer':'codex', 'image_sha256':item['image']['sha256'],
                            'request_hash':item['request_hash'], 'reference_ids':['ref'],
                            'observed_images':[item['image']['path'], item['reference_snapshot']['ref']['path']],
                            'summary':'Synthetic gate assertion; no real quality claim', 'limitations':['Synthetic fixtures'],
                            'decision':'accept', 'checks':{name:{'status':'pass','reason':'Synthetic fixture'} for name in keyframes.CHECKS},
                            'invariant_checks':{'toy_pose':{'status':'pass','reason':'Synthetic fixture'}}}
            call('keyframe-review', {'keyframe_id':item['id'], 'report':frame_report})
            call('authorize', {'user_instruction':'Synthetic test only', 'max_generations':1,
                               'max_generated_seconds':5, 'max_attempts_per_shot':1})
            state = call('keyframe-upload', {'keyframe_id':item['id']}, live=True)
            item = state['keyframes'][item['id']]
            url = state['evidence'][item['evidence_id']]['generation_url']
            call('prepare', {'shot_id':'a', 'keyframe_id':item['id'], 'endpoint':'minimax/h3-max/image-to-video',
                             'arguments':{'prompt':'Synthetic film', 'image_url':url, 'duration':5,
                                          'resolution':'768P','prompt_expansion_mode':'balanced'},
                             'reference_bindings':{url:item['evidence_id']},
                             'director_timeline':script['shots'][0]['director_timeline']})
            call('submit', {'attempt_id':'attempt_0001'}, live=True)
            call('poll', {'attempt_id':'attempt_0001'})
            state = call('collect', {'attempt_id':'attempt_0001'})
            call('observation-policy', {'user_instruction':'Synthetic observation only', 'allow_task_spec':True,
                                       'allow_generated_media':True, 'imported_target_ids':[],
                                       'reference_transfer_ids':list(state['evidence']), 'max_calls':3,
                                       'max_calls_per_target':2, 'max_media_seconds':30, 'reserve_final_calls':1,
                                       'review_authority':'codex_director'})
            def review_report(state, target_id, oid):
                target, _, refs = runtime.review_context(state, target_id)
                obs = target['observation']
                checks = {}
                for name in directing.VISUAL:
                    check = {'status':'pass', 'reason':'Synthetic fixture assertion'}
                    if name in observation.DYNAMIC_CHECKS:
                        check.update(director_reason='Synthetic observation contract', evidence_refs=[oid+'/events/0'])
                    checks[name] = check
                return {'media_sha256':obs['media_sha256'], 'observed_frames':[f['path'] for f in obs['frames']],
                        'observation_mode':'video', 'summary':'Synthetic full-video gate only', 'observation_ids':[oid],
                        'checks':checks, 'decision':'accept', 'continuity_state':'Synthetic still state',
                        'review_authority':'codex_director', 'director_review':{'reviewer':'codex',
                            'reference_ids':list(refs), 'rationale':'Offline synthetic contract only', 'limitations':['No real quality acceptance']},
                        'invariant_checks':{'toy_pose':{'status':'pass','reason':'Synthetic fixture'}}}
            state = call('observe-av', {'target_id':'attempt_0001',
                                       'reference_ids':list(runtime.review_context(state, 'attempt_0001')[2])}, live=True)
            call('review', {'attempt_id':'attempt_0001', 'report':review_report(state, 'attempt_0001', 'observation_0001')})
            state = call('assemble')
            self.assertFalse((task/'final_output').exists())
            state = call('observe-av', {'target_id':'final',
                                       'reference_ids':list(runtime.review_context(state, 'final')[2])}, live=True)
            state = call('final-review', review_report(state, 'final', 'observation_0002'))
            output = delivery.export(task)
            self.assertEqual(output.parent, task/'final_output')
            self.assertEqual(output.suffix, '.mp4')
            self.assertFalse(list((task/'final_output').glob('*.png')))
            manifest = json.loads((task/'delivery.json').read_text())
            self.assertEqual(manifest['current_version'], output.name)
            self.assertEqual(manifest['versions'][0]['acceptance'], 'pending')
            self.assertTrue(manifest['versions'][0]['approval_valid'])
            self.assertEqual(len(provider.submissions), 1)
            self.assertEqual(image_backend.calls, 1)
            self.assertEqual(observer.calls, 2)
            if compact:
                from anigen.storage import compact as compact_task, archive_approval
                from anigen.workspace import TaskError
                before = runtime.validate_final(state)
                original_unlink = Path.unlink
                interrupted = [False]
                def stop_once(file, *args, **kwargs):
                    if 'frames' in file.parts and not interrupted[0]:
                        interrupted[0] = True
                        raise OSError('synthetic cleanup interruption')
                    return original_unlink(file, *args, **kwargs)
                with patch.object(Path, 'unlink', stop_once):
                    with self.assertRaisesRegex(OSError, 'synthetic cleanup'):
                        compact_task(task)
                self.assertTrue((task/'.archive.json').is_file())
                result = compact_task(task)
                self.assertGreater(result['removed_files'], 0)
                self.assertEqual(archive_approval(task), before)
                self.assertEqual(compact_task(task)['removed_files'], 0)
                self.assertEqual(delivery.export(task), output)
                self.assertEqual(len(provider.submissions), 1)
                with self.assertRaisesRegex(TaskError, 'compacted'):
                    call('invalidate', {'shot_id':'a', 'reason':'blocked after compaction'})
                seal_path = task/'.archive.json'
                original_seal = seal_path.read_text()
                seal = json.loads(original_seal)
                seal['approval']['status'] = 'changed'
                seal_path.write_text(json.dumps(seal))
                with self.assertRaisesRegex(TaskError, 'differs'):
                    archive_approval(task)
                seal_path.write_text(original_seal)
                retained = task/'video'/state['id']/'attempt_0001'/'video.mp4'

                retained.write_bytes(b'changed after approval')
                refresh_index(task)
                manifest = json.loads((task/'delivery.json').read_text())
                self.assertFalse(manifest['versions'][0]['approval_valid'])
                self.assertIsNone(manifest['current_version'])
                return
            call('invalidate', {'shot_id':'a', 'reason':'Synthetic withdrawal regression'})
            refresh_index(task)
            manifest = json.loads((task/'delivery.json').read_text())
            self.assertIsNone(manifest['current_version'])
            self.assertFalse(manifest['versions'][0]['approval_valid'])
            self.assertEqual(manifest['versions'][0]['acceptance'], 'pending')
            self.assertTrue(output.is_file())
            with self.assertRaises(ValueError):
                delivery.export(task)
