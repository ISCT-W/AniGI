"""Synthetic regression tests for shared-image handoffs and managed video tasks."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from anigen import bridge
from anigen.cli import video_action
from anigen.image.backends.types import BackendError
from anigen.image.task_store import TaskStore
from anigen.video import keyframes, runtime
from anigen.workspace import create_task, load_task
from tests.video import test_keyframes as fixtures
from tests.video.test_directing import plan


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class BridgeTests(unittest.TestCase):
    image_limit = 6
    setUpClass = classmethod(fixtures.KeyframeTests.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.KeyframeTests.tearDownClass.__func__)
    setUp = fixtures.KeyframeTests.setUp
    call = fixtures.KeyframeTests.call
    prepare = fixtures.KeyframeTests.prepare
    generate = fixtures.KeyframeTests.generate
    report = fixtures.KeyframeTests.report
    accept_upload = fixtures.KeyframeTests.accept_upload
    video = fixtures.KeyframeTests.video

    def store(self):
        return TaskStore(Path(self.tmp.name) / 'keyframe')

    def reject(self, state, number):
        kid = f'keyframe_{number:04d}'
        report = self.report(state, kid)
        report.update(decision='reject', feedback=['Correct the synthetic pose'])
        report['checks']['contact']['status'] = 'fail'
        return self.call('keyframe-review', {'keyframe_id': kid, 'report': report})

    def next_image(self, number, **kwargs):
        previous = f'keyframe_{number - 1:04d}'
        revision = {'previous_keyframe_id': previous, 'feedback': ['Correct the synthetic pose'],
                    'preserve': ['Identity'], 'regression_checks': ['Pose and identity']}
        self.prepare(prompt=f'Revision {number}: correct pose', revision=revision, **kwargs)
        return self.call('keyframe-generate', {'keyframe_id': f'keyframe_{number:04d}'}, live=True)

    def test_six_total_requests_last_review_and_seventh_blocked(self):
        state = self.generate()
        for number in range(1, 7):
            self.reject(state, number)
            if number < 6:
                state = self.next_image(number + 1)
        self.assertEqual(self.fake.calls, 6)
        self.assertEqual(len(self.store().snapshot()['attempts'][-1]['reviews']), 1)
        with self.assertRaises(ValueError):
            self.next_image(7)
        self.assertEqual(self.fake.calls, 6)
        self.assertEqual(self.store().recover()['remaining'], 0)
        self.assertFalse((Path(self.tmp.name) / 'final_output').exists())
        self.assertFalse(self.call('status').get('accepted_keyframe'))
        with self.assertRaisesRegex(ValueError, 'not accepted'):
            self.call('keyframe-upload', {'keyframe_id': 'keyframe_0006'}, live=True)

    def test_single_ledger_counts_known_failures_unknown_and_not_sent(self):
        self.prepare()
        self.fake.error = BackendError('Local rejection', 'not_sent')
        with self.assertRaises(BackendError):
            self.call('keyframe-generate', {'keyframe_id': 'keyframe_0001'}, live=True)
        self.assertEqual(self.store().recover()['remaining'], 6)
        self.fake.error = BackendError('Remote rejection', 'failed')
        with self.assertRaises(BackendError):
            self.next_image(2, strategy='recompose')
        self.assertEqual(self.store().recover()['remaining'], 5)
        self.fake.error = BackendError('Uncertain response', 'unknown')
        with self.assertRaises(BackendError):
            self.next_image(3, strategy='recompose')
        self.assertEqual(self.store().recover()['remaining'], 4)
        self.assertTrue(all('charged' not in k for k in self.call('status')['keyframes'].values()))
        with self.assertRaisesRegex(ValueError, 'resolve existing'):
            self.prepare()
        self.call('keyframe-resolve', {'keyframe_id': 'keyframe_0003', 'user_instruction': 'Reconcile this request',
                   'reconciliation_evidence': 'Synthetic provider record confirms definitive failure'})
        self.assertEqual(self.store().recover()['remaining'], 4)

    def test_shared_success_recovers_after_video_state_write_interruption(self):
        self.prepare()
        original = keyframes.attach_image
        with patch.object(keyframes, 'attach_image', side_effect=OSError('synthetic interrupted write')):
            with self.assertRaises(OSError):
                self.call('keyframe-generate', {'keyframe_id': 'keyframe_0001'}, live=True)
        self.assertEqual(self.fake.calls, 1)
        self.assertEqual(self.call('status')['keyframes']['keyframe_0001']['status'], 'generation_unknown')
        state = self.call('keyframe-recover', {'keyframe_id': 'keyframe_0001'})
        self.assertEqual(state['keyframes']['keyframe_0001']['status'], 'review_pending')
        self.assertEqual(self.fake.calls, 1)
        self.assertEqual(len(self.store().snapshot()['attempts']), 1)

    def test_prepare_commit_adopted_after_video_write_interruption(self):
        self.prepare()
        with patch.object(runtime, 'save', side_effect=OSError('synthetic write failure')):
            with self.assertRaises(OSError):
                self.call('keyframe-generate', {'keyframe_id': 'keyframe_0001'}, live=True)
        self.assertEqual(len(self.store().snapshot()['attempts']), 1)
        self.assertEqual(self.fake.calls, 0)
        self.call('keyframe-generate', {'keyframe_id': 'keyframe_0001'}, live=True)
        self.assertEqual(len(self.store().snapshot()['attempts']), 1)
        self.assertEqual(self.fake.calls, 1)

    def test_review_commit_adopted_after_video_state_interruption(self):
        state = self.generate()
        report = self.report(state)
        with patch.object(runtime, 'save', side_effect=OSError('synthetic write failure')):
            with self.assertRaises(OSError):
                self.call('keyframe-review', {'keyframe_id': 'keyframe_0001', 'report': report})
        self.call('keyframe-review', {'keyframe_id': 'keyframe_0001', 'report': report})
        self.assertEqual(len(self.store().snapshot()['attempts'][0]['reviews']), 1)

    def test_generic_image_pass_cannot_bypass_video_checks(self):
        state = self.generate()
        item = state['keyframes']['keyframe_0001']
        self.store().review('001', item['candidate'], 'Static-image acceptance only', 'pass')
        with self.assertRaisesRegex(ValueError, 'not accepted'):
            keyframes.gate(state, {'shot_id': 'a'})
        report = self.report(state)
        report['checks']['motion_feasibility']['status'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'failed or unknown'):
            self.call('keyframe-review', {'keyframe_id': item['id'], 'report': report})

    def test_recompose_has_no_base_and_historical_edit_binds_originals(self):
        state = self.generate(); self.reject(state, 1)
        state = self.next_image(2, strategy='recompose')
        self.assertEqual([x['id'] for x in state['keyframes']['keyframe_0002']['request']['inputs']], ['ref'])
        self.reject(state, 2)
        state = self.next_image(3, strategy='edit', base_keyframe_id='keyframe_0001')
        refs = state['keyframes']['keyframe_0003']['request']['inputs']
        self.assertEqual([x['id'] for x in refs], ['keyframe_0001', 'ref'])
        self.assertTrue(refs[0]['is_base'])
        self.assertEqual(self.store().recover()['remaining'], 3)

    def test_changed_backend_needs_explicit_authorization_and_keeps_budget(self):
        state = self.generate(); self.reject(state, 1)
        self.fake.name = 'gemini'
        with self.assertRaises(ValueError):
            self.next_image(2, strategy='recompose', model='gemini-3-pro-image')
        # Preparation persisted in video only; no shared request has been sent.
        state = self.call('status')
        self.assertNotIn('keyframe_0002', state['keyframes'])
        self.assertEqual(self.fake.calls, 1)
        self.assertEqual(self.store().recover()['remaining'], 5)

    def test_explicit_backend_change_uses_same_shared_ledger(self):
        state = self.generate(); self.reject(state, 1)
        self.fake.name = 'gemini'
        state = self.next_image(2, strategy='recompose', model='gemini-3-pro-image',
                                backend_authorization='User explicitly selects Gemini for this revision')
        self.assertEqual(state['keyframes']['keyframe_0002']['request']['backend'], 'gemini')
        self.assertEqual([a['backend'] for a in self.store().snapshot()['attempts']], ['gpt', 'gemini'])
        self.assertEqual(self.store().recover()['remaining'], 4)
        self.assertFalse((Path(self.tmp.name) / 'figs').exists())

    def test_wrong_frame_dimensions_require_rejection_even_with_pass_report(self):
        self.prepare(size='1792x1008')
        state = self.call('keyframe-generate', {'keyframe_id': 'keyframe_0001'}, live=True)
        self.assertTrue(state['keyframes']['keyframe_0001']['technical_issues'])
        with self.assertRaisesRegex(ValueError, 'technical constraints'):
            self.call('keyframe-review', {'keyframe_id': 'keyframe_0001', 'report': self.report(state)})
        self.reject(state, 1)

    def test_original_and_director_plan_changes_invalidate_handoff(self):
        state = self.accept_upload()
        changed = copy.deepcopy(state)
        changed['plan']['script'] += ' Changed action.'
        with self.assertRaisesRegex(ValueError, 'director plan changed'):
            keyframes.gate(changed, self.video(changed))
        changed = copy.deepcopy(state)
        changed['evidence']['ref']['claim'] += ' Changed original claim.'
        with self.assertRaisesRegex(ValueError, 'references changed'):
            keyframes.gate(changed, self.video(changed))

    def test_uploaded_url_and_bytes_cannot_be_replaced_in_state(self):
        state = self.accept_upload()
        upload = state['keyframe_uploads'][0]
        item = state['keyframes']['keyframe_0001']
        upload['result']['access_url'] = 'https://assets.example/replaced.png'
        state['evidence'][item['evidence_id']]['generation_url'] = upload['result']['access_url']
        with self.assertRaisesRegex(ValueError, 'durable provider receipt'):
            keyframes.gate(state, self.video(state))

    def test_upload_receipt_recovers_without_second_upload(self):
        state = self.generate()
        self.call('keyframe-review', {'keyframe_id': 'keyframe_0001', 'report': self.report(state)})
        self.call('authorize', {'user_instruction':'Synthetic only','max_generations':2,'max_generated_seconds':10,'max_attempts_per_shot':2})
        with patch.object(keyframes, 'complete_upload', side_effect=OSError('synthetic video-state write failure')):
            with self.assertRaises(OSError):
                self.call('keyframe-upload', {'keyframe_id':'keyframe_0001'}, live=True)
        with patch.object(fixtures.FakeUpload, 'upload', side_effect=AssertionError('must not upload again')):
            state = self.call('keyframe-recover', {'keyframe_id':'keyframe_0001','upload_id':'keyframe_upload_0001'})
        keyframes.gate(state, self.video(state))
        self.assertEqual(len(state['keyframe_uploads']), 1)

    def test_reopen_keeps_ledger_and_revokes_shared_pass(self):
        state = self.accept_upload()
        candidate = state['keyframes']['keyframe_0001']['candidate']
        self.call('keyframe-reopen', {'keyframe_id':'keyframe_0001','reason':'Correct opening action'})
        with self.assertRaises(ValueError):
            self.store().validate_pass('001', candidate)
        self.assertEqual(self.store().recover()['remaining'], 5)

    def test_root_offline_video_authorization_never_reads_env(self):
        with tempfile.TemporaryDirectory() as root:
            task = create_task(root, 'video', 'Synthetic film', backend='gpt', brief='A synthetic film')
            p = plan(); p['reference_scope_id'] = 'offline-fixture'
            run_id = load_task(task)[2]['video_run_id']
            with patch('anigen.video.settings.read_settings', side_effect=AssertionError('no configuration read')):
                video_action(task, 'init', p)
                video_action(task, 'keyframe-authorize', self.policy)
            self.assertEqual(TaskStore(task / 'keyframe').snapshot()['mode'], 'offline')
            self.assertFalse((task.parent.parent / 'figs').exists())
            with self.assertRaisesRegex(ValueError, 'offline'):
                runtime.dispatch('poll', run_id, {'attempt_id':'none'}, runs=task/'video')
            with self.assertRaisesRegex(ValueError, 'offline'):
                runtime.dispatch('keyframe-upload', run_id, {'keyframe_id':'none'}, runs=task/'video',
                                 live=True, image_provider=self.fake)

    def test_root_task_freezes_references_and_enforces_initial_backend(self):
        with tempfile.TemporaryDirectory() as root:
            task = create_task(root, 'video', 'Synthetic framed film', backend='gpt', brief='Synthetic only')
            p = plan(); p['reference_scope_id'] = 'offline-fixture'; p['source_scope'] = 'synthetic-source'
            video_action(task, 'init', p)
            evidence = {'id':'ref', 'reference_scope_id':'offline-fixture', 'source_scope':'synthetic-source',
                        'kind':'user_reference', 'verification':'verified', 'verification_basis':'direct_observation',
                        'verification_method':'synthetic_fixture', 'observation':'Synthetic pixels',
                        'locator':'fixture:original', 'claim':'Synthetic toy identity',
                        'observation_path':str(self.fixture), 'observation_sha256':runtime.digest(self.fixture),
                        'observation_mime_type':'image/png'}
            state = video_action(task, 'evidence', evidence)
            self.assertEqual(Path(state['evidence']['ref']['observation_path']), task / 'references' / 'ref.png')
            self.assertEqual((task/'references/ref.png').read_bytes(), self.fixture.read_bytes())
            video_action(task, 'keyframe-authorize', self.policy)
            data = {'prompt':'Synthetic opening', 'design':self.design, 'backend':'gemini', 'model':'gemini-3-pro-image'}
            with self.assertRaisesRegex(ValueError, 'explicit user instruction'):
                video_action(task, 'keyframe-prepare', data)
            state = video_action(task, 'keyframe-prepare', {**data, 'backend_authorization':'Explicit user choice'})
            self.assertEqual(state['keyframes']['keyframe_0001']['request']['backend'], 'gemini')

    def test_video_lock_contention_never_duplicates_shared_send(self):
        self.prepare()
        with runtime.locked('test', Path(self.tmp.name) / 'video'):
            with self.assertRaises(BlockingIOError):
                self.call('keyframe-generate', {'keyframe_id':'keyframe_0001'}, live=True)
        self.assertEqual(self.fake.calls, 0)
        self.call('keyframe-generate', {'keyframe_id':'keyframe_0001'}, live=True)
        self.assertEqual(self.fake.calls, 1)

    def test_reference_capacity_never_discards_review_originals(self):
        state = self.call('status')
        base = state['evidence']['ref']
        for number in range(1, 8):
            state['evidence'][f'copy{number}'] = {**base, 'id':f'copy{number}'}
        state['plan']['shots'][0]['evidence_ids'] = list(state['evidence'])
        with self.assertRaisesRegex(ValueError, '1..7'):
            keyframes.references(state)
        self.assertEqual(len(state['plan']['shots'][0]['evidence_ids']), 8)
        self.assertEqual(self.fake.calls, 0)

    def test_incomplete_shared_response_cannot_mark_keyframe_accepted(self):
        from anigen.image.backends.types import ImageOutput, ImageResult
        self.prepare()
        result = ImageResult((ImageOutput(self.fixture.read_bytes(), 'image/png'),), complete=False)
        with patch.object(self.fake, 'generate', return_value=result):
            state = self.call('keyframe-generate', {'keyframe_id':'keyframe_0001'}, live=True)
        with self.assertRaises(ValueError):
            self.call('keyframe-review', {'keyframe_id':'keyframe_0001','report':self.report(state)})
        self.assertFalse(self.call('status').get('accepted_keyframe'))

    def test_corrupt_returned_image_preserves_charge_and_can_recompose(self):
        from anigen.image.backends.types import ImageOutput, ImageResult
        self.prepare()
        result = ImageResult((ImageOutput(b'\x89PNG\r\n\x1a\nnot-a-decoded-image', 'image/png'),))
        with patch.object(self.fake, 'generate', return_value=result):
            with self.assertRaisesRegex(ValueError, 'failed decoding'):
                self.call('keyframe-generate', {'keyframe_id':'keyframe_0001'}, live=True)
        self.assertEqual(self.call('status')['keyframes']['keyframe_0001']['status'], 'failed')
        self.assertEqual(self.store().recover()['remaining'], 5)
        self.next_image(2, strategy='recompose')
        self.assertEqual(self.store().recover()['remaining'], 4)

    def _interrupt_failure_sync(self, failure, *, before_reservation=False):
        self.prepare()
        original_save = runtime.save
        calls = 0
        def fail_last_save(path, state):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError('synthetic outer-state failure')
            return original_save(path, state)
        target = 'validate' if before_reservation else 'generate'
        with patch.object(self.fake, target, side_effect=failure), patch.object(runtime, 'save', side_effect=fail_last_save):
            with self.assertRaises(OSError):
                self.call('keyframe-generate', {'keyframe_id':'keyframe_0001'}, live=True)
        self.assertEqual(self.call('status')['keyframes']['keyframe_0001']['status'], 'generation_unknown')
        return self.call('keyframe-recover', {'keyframe_id':'keyframe_0001'})

    def test_recover_unreserved_shared_round_after_outer_state_loss(self):
        state = self._interrupt_failure_sync(BackendError('Local validation', 'not_sent'), before_reservation=True)
        self.assertEqual(state['keyframes']['keyframe_0001']['status'], 'prepared')
        self.assertEqual(self.store().recover()['remaining'], 6)
        self.call('keyframe-generate', {'keyframe_id':'keyframe_0001'}, live=True)
        self.assertEqual(self.fake.calls, 1)

    def test_recover_definitive_shared_failure_after_outer_state_loss(self):
        state = self._interrupt_failure_sync(BackendError('Definitive remote failure', 'failed'))
        self.assertEqual(state['keyframes']['keyframe_0001']['status'], 'failed')
        self.assertEqual(self.store().recover()['remaining'], 5)
        self.next_image(2, strategy='recompose')
        self.assertEqual(self.store().recover()['remaining'], 4)

    def test_recover_confirmed_not_sent_after_outer_state_loss(self):
        state = self._interrupt_failure_sync(BackendError('Confirmed local send failure', 'not_sent'))
        self.assertEqual(state['keyframes']['keyframe_0001']['status'], 'failed')
        self.assertEqual(self.store().recover()['remaining'], 6)

    def test_reopen_commit_recovers_without_duplicate_revocation(self):
        self.accept_upload()
        with patch.object(runtime, 'save', side_effect=OSError('synthetic outer-state failure')):
            with self.assertRaises(OSError):
                self.call('keyframe-reopen', {'keyframe_id':'keyframe_0001','reason':'Correct opening action'})
        state = self.call('keyframe-reopen', {'keyframe_id':'keyframe_0001','reason':'Correct opening action'})
        self.assertFalse(state.get('accepted_keyframe'))
        self.assertEqual(self.store().recover()['remaining'], 5)
        self.assertEqual(len(self.store().snapshot()['attempts'][0]['reviews']), 2)
        self.next_image(2, strategy='recompose')

    def test_multiple_returned_images_rejects_all_without_wedging_or_selecting(self):
        from anigen.image.backends.types import ImageOutput, ImageResult
        self.prepare()
        output = ImageOutput(self.fixture.read_bytes(), 'image/png')
        with patch.object(self.fake, 'generate', return_value=ImageResult((output, output))):
            with self.assertRaises(ValueError):
                self.call('keyframe-generate', {'keyframe_id':'keyframe_0001'}, live=True)
        item = self.call('status')['keyframes']['keyframe_0001']
        self.assertEqual(item['status'], 'failed')
        self.assertNotIn('image', item)
        attempt = self.store().snapshot()['attempts'][0]
        self.assertEqual(len(attempt['outputs']), 2)
        self.assertEqual([r['verdict'] for r in attempt['reviews']], ['fail', 'fail'])
        self.assertEqual(self.store().recover()['remaining'], 5)
        self.next_image(2, strategy='recompose')

    def test_definitive_service_failure_allows_documented_same_prompt_retry(self):
        self.prepare()
        self.fake.error = BackendError('Definitive service error', 'failed')
        with self.assertRaises(BackendError):
            self.call('keyframe-generate', {'keyframe_id':'keyframe_0001'}, live=True)
        self.fake.error = None
        revision = {'previous_keyframe_id':'keyframe_0001','feedback':['Service rejected the request'],
                    'preserve':['Original valid prompt and references'],'regression_checks':['Same acceptance criteria'],
                    'technical_retry_reason':'Provider confirmed a temporary service failure'}
        self.prepare(revision=revision, strategy='recompose')
        self.call('keyframe-generate', {'keyframe_id':'keyframe_0002'}, live=True)
        self.assertEqual(self.store().recover()['remaining'], 4)

    def test_explicit_reauthorization_keeps_used_budget_and_scope(self):
        state = self.generate(); self.reject(state, 1)
        self.call('stop')
        renewed = self.call('keyframe-reauthorize', {'user_instruction':'Continue the same task within its remaining budget'})
        self.assertTrue(renewed['keyframe_policy']['active'])
        self.assertEqual(renewed['keyframe_policy']['max_generations'], 6)
        self.assertEqual(renewed['keyframe_policy']['max_uploads'], 2)
        self.assertEqual(self.store().recover()['remaining'], 5)
        self.next_image(2, strategy='recompose')
        self.assertEqual(self.store().recover()['remaining'], 4)

    def test_exhausted_stage_reauthorization_never_replenishes(self):
        state = self.generate()
        for number in range(1, 7):
            self.reject(state, number)
            if number < 6:
                state = self.next_image(number + 1)
        self.call('stop')
        self.call('keyframe-reauthorize', {'user_instruction':'Continue only within the original six requests'})
        self.assertEqual(self.store().recover()['remaining'], 0)
        with self.assertRaises(ValueError):
            self.next_image(7, strategy='recompose')
        self.assertEqual(self.fake.calls, 6)

    def test_reauthorization_is_idempotent_after_outer_state_loss(self):
        state = self.generate(); self.reject(state, 1); self.call('stop')
        instruction = {'user_instruction':'Resume the remaining original requests'}
        with patch.object(runtime, 'save', side_effect=OSError('synthetic outer-state failure')):
            with self.assertRaises(OSError):
                self.call('keyframe-reauthorize', instruction)
        self.call('keyframe-reauthorize', instruction)
        self.assertEqual(len(self.store().snapshot()['batches']), 2)
        self.assertEqual(self.store().recover()['remaining'], 5)

    def test_reauthorization_cannot_change_caps_or_reinterpret_prepared_request(self):
        self.prepare(); self.call('stop')
        with self.assertRaisesRegex(ValueError, 'caps cannot change'):
            self.call('keyframe-reauthorize', {'user_instruction':'Resume', 'max_generations':6})
        with self.assertRaisesRegex(ValueError, 'prepared or uncertain'):
            self.call('keyframe-reauthorize', {'user_instruction':'Resume'})

    def test_stopped_prepared_draft_can_be_discarded_then_reauthorized(self):
        self.prepare(); self.call('stop')
        state = self.call('keyframe-discard', {'keyframe_id':'keyframe_0001','reason':'Cancel unsubmitted draft'})
        self.assertEqual(state['keyframes']['keyframe_0001']['status'], 'failed')
        self.assertEqual(self.store().recover()['remaining'], 6)
        self.call('keyframe-reauthorize', {'user_instruction':'Continue with a new draft under the original cap'})
        self.next_image(2, strategy='recompose')
        self.assertEqual(self.store().recover()['remaining'], 5)

    def test_discard_recovers_from_outer_state_loss_without_budget_change(self):
        self.prepare(); self.call('stop')
        data = {'keyframe_id':'keyframe_0001','reason':'Cancel unsubmitted draft'}
        with patch.object(runtime, 'save', side_effect=OSError('synthetic outer-state failure')):
            with self.assertRaises(OSError):
                self.call('keyframe-discard', data)
        self.call('keyframe-discard', data)
        self.assertEqual(self.store().recover()['remaining'], 6)
        self.assertEqual(len(self.store().snapshot()['attempts']), 1)
        self.assertEqual(self.fake.calls, 0)

    def test_discard_cannot_clear_uncertain_generation(self):
        self.prepare(); self.fake.error = BackendError('Uncertain response', 'unknown')
        with self.assertRaises(BackendError):
            self.call('keyframe-generate', {'keyframe_id':'keyframe_0001'}, live=True)
        with self.assertRaises(ValueError):
            self.call('keyframe-discard', {'keyframe_id':'keyframe_0001','reason':'Not a valid cancellation'})
        with self.assertRaises(ValueError):
            self.store().abandon_prepared('001', 'Not a valid cancellation')
        self.assertEqual(self.store().recover()['remaining'], 5)
