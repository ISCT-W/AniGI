"""Offline contract counterexamples and real local audio removal; no model calls."""
import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from anigen.video import directing, media, runtime


def plan():
    rows = [{'second':i, 'scene':'Corridor unchanged', 'camera':'Hold oblique shot',
             'actors':{'a':{'action':'Continue holding fixed toy', 'attention':'Partner', 'emotion':'Amused'}},
             'cause':'Respond after partner gesture', 'end_state':'Toy supported'} for i in range(5)]
    return {'brief':'Test visual story','script':'A holds toy', 'reference_scope_id':'synthetic-scope',
            'sequence_mode':'reviewed_segments',
            'production_contract':{'version':2,'audio_review':'disabled','subjects':['a'],
                'invariants':[{'id':'toy_pose','subject_id':'toy','attribute':'pose','requirement':'Rigid wings', 'evidence_ids':['ref']}],
                'allowed_changes':['Lighting and perspective only']},
            'shots':[{'id':'a','goal':'Hold toy','duration_s':5,'require_audio':False,
                      'required_checks':sorted(directing.VISUAL),'evidence_ids':['ref'], 'director_timeline':rows}],
            'final_checks':sorted(directing.VISUAL)}


class DirectorContractTests(unittest.TestCase):
    def test_new_run_cannot_omit_contract_or_raise_attempt_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=plan(); del p['production_contract']
            with self.assertRaisesRegex(ValueError, 'new runs require'):
                runtime.dispatch('init','missing',p,runs=tmp)
            runtime.dispatch('init','valid',plan(),runs=tmp)
            with self.assertRaisesRegex(ValueError, 'at most five'):
                runtime.dispatch('authorize','valid',{'user_instruction':'offline only','max_generations':6,'max_generated_seconds':30,'max_attempts_per_shot':6},runs=tmp)

    def test_init_freezes_contract_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=runtime.dispatch('init','test',plan(),runs=tmp)
            directing.verify(s)
            s['plan']['production_contract']['invariants'][0]['requirement']='Can flap'
            with self.assertRaisesRegex(ValueError,'frozen'):
                directing.verify(s)

    def test_each_second_and_actor_required(self):
        for mutate in ('missing_second','wrong_second','missing_actor'):
            p=plan(); rows=p['shots'][0]['director_timeline']
            if mutate=='missing_second': rows.pop()
            elif mutate=='wrong_second': rows[2]['second']=3
            else: rows[2]['actors']={}
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): runtime.validate_plan(p)

    def test_audio_cannot_reenter_new_plan(self):
        p=plan();p['shots'][0]['required_checks'].append('dialogue')
        with self.assertRaises(ValueError):runtime.validate_plan(p)
        with self.assertRaises(ValueError):directing.review({'plan':plan()},{'checks':{'audio':{'status':'fail'}}})

    def test_strict_failure_cannot_be_accepted(self):
        for status in ('fail','unknown'):
            with self.subTest(status=status), self.assertRaises(ValueError):
                directing.review({'plan':plan()},{'decision':'accept','invariant_checks':{'toy_pose':{'status':status,'reason':'Wings moved'}}})
        directing.review({'plan':plan()},{'decision':'accept','invariant_checks':{'toy_pose':{'status':'pass','reason':'Rigid wings throughout visible turn'}}})

    def test_retry_needs_latest_feedback_and_real_change(self):
        p=plan();previous={'id':'attempt_0001','status':'rejected','arguments':{'prompt':'Hold toy'},'director_timeline':p['shots'][0]['director_timeline']}
        data=copy.deepcopy(previous);data['revision']={'previous_attempt_id':'attempt_0001', 'feedback':[{'issue':'Wings flap','evidence':'review at1s','change':'Specify rigid paper sculpture'}], 'preserve':['Camera'], 'regression_checks':['Face identity']}
        with self.assertRaisesRegex(ValueError,'unchanged'): directing.prepare({'plan':p},data,previous)
        data['arguments']['seed']=2
        with self.assertRaisesRegex(ValueError,'unchanged'): directing.prepare({'plan':p},data,previous)
        data['arguments']['prompt']='Hold rigid paper sculpture, wings fixed'
        directing.prepare({'plan':p},data,previous)
        data['revision']['previous_attempt_id']='attempt_0000'
        with self.assertRaisesRegex(ValueError,'latest'):directing.prepare({'plan':p},data,previous)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
    def test_runtime_sends_silent_copy_to_observer(self):
        from tests.video.test_observation_runtime import FakeObserver
        DEFAULT_TEST_SCOPE = "synthetic-scope"
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=red:s=160x90:r=24:d=5','-f','lavfi','-i','sine=duration=5','-c:v','libx264','-c:a','aac','-shortest',str(source)],check=True)
            p=plan();p['reference_scope_id']=DEFAULT_TEST_SCOPE;p['source_scope']='synthetic-source'
            def call(action,data=None,**kw):return runtime.dispatch(action,'visual',data,runs=tmp,**kw)
            from tests.video.legacy_fixture import seed_legacy
            p['production_contract']['version']=1
            seed_legacy('visual',p,tmp)
            call('evidence',{'id':'ref','reference_scope_id':DEFAULT_TEST_SCOPE,'source_scope':'synthetic-source','kind':'user_reference','verification':'verified', "verification_basis": "direct_observation",'verification_method':'synthetic_fixture','observation':'red','locator':'synthetic:red','claim':'red'})
            call('import-video',{'shot_id':'a','media_path':str(source),'user_instruction':'offline synthetic fixture'})
            call('observation-policy',{'user_instruction':'offline fake observation','allow_task_spec':True,'allow_generated_media':False,'imported_target_ids':['attempt_0001'],'reference_transfer_ids':[], 'max_calls':3,'max_calls_per_target':2,'max_local_calls_per_target':1,'max_media_seconds':30,'reserve_final_calls':1,'reserve_final_media_seconds':5,'max_output_tokens':1000,'max_total_output_tokens':3000,'qualifications':[],'review_authority':'codex_director'})
            s=call('observe-av',{'target_id':'attempt_0001'},observer=FakeObserver(),live=True)
            req=s['av_observations']['observation_0001']['request']
            self.assertTrue(req['audio_review_disabled'])
            self.assertFalse(req['audio_included_in_request'])
            self.assertFalse(media.inspect_video(Path(req['media_path']))['has_audio'])
            self.assertTrue(media.inspect_video(source)['has_audio'])
            self.assertEqual(req['spec']['production_contract'],p['production_contract'])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
    def test_visual_copy_keeps_video_packets_and_removes_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=red:s=160x90:r=24:d=1','-f','lavfi','-i','sine=duration=1','-c:v','libx264','-c:a','aac','-shortest',str(source)],check=True)
            output=media.visual_only_copy(source,root/'silent')
            self.assertTrue(media.inspect_video(source)['has_audio'])
            self.assertFalse(media.inspect_video(output)['has_audio'])
            def packets(path):
                return subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-c','copy','-f','hash','-hash','sha256','-'])
            self.assertEqual(packets(source),packets(output))
