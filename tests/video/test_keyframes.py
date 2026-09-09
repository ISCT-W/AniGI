"""Offline opening-frame gates and one-request adapter tests; never paid calls."""
import base64
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from anigen.video import keyframes, runtime
from anigen.image.backends.types import BackendError, ImageResult, ImageOutput, ImageRequest, ImageInput
from anigen.image.backends.gpt import GPTBackend
from anigen.image.task_store import TaskStore
from tests.video.test_directing import plan


class FakeImage:
    name = 'gpt'
    production = False
    def __init__(self, fixture, error=None):
        self.fixture, self.error, self.calls = fixture, error, 0
    def validate(self, request): pass
    def generate(self, request):
        self.calls += 1
        if self.error: raise self.error
        return ImageResult((ImageOutput(self.fixture.read_bytes(), 'image/png'),), model=request.model)


class FakeUpload:
    production = False
    def preflight(self): pass
    def upload(self, path, mime_type, expected_sha256):
        return {'source_sha256':expected_sha256, 'access_url':'https://fal.media/fixture-keyframe.png'}


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class KeyframeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.fixture = Path(cls.temp.name)/'original.png'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=red:s=1536x864','-frames:v','1',str(cls.fixture)],check=True)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.fake = FakeImage(self.fixture); self.p = plan(); self.p['source_scope']='synthetic-source'
        self.call('init',self.p)
        self.call('evidence',{'id':'ref','reference_scope_id':'synthetic-scope','source_scope':'synthetic-source','kind':'user_reference','verification':'verified', "verification_basis": "direct_observation",'verification_method':'synthetic_fixture','observation':'Synthetic red image','locator':'fixture:original','claim':'Synthetic original','observation_path':str(self.fixture),'observation_sha256':runtime.digest(self.fixture),'observation_mime_type':'image/png'})
        self.policy={'user_instruction':'Synthetic test only','transfer_basis':'Fake endpoints only','max_generations':getattr(self,'image_limit',2),'max_uploads':2,'allow_generated_edits':True,'allow_fal_upload':True,'reference_transfer_ids':['ref']}
        self.call('keyframe-authorize',self.policy)
        self.design={'composition':'Oblique','starting_action':'Holding toy','motion_feasibility':'Hands occupied; eye movement available','reference_roles':{'ref':'Original design authority'}}
    def call(self,action,data=None,**kwargs):
        return runtime.dispatch(action,'test',data,runs=Path(self.tmp.name)/'video',image_provider=self.fake,uploader=FakeUpload(),**kwargs)
    def prepare(self,**kwargs):
        return self.call('keyframe-prepare',{'prompt':'Opening toy scene','design':self.design,'model':'gpt-image-2','backend':self.fake.name,**kwargs})
    def generate(self):
        self.prepare();return self.call('keyframe-generate',{'keyframe_id':'keyframe_0001'},live=True)
    def report(self,s,kid='keyframe_0001'):
        k=s['keyframes'][kid]
        return {'review_authority':'codex_director','reviewer':'codex','image_sha256':k['image']['sha256'],'request_hash':k['request_hash'],'reference_ids':['ref'],'observed_images':[k['image']['path'],k['reference_snapshot']['ref']['path']],'summary':'Synthetic gate fixture, not real visual acceptance','limitations':['Synthetic pixels only'],'decision':'accept','checks':{c:{'status':'pass','reason':'Synthetic fixture'} for c in keyframes.CHECKS},'invariant_checks':{'toy_pose':{'status':'pass','reason':'Synthetic fixture'}}}
    def accept_upload(self):
        s=self.generate();self.call('keyframe-review',{'keyframe_id':'keyframe_0001','report':self.report(s)})
        self.call('authorize',{'user_instruction':'Offline fixture','max_generations':2,'max_generated_seconds':10,'max_attempts_per_shot':2})
        return self.call('keyframe-upload',{'keyframe_id':'keyframe_0001'},live=True)
    def video(self,s):
        k=s['keyframes']['keyframe_0001'];eid=k['evidence_id'];url=s['evidence'][eid]['generation_url']
        return {'shot_id':'a','keyframe_id':k['id'],'endpoint':'minimax/h3-max/image-to-video','arguments':{'prompt':'Animate eye movement','image_url':url,'duration':5,'resolution':'768P','prompt_expansion_mode':'balanced'},'reference_bindings':{url:eid},'director_timeline':self.p['shots'][0]['director_timeline']}
    def test_missing_rejected_unknown_cannot_pass(self):
        with self.assertRaisesRegex(ValueError,'not accepted'):keyframes.gate(self.call('status'),{'shot_id':'a'})
        s=self.generate();r=self.report(s)
        for status in ('fail','unknown'):
            r['checks']['contact']['status']=status
            with self.assertRaisesRegex(ValueError,'failed or unknown'):self.call('keyframe-review',{'keyframe_id':'keyframe_0001','report':r})
        r['decision']='needs_review';r['feedback']=['Inspect contact'];self.call('keyframe-review',{'keyframe_id':'keyframe_0001','report':r})
        with self.assertRaisesRegex(ValueError,'resolve existing'):self.prepare()
    def test_accept_actual_upload_and_prepare_then_tamper(self):
        s=self.accept_upload();v=self.video(s);self.call('prepare',v)
        with self.assertRaisesRegex(ValueError,'already uploaded'):self.call('keyframe-upload',{'keyframe_id':'keyframe_0001'},live=True)
        img=Path(s['keyframes']['keyframe_0001']['image']['path']);img.write_bytes(img.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'image changed'):self.call('submit',{'attempt_id':'attempt_0001'},live=True)
    def test_wrong_url_design_hash_and_review_identity(self):
        s=self.accept_upload();v=self.video(s);v['arguments']['image_url']='https://fal.media/wrong.png'
        with self.assertRaisesRegex(ValueError,'exact accepted'):self.call('prepare',v)
        k=s['keyframes']['keyframe_0001'];k['request']['design']['composition']='Other'
        with self.assertRaisesRegex(ValueError,'request changed'):keyframes.gate(s,self.video(s))
    def test_reopen_invalidates_prepared_video(self):
        s=self.accept_upload();self.call('prepare',self.video(s))
        s=self.call('keyframe-reopen',{'keyframe_id':'keyframe_0001','reason':'Actual defect found'})
        self.assertEqual(s['attempts'][0]['status'],'superseded')
        with self.assertRaisesRegex(ValueError,'not accepted'):keyframes.gate(s,self.video(s))
    def test_budget_retries_keep_parent_and_originals(self):
        s=self.generate();r=self.report(s);r.update(decision='reject',feedback=['Wrong contact']);r['checks']['contact']['status']='fail'
        self.call('keyframe-review',{'keyframe_id':'keyframe_0001','report':r})
        rev={'previous_keyframe_id':'keyframe_0001','feedback':['Contact wrong: move fingers'],'preserve':['Face'],'regression_checks':['Face unchanged']}
        with self.assertRaisesRegex(ValueError,'unchanged'):self.prepare(revision=rev)
        s=self.prepare(prompt='Repair contact; preserve face',revision=rev)
        self.assertEqual([i['id'] for i in s['keyframes']['keyframe_0002']['request']['inputs']],['keyframe_0001','ref'])
        with self.assertRaisesRegex(ValueError,'cannot reset'):self.call('keyframe-authorize',self.policy)
        self.call('keyframe-generate',{'keyframe_id':'keyframe_0002'},live=True)
        with self.assertRaisesRegex(ValueError,'already submitted'):self.call('keyframe-generate',{'keyframe_id':'keyframe_0002'},live=True)
    def test_exhausted_budget_blocks_third_request(self):
        for n in (1,2):
            kid=f'keyframe_{n:04d}'
            extra={} if n==1 else {'prompt':'Fix contact again','revision':{'previous_keyframe_id':'keyframe_0001','feedback':['Contact evidence requires revision'],'preserve':['Face'],'regression_checks':['Pose']}}
            self.prepare(**extra);s=self.call('keyframe-generate',{'keyframe_id':kid},live=True)
            r=self.report(s,kid);r.update(decision='reject',feedback=['Wrong contact']);r['checks']['contact']['status']='fail'
            self.call('keyframe-review',{'keyframe_id':kid,'report':r})
        self.prepare(prompt='Change palm angle specifically',revision={'previous_keyframe_id':'keyframe_0002','feedback':['Wrong support surface'],'preserve':['Face'],'regression_checks':['Pose']})
        with self.assertRaisesRegex(ValueError,'上限|budget exhausted'):self.call('keyframe-generate',{'keyframe_id':'keyframe_0003'},live=True)
        self.assertEqual(self.fake.calls,2)
    def test_known_local_failure_does_not_require_remote_reconciliation(self):
        self.prepare();self.fake.error=BackendError('Input changed before send', 'not_sent')
        with self.assertRaises(BackendError):self.call('keyframe-generate',{'keyframe_id':'keyframe_0001'},live=True)
        self.assertEqual(self.call('status')['keyframes']['keyframe_0001']['status'],'failed')
        store=TaskStore(Path(self.tmp.name)/'keyframe')
        self.assertEqual(store.snapshot()['attempts'][0]['status'],'not_sent')
        self.assertEqual(store.recover()['remaining'],2)
        bad=ImageRequest('gpt-image-2','fixture',inputs=(ImageInput('reference',b'wrong','image/png'),))
        with self.assertRaises(BackendError): GPTBackend('').validate(bad)
    def test_unknown_submission_blocks_retry_and_recovery_no_post(self):
        self.prepare();self.fake.error=BackendError('Uncertain send', 'unknown')
        with self.assertRaises(BackendError):self.call('keyframe-generate',{'keyframe_id':'keyframe_0001'},live=True)
        with self.assertRaisesRegex(ValueError,'resolve existing'):self.prepare()
        store=TaskStore(Path(self.tmp.name)/'keyframe')
        store.stage_result('001', [(self.fixture.read_bytes(), '.png')], 'Recovered exact synthetic response')
        s=self.call('keyframe-recover',{'keyframe_id':'keyframe_0001'})
        self.assertEqual(s['keyframes']['keyframe_0001']['status'],'review_pending');self.assertEqual(self.fake.calls,1)
    def test_no_live_and_closed_policy(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError,'authorization'):self.call('keyframe-generate',{'keyframe_id':'keyframe_0001'})
        self.call('stop')
        with self.assertRaisesRegex(ValueError,'authorization'):self.call('keyframe-generate',{'keyframe_id':'keyframe_0001'},live=True)
        self.assertEqual(self.fake.calls,0)
    def test_old_contract_only_readable_not_new_init(self):
        p=copy.deepcopy(self.p);p['production_contract']['version']=1
        with self.assertRaisesRegex(ValueError,'version2'):runtime.dispatch('init','old',p,runs=self.tmp.name)
    def test_adapter_one_multipart_request_and_safe_output(self):
        raw=json.dumps({'data':[{'b64_json':base64.b64encode(self.fixture.read_bytes()).decode()}],'usage':{'total_tokens':1}}).encode()
        calls=[]
        def transport(url,headers,body,timeout):
            calls.append((url,headers,body));return 200,raw
        req=ImageRequest('gpt-image-2','Synthetic only',inputs=(ImageInput('reference',self.fixture.read_bytes(),'image/png'),),pixel_size='1536x864')
        result=GPTBackend('fixture-key',transport=transport).generate(req)
        self.assertEqual(len(calls),1);self.assertEqual(result.images[0].data,self.fixture.read_bytes())
        self.assertIn(b'name="image[]"',calls[0][2])
        self.assertNotIn(str(self.fixture).encode(),calls[0][2])
