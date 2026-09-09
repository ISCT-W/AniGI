"""Versioned opening-frame design, bounded image requests and semantic review gate.

Reports are trusted agent attestations, not proof that a model saw pixels.
"""
import json
import os
from pathlib import Path
import re
import subprocess

from . import directing
from .. import bridge
from .settings import get_setting

ACTIONS = {'keyframe-authorize', 'keyframe-reauthorize', 'keyframe-discard', 'keyframe-prepare', 'keyframe-generate',
           'keyframe-review', 'keyframe-upload', 'keyframe-reopen', 'keyframe-recover',
           'keyframe-resolve'}
CHECKS = {'identity', 'style', 'product', 'contact', 'composition', 'motion_feasibility'}
DESIGN = {'composition', 'starting_action', 'motion_feasibility', 'reference_roles'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def enabled(state):
    return state['plan'].get('production_contract', {}).get('version') == 2


def image_info(path):
    path = Path(path).resolve()
    require(path.is_file() and 0 < path.stat().st_size < 50 * 1024 * 1024, 'image missing or exceeds50MiB')
    info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)],
                     check=True, capture_output=True, timeout=30).stdout)
    streams = info['streams']
    require(len(streams) == 1 and streams[0].get('codec_name') in ('png', 'mjpeg', 'webp'), 'still PNG/JPEG/WebP required')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-frames:v', '1', '-f', 'null', '-'],
                   check=True, capture_output=True, timeout=30)
    mime = {'png': 'image/png', 'mjpeg': 'image/jpeg', 'webp': 'image/webp'}[streams[0]['codec_name']]
    from .runtime import digest
    return {'path': str(path), 'sha256': digest(path), 'mime_type': mime,
            'width': streams[0]['width'], 'height': streams[0]['height']}


def references(state):
    ids = state['plan']['shots'][0]['evidence_ids']
    result = {}
    for eid in ids:
        ref = state['evidence'].get(eid, {})
        require(ref.get('verification') == 'verified' and ref.get('source_scope') == state['plan'].get('source_scope') and
                ref.get('reference_scope_id') == state['plan']['reference_scope_id'], 'keyframe needs verified original references in the selected scope')
        info = image_info(ref.get('observation_path', ''))
        require(info['sha256'] == ref.get('observation_sha256') and info['mime_type'] == ref.get('observation_mime_type'),
                'original reference file binding changed')
        result[eid] = {**info, 'evidence_hash': directing.fingerprint(ref)}
    require(1 <= len(result) <= 7, 'opening frame supports1..7 original image references')
    return result


def verify_candidate(state, item):
    require(item['reference_snapshot'] == references(state), 'keyframe original references changed')
    require(item['request_hash'] == directing.fingerprint(item['request']), 'keyframe request changed')
    require(item['request'].get('plan_hash') == directing.fingerprint(state['plan']), 'keyframe director plan changed')
    if item.get('image'):
        require(item['image'] == image_info(item['image']['path']), 'keyframe image changed')
    if item.get('shared_round'):
        bridge.verify(state, item)


def validate_review(state, item, report):
    require(report.get('reviewer') == 'codex' and report.get('review_authority') == 'codex_director', 'keyframe reviewer must be current Codex; independent app needs explicit identity migration')
    require(report.get('image_sha256') == item['image']['sha256'] and report.get('request_hash') == item['request_hash'], 'keyframe review belongs to other image or design')
    require(set(report.get('reference_ids', [])) == set(item['reference_snapshot']), 'keyframe review must compare every original')
    require(set(report.get('observed_images', [])) == {item['image']['path'], *[r['path'] for r in item['reference_snapshot'].values()]}, 'attest actual candidate and all original images viewed')
    require(directing.text(report.get('summary')) and directing.strings(report.get('limitations')), 'keyframe summary and limitations required')
    require(report.get('decision') in ('accept', 'reject', 'needs_review'), 'invalid keyframe verdict')
    inv = {r['id'] for r in state['plan']['production_contract']['invariants']}
    for key, expected in [('checks', CHECKS), ('invariant_checks', inv)]:
        rows = report.get(key, {})
        require(set(rows) == expected, 'review all keyframe checks and strict invariants')
        require(all(isinstance(r, dict) and r.get('status') in ('pass', 'fail', 'unknown') and directing.text(r.get('reason')) for r in rows.values()), 'each keyframe check needs a status and visible reason')
        if report['decision'] == 'accept':
            require(all(r['status'] == 'pass' for r in rows.values()), 'keyframe has failed or unknown checks')
    if report['decision'] == 'accept':
        require(not item.get('technical_issues'), 'keyframe technical constraints have not passed')
    if report['decision'] != 'accept':
        require(directing.strings(report.get('feedback')), 'nonaccepted keyframe needs actionable feedback')


def accepted(state, kid):
    k = state.get('keyframes', {}).get(kid)
    require(k and state.get('accepted_keyframe') == kid and k['status'] == 'accepted', 'opening frame is not accepted')
    verify_candidate(state, k)
    report = k['reviews'][-1]
    validate_review(state, k, report)
    require(report['decision'] == 'accept' and k['accepted_review_hash'] == directing.fingerprint(report), 'keyframe acceptance changed')
    bridge.verify(state, k, accepted=True)
    return k


def gate(state, data):
    if not enabled(state) or data['shot_id'] != state['plan']['shots'][0]['id']:
        return
    k = accepted(state, data.get('keyframe_id'))
    require(data.get('endpoint', '').endswith('/image-to-video'), 'first segment requires accepted opening I2V')
    eid = k.get('evidence_id'); ref = state['evidence'].get(eid, {})
    require(eid and ref.get('observation_sha256') == k['image']['sha256'] and ref.get('observation_path') == k['image']['path'], 'accepted keyframe must be uploaded and bound')
    uploads = state.get('keyframe_uploads', [])
    u = next((u for u in uploads if u['status'] == 'completed' and u['evidence_id'] == eid and u['keyframe_id'] == k['id']), None)
    require(u is not None, 'keyframe URL must match actual upload receipt')
    result = bridge.verify_upload(state, k, u)
    require(result['access_url'] == ref.get('generation_url'), 'keyframe URL must match actual upload receipt')
    require(data.get('arguments', {}).get('image_url') == ref.get('generation_url') and
            data.get('reference_bindings', {}).get(ref.get('generation_url')) == eid, 'video must use exact accepted keyframe input')


def attach_image(state, item, candidate):
    require(candidate is not None, 'shared image response has no candidate')
    try:
        info = image_info(candidate)
    except subprocess.CalledProcessError:
        reason = 'Returned bytes are not a decodable still image'
        item.update(status='failed', technical_issues=[reason])
        bridge.technical_reject(state, item, reason)
        raise ValueError('returned image failed decoding; preserve receipt and revise without resending this request') from None
    item.update(image=info, status='review_pending', technical_issues=[])
    width, height = map(int, item['request']['size'].split('x'))
    if item['request']['backend'] == 'gpt' and (info['width'], info['height']) != (width, height):
        item['technical_issues'].append('Image dimensions differ from the exact request')
    a, b = map(int, item['request']['aspect_ratio'].split(':'))
    if abs(info['width'] / info['height'] - a / b) >= 0.01:
        item['technical_issues'].append('Opening-frame aspect ratio differs from the first video')
    verify_candidate(state, item)


def complete_upload(state, item, upload, result, receipt):
    from .runtime import _https_url
    require(result.get('source_sha256') == item['image']['sha256'], 'keyframe upload hash mismatch')
    _https_url(result.get('access_url'))
    verify_candidate(state, item)
    eid = upload['evidence_id']
    state['evidence'][eid] = {'id': eid, 'reference_scope_id': state['plan']['reference_scope_id'],
        'source_scope': state['plan'].get('source_scope'), 'kind': 'user_reference', 'verification': 'verified',
        'locator': 'keyframe:' + item['id'], 'claim': 'Accepted opening layout; original references remain authoritative',
        'verification_method': 'reviewed_image_upload', 'verification_basis': 'artifact_review',
        'observation': 'Actual image and original reference hashes bound to review',
        'observation_path': item['image']['path'], 'observation_sha256': item['image']['sha256'],
        'observation_mime_type': item['image']['mime_type'], 'generation_url': result['access_url']}
    references = state['authorization'].setdefault('reference_transfer_ids', [])
    if eid not in references:
        references.append(eid)
    item['evidence_id'] = eid
    upload.update(status='completed', result=result, receipt=receipt)


def handle(action, state, directory, path, data, live, client=None, uploader=None):
    from .runtime import save, event, _https_url
    require(enabled(state), 'keyframe tools require production contract v2; historical runs are unchanged')
    records = state.setdefault('keyframes', {})
    policy = state.get('keyframe_policy')
    if action == 'keyframe-authorize':
        require(policy is None, 'image authorization already recorded; cannot reset run budget')
        require(directing.text(data.get('user_instruction')) and directing.text(data.get('transfer_basis')), 'explicit image generation and provider transfer instruction required')
        require(type(data.get('max_generations')) is int and 1 <= data['max_generations'] <= 6, 'image generation cap must be1..6')
        require(type(data.get('max_uploads')) is int and 1 <= data['max_uploads'] <= 5, 'image upload cap must be1..5')
        require(data.get('allow_generated_edits') is True and data.get('allow_fal_upload') is True, 'explicit generated-image editing and fal upload permission required')
        require(set(data.get('reference_transfer_ids', [])) == set(state['plan']['shots'][0]['evidence_ids']), 'authorize exact original reference set')
        bridge.initialize(state, directory, data, backend=client)
        state['keyframe_policy'] = {**data, 'active': True}
    elif action == 'keyframe-reauthorize':
        require(policy and not policy['active'], 'only an inactive image authorization can be renewed')
        require(set(data) == {'user_instruction'} and directing.text(data.get('user_instruction')), 'renewal needs a new explicit instruction; original scope and caps cannot change')
        require(not any(k['status'] in ('prepared', 'generation_unknown') for k in records.values()), 'resolve prepared or uncertain image requests before renewal')
        require(not any(u['status'] == 'upload_unknown' for u in state.get('keyframe_uploads', [])), 'resolve uncertain uploads before renewal')
        receipt = bridge.reauthorize(state, data['user_instruction'])
        state.setdefault('keyframe_reauthorizations', []).append({'user_instruction': data['user_instruction'], 'receipt': receipt})
        policy['active'] = True
    elif action == 'keyframe-prepare':
        require(not state.get('accepted_keyframe'), 'reopen accepted keyframe explicitly before redesign')
        require(not any(k['status'] in ('prepared', 'generation_unknown', 'review_pending') for k in records.values()), 'resolve existing keyframe first')
        require(not any(u['status'] == 'upload_unknown' for u in state.get('keyframe_uploads', [])), 'resolve uncertain keyframe upload first')
        require(directing.text(data.get('prompt')), 'keyframe prompt required')
        design = data.get('design', {})
        require(set(design) == DESIGN and all(directing.text(design.get(k)) for k in DESIGN - {'reference_roles'}), 'composition, starting action, feasibility and reference roles required')
        refs = references(state)
        require(set(design.get('reference_roles', {})) == set(refs) and all(directing.text(v) for v in design['reference_roles'].values()), 'assign each original reference a role')
        size = data.get('size', '1536x864'); match = re.fullmatch(r'(\d+)x(\d+)', size)
        require(match is not None, 'explicit image dimensions required')
        width, height = map(int, match.groups())
        require(all(16 <= x <= 3840 and x % 16 == 0 for x in (width, height)) and 655360 <= width * height <= 8294400 and 1/3 <= width/height <= 3, 'unsupported bounded image size')
        ratio = state['plan']['shots'][0].get('aspect_ratio', '16:9').split(':')
        require(abs(width / height - int(ratio[0]) / int(ratio[1])) < 0.01, 'keyframe aspect must match first video')
        task_contract = state.get('task_contract')
        backend = data.get('backend') or (task_contract['initial_backend'] if task_contract else None) or get_setting('IMAGE_BACKEND')
        require(backend in ('gpt', 'gemini'), 'select gpt or gemini image backend')
        previous_backend = list(records.values())[-1]['request']['backend'] if records else (task_contract['initial_backend'] if task_contract else backend)
        if backend != previous_backend:
            require(directing.text(data.get('backend_authorization')), 'changing the selected image backend requires explicit user instruction')
        model = data.get('model') or get_setting('OPENAI_IMAGE_MODEL' if backend == 'gpt' else 'GEMINI_IMAGE_MODEL')
        require(isinstance(model, str) and bool(model.strip()), 'configure an available image model')
        strategy = data.get('strategy', 'edit' if records and list(records.values())[-1].get('image') else 'recompose')
        require(strategy in ('edit', 'recompose'), 'choose edit or recompose strategy')
        inputs = [{**v, 'id': eid, 'role': design['reference_roles'][eid]} for eid, v in refs.items()]
        if records:
            old = list(records.values())[-1]
            rev = data.get('revision', {})
            require(rev.get('previous_keyframe_id') == old['id'] and directing.strings(rev.get('feedback')) and directing.strings(rev.get('preserve')) and directing.strings(rev.get('regression_checks')), 'keyframe retry needs latest feedback, preserved items and regression checks')
            technical_retry = (old['status'] == 'failed' and not old.get('image')
                               and backend == old['request']['backend'] and directing.text(rev.get('technical_retry_reason'))
                               and old.get('shared_round') and bridge.bound_attempt(state, old)[1]['status'] in ('failed', 'not_sent'))
            require(data['prompt'] != old['request']['director_prompt'] or technical_retry, 'unchanged image retry is prohibited without a definitive technical failure and recorded retry reason')
            if strategy == 'edit':
                base = records.get(data.get('base_keyframe_id', old['id']))
                require(base and base.get('image'), 'edit strategy requires a recorded historical candidate')
                verify_candidate(state, base)
                inputs.insert(0, {**base['image'], 'id': base['id'], 'is_base': True, 'role': 'Candidate to edit; fix recorded defects, original references remain authoritative.'})
        else:
            require(strategy == 'recompose', 'first image must use original references without a generated base')
        prompt = data['prompt'] + '\nInput order and roles:\n' + '\n'.join(f'{i+1}: {r["id"]}: {r["role"]}' for i, r in enumerate(inputs))
        request = {'plan_hash': directing.fingerprint(state['plan']), 'backend': backend, 'backend_authorization': data.get('backend_authorization'), 'strategy': strategy, 'aspect_ratio': ':'.join(ratio), 'image_size': data.get('image_size'), 'model': model, 'size': size, 'prompt': prompt, 'director_prompt': data['prompt'], 'design': design, 'inputs': inputs, 'revision': data.get('revision')}
        kid = f'keyframe_{len(records)+1:04d}'
        records[kid] = {'id': kid, 'status': 'prepared', 'request': request, 'request_hash': directing.fingerprint(request), 'reference_snapshot': refs, 'reviews': []}
    else:
        k = records.get(data.get('keyframe_id'))
        require(k is not None, 'unknown keyframe')
        if action == 'keyframe-discard':
            require(k['status'] == 'prepared' and directing.text(data.get('reason')), 'only a prepared draft can be discarded with a reason')
            receipt = bridge.discard(state, k, data['reason'])
            k.update(status='failed', discard_reason=data['reason'], discard_receipt=receipt)
        elif action == 'keyframe-generate':
            require(live and policy and policy['active'], 'live image generation authorization required')
            require(k['status'] == 'prepared' and not any(x['status'] == 'generation_unknown' for x in records.values()), 'unresolved or already submitted image; recover first')
            verify_candidate(state, k)
            require(set(k['reference_snapshot']) <= set(policy['reference_transfer_ids']), 'image reference transfer not authorized')
            bridge.prepare(state, k)
            save(path, state)  # Round identity is durable before the shared ledger sends.
            k['status'] = 'generation_unknown'
            save(path, state)
            try:
                candidate = bridge.generate(state, k, backend=client)
                attach_image(state, k, candidate)
            except Exception as exc:
                k['error_type'] = type(exc).__name__
                candidate = bridge.sync(state, k)
                if candidate is not None and k['status'] != 'failed':
                    attach_image(state, k, candidate)
                save(path, state)
                raise
        elif action == 'keyframe-recover':
            if data.get('upload_id'):
                u = next((u for u in state.get('keyframe_uploads', []) if u['id'] == data['upload_id'] and u['keyframe_id'] == k['id']), None)
                require(u and u['status'] == 'upload_unknown', 'upload is not unresolved')
                accepted(state, k['id'])
                result, receipt = bridge.load_upload(state, k, u)
                complete_upload(state, k, u, result, receipt)
            else:
                require(k['status'] in ('generation_unknown', 'prepared'), 'only unresolved image can recover')
                candidate = bridge.recover(state, k)
                if candidate is not None:
                    try:
                        attach_image(state, k, candidate)
                    except ValueError:
                        save(path, state)
                        raise
                else:
                    require(k['status'] in ('prepared', 'failed'), 'shared request is still unresolved; reconcile without resending')
        elif action == 'keyframe-resolve':
            require(directing.text(data.get('user_instruction')) and directing.text(data.get('reconciliation_evidence')), 'explicit external reconciliation and instruction required; timeout is not proof of failure')
            if data.get('upload_id'):
                u = next((u for u in state.get('keyframe_uploads', []) if u['id'] == data['upload_id'] and u['keyframe_id'] == k['id']), None)
                require(u and u['status'] == 'upload_unknown', 'upload is not unresolved')
                require(not (Path(state['shared_image_store']['path']) / 'bridges' / (u['id'] + '.json')).exists(), 'saved upload receipt exists; recover it instead')
                u.update(status='failed', reconciliation=data)
            else:
                require(k['status'] == 'generation_unknown', 'image is not unresolved')
                bridge.resolve(state, k, data)
                k.update(status='failed', reconciliation=data)
            # Reservation retained; no remote cancellation or refund is claimed.
        elif action == 'keyframe-review':
            require(k['status'] == 'review_pending', 'keyframe not awaiting review')
            verify_candidate(state, k); report = data['report']; validate_review(state, k, report)
            bridge.review(state, k, report)
            k['reviews'].append(report)
            if report['decision'] == 'accept':
                bridge.verify(state, k, accepted=True)
                k.update(status='accepted', accepted_review_hash=directing.fingerprint(report)); state['accepted_keyframe'] = k['id']
            elif report['decision'] == 'reject':
                k['status'] = 'rejected'
        elif action == 'keyframe-reopen':
            require(directing.text(data.get('reason')), 'reopen reason required')
            require(not state['accepted'] and not state.get('final') and not any(a['status'] in ('submitted', 'submission_unknown', 'completed', 'review_pending') for a in state['attempts']), 'invalidate downstream and resolve active video before keyframe revision')
            if bridge.reopen_pending(state, k, data['reason']):
                require(state.get('accepted_keyframe') == k['id'] and k['status'] == 'accepted', 'reopen has no matching current acceptance')
                verify_candidate(state, k)
            else:
                accepted(state, k['id'])
            require(not any(u['status'] == 'upload_unknown' for u in state.get('keyframe_uploads', [])), 'reconcile upload before reopening')
            for a in state['attempts']:
                if a['status'] == 'prepared': a['status'] = 'superseded'
            bridge.revoke(state, k, data['reason'])
            k.update(status='rejected', reopen_reason=data['reason']); state['accepted_keyframe'] = None
        elif action == 'keyframe-upload':
            require(live and policy and policy['active'] and policy['allow_fal_upload'], 'live keyframe upload permission required')
            accepted(state, k['id'])
            auth = state.get('authorization'); require(auth and auth['active'], 'active video authorization required before upload')
            require(not k.get('evidence_id'), 'keyframe already uploaded; reuse receipt')
            uploads = state.setdefault('keyframe_uploads', [])
            require(not any(u['status'] == 'upload_unknown' for u in uploads), 'reconcile uncertain upload; no automatic retry')
            require(len(uploads) < policy['max_uploads'], 'keyframe upload budget exhausted')
            if uploads: require(directing.text(data.get('reason')), 'additional upload needs reason')
            if uploader is None:
                from .fal_upload import FalMediaUploader
                uploader = FalMediaUploader()
            uploader.preflight()
            uid = f'keyframe_upload_{len(uploads)+1:04d}'; eid = uid + '_image'
            require(eid not in state['evidence'], 'keyframe evidence ID already exists')
            u = {'id': uid, 'keyframe_id': k['id'], 'evidence_id': eid, 'status': 'upload_unknown'}; uploads.append(u); save(path, state)
            try:
                result = uploader.upload(Path(k['image']['path']), k['image']['mime_type'], expected_sha256=k['image']['sha256'])
                require(result.get('source_sha256') == k['image']['sha256'], 'keyframe upload hash mismatch')
                _https_url(result.get('access_url')); verify_candidate(state, k)
                receipt = bridge.upload_receipt(state, k, u, result)
            except Exception as exc:
                u['error_type'] = type(exc).__name__
                if getattr(exc, 'outcome_unknown', True) is False: u['status'] = 'failed'
                save(path, state); raise
            complete_upload(state, k, u, result, receipt)
    event(state, action, keyframe_id=data.get('keyframe_id'))
