"""Durable handoff between the shared image ledger and video opening-frame checks.

Video state references this ledger; it never owns a second image request budget.
The video run lock is acquired before any image-store lock. Each shared mutation
is committed first, so a retry can adopt its immutable result without sending.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .image.generation import generate_round
from .image.task_store import TaskStore, StoreError, immutable_write


def _record_text(item, value):
    if not item.get('readable_records'):
        return _json(value)
    from .workspace import readable_record
    return "\n".join(readable_record(value)) + "\n"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _require(condition, message):
    if not condition:
        raise StoreError(message)


def _check_file(store, record):
    path = (store.path / record['path']).resolve()
    _require(path.is_relative_to(store.path.resolve()), 'shared image record escapes its task')
    _require(path.is_file() and _digest(path.read_bytes()) == record['sha256'], 'shared image record changed')
    return path


def _store(state):
    link = state.get('shared_image_store')
    _require(isinstance(link, dict), 'shared image stage is not authorized')
    store = TaskStore(link['path'])
    _require(store.snapshot()['id'] == link['task_id'], 'shared image task identity changed')
    return store


def initialize(state, directory, policy, *, backend=None):
    """Create once under the enclosing video task, or adopt a committed creation."""
    directory = Path(directory).resolve()
    _require(directory.parent.name == 'video', 'video runtime needs task_root/video and a safe internal run ID')
    stage = directory.parent.parent / 'keyframe'
    task_contract = state.get('task_contract')
    mode = task_contract['mode'] if task_contract else ('offline' if backend is not None and not getattr(backend, 'production', True) else 'generation')
    brief = _json({'video_run_id': state['id'], 'brief': state['plan']['brief'],
                   'plan': state['plan'], 'image_authorization': policy})
    if stage.exists():
        store = TaskStore(stage)
        saved = store.snapshot()
        _require(_check_file(store, saved['brief']).read_text() == brief,
                 'existing shared image task belongs to different video authorization')
    else:
        store = TaskStore.create(stage, brief, 'Opening frame', mode=mode,
                                 authorization=policy['user_instruction'], limit=policy['max_generations'])
    saved = store.snapshot()
    state['shared_image_store'] = {'path': str(stage), 'task_id': saved['id']}


def prepare(state, item):
    store = _store(state)
    request = item['request']
    inputs = [('base' if ref.get('is_base') else 'reference', ref['path']) for ref in request['inputs']]
    base = next((ref for ref in request['inputs'] if ref.get('is_base')), None)
    parent = str(Path(base['path']).relative_to(store.path)) if base else None
    if not item.get('shared_round'):
        item['readable_records'] = True
    round_id = store.prepare(_record_text(item, {'keyframe_id': item['id'], 'request': request}),
                             _record_text(item, item['reference_snapshot']), request['backend'], request['model'],
                             inputs=inputs, parent=parent, submission=request['prompt'],
                             operation='edit' if base else 'generate',
                             aspect_ratio=request['aspect_ratio'],
                             pixel_size=request['size'] if request['backend'] == 'gpt' else None,
                             quality='high' if request['backend'] == 'gpt' else None,
                             image_size=request.get('image_size') if request['backend'] == 'gemini' else None,
                             idempotency_key=state['id'] + ':' + item['id'],
                             backend_authorization=request.get('backend_authorization'))
    link = {'task_id': store.snapshot()['id'], 'round_id': round_id}
    if item.get('shared_round'):
        _require(item['shared_round'] == link, 'shared image round changed')
    item['shared_round'] = link
    return store


def bound_attempt(state, item):
    store = _store(state)
    link = item.get('shared_round')
    _require(link and link['task_id'] == store.snapshot()['id'], 'opening frame has no shared image round')
    snapshot = store.snapshot()
    attempt = next((a for a in snapshot['attempts'] if a['id'] == link['round_id']), None)
    _require(attempt is not None, 'shared image round is missing')
    _require(_check_file(store, attempt['prompt']).read_text() == _record_text(item, {'keyframe_id': item['id'], 'request': item['request']}),
             'shared image request binding changed')
    _require(_check_file(store, attempt['reference']).read_text() == _record_text(item, item['reference_snapshot']),
             'shared image reference binding changed')
    for ref in attempt['inputs']:
        _check_file(store, ref)
    _require(attempt['backend'] == item['request']['backend'] and attempt['model'] == item['request']['model'],
             'shared image provider binding changed')
    return store, attempt


def _immutable(store, relative, payload):
    path = store._path(relative)
    raw = _json(payload).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    immutable_write(path, raw, adopt=True)
    return {'path': str(path), 'sha256': _digest(raw)}


def sync(state, item):
    store, attempt = bound_attempt(state, item)
    status = attempt['status']
    if status in ('reserved', 'unknown'):
        item['status'] = 'generation_unknown'
        return None
    if status in ('failed', 'not_sent'):
        item['status'] = 'failed'
        return None
    if status == 'prepared':
        item['status'] = 'prepared'
        return None
    _require(status == 'succeeded', 'unsupported shared image state')
    if len(attempt['outputs']) != 1:
        reason = 'Opening-frame request returned an unexpected candidate count; none selected automatically'
        item.update(status='failed', technical_issues=[reason])
        technical_reject(state, item, reason)
        return None
    output = attempt['outputs'][0]
    candidate = _check_file(store, output)
    receipt = {'task_id': store.snapshot()['id'], 'round_id': attempt['id'], 'image': output,
               'request_hash': item['request_hash'], 'reference_snapshot': item['reference_snapshot'],
               'backend': attempt['backend'], 'model': attempt['model'], 'response': attempt.get('receipt')}
    item['shared_receipt'] = _immutable(store, 'bridges/' + item['id'] + '-generation.json', receipt)
    item['candidate'] = candidate.name
    return candidate


def generate(state, item, *, backend=None):
    store = prepare(state, item)
    _, attempt = bound_attempt(state, item)
    if attempt['status'] == 'prepared':
        # generate_round validates, reserves, sends once and persists the result.
        generate_round(store, attempt['id'], execute=True, evidence_ready=True, backend=backend)
    elif attempt['status'] != 'succeeded':
        sync(state, item)
        raise StoreError('shared image request is unresolved or completed; recover instead of resending')
    return sync(state, item)


def recover(state, item):
    store, attempt = bound_attempt(state, item)
    if attempt['status'] in ('unknown', 'reserved'):
        store.collect(attempt['id'])
    return sync(state, item)


def resolve(state, item, reconciliation):
    store, attempt = bound_attempt(state, item)
    _require(attempt['status'] in ('unknown', 'reserved'), 'shared image request is not unresolved')
    store.finish(attempt['id'], 'failed', note=_json(reconciliation))
    sync(state, item)


def review(state, item, report):
    store, attempt = bound_attempt(state, item)
    verdict = {'accept': 'pass', 'reject': 'fail', 'needs_review': 'unverified'}[report['decision']]
    report_text = _record_text(item, report)
    match = next((r for r in reversed(attempt['reviews']) if r['candidate'] == item['candidate']
                  and r['report']['sha256'] == _digest(report_text.encode())), None)
    if match:
        _check_file(store, match['report'])
        return match
    rows = {**report['checks'], **report['invariant_checks']}
    return store.review(attempt['id'], item['candidate'], report_text, verdict,
                        blockers=[name for name, row in rows.items() if row['status'] == 'fail'],
                        unknowns=[name for name, row in rows.items() if row['status'] == 'unknown'])


def verify(state, item, *, accepted=False):
    store, attempt = bound_attempt(state, item)
    if item.get('image'):
        output = next((o for o in attempt['outputs'] if Path(o['path']).name == item['candidate']), None)
        _require(output is not None, 'opening-frame output missing from shared ledger')
        path = _check_file(store, output)
        _require(str(path) == item['image']['path'] and output['sha256'] == item['image']['sha256'],
                 'opening-frame pixels differ from shared image result')
        receipt = item.get('shared_receipt', {})
        expected_path = store.path / ('bridges/' + item['id'] + '-generation.json')
        expected = {'task_id': store.snapshot()['id'], 'round_id': attempt['id'], 'image': output,
                    'request_hash': item['request_hash'], 'reference_snapshot': item['reference_snapshot'],
                    'backend': attempt['backend'], 'model': attempt['model'], 'response': attempt.get('receipt')}
        _require(receipt.get('path') == str(expected_path) and expected_path.is_file()
                 and _digest(expected_path.read_bytes()) == receipt.get('sha256')
                 and expected_path.read_text() == _json(expected), 'shared image handoff receipt changed')
    if accepted:
        approved = store.validate_pass(attempt['id'], item['candidate'])
        _require(approved['review']['report']['sha256'] == _digest(_record_text(item, item['reviews'][-1]).encode()),
                 'shared image review differs from opening-frame acceptance')
        return approved


def _reopen_payload(item, reason):
    return {'keyframe_id': item['id'], 'reason': reason, 'image_sha256': item['image']['sha256'],
            'request_hash': item['request_hash'], 'accepted_review_sha256': _digest(_record_text(item, item['reviews'][-1]).encode())}


def reopen_pending(state, item, reason):
    store, _ = bound_attempt(state, item)
    path = store.path / ('bridges/' + item['id'] + '-reopen.json')
    if not path.exists():
        return False
    _require(path.read_text() == _json(_reopen_payload(item, reason)), 'pending reopen intent changed')
    return True


def revoke(state, item, reason):
    store, attempt = bound_attempt(state, item)
    report = _json({'reopened': reason})
    latest = attempt['reviews'][-1]
    existing = latest['verdict'] == 'fail' and latest['report']['sha256'] == _digest(report.encode())
    if existing:
        _require(reopen_pending(state, item, reason), 'revocation has no durable reopen intent')
        _check_file(store, latest['report'])
        return
    store.validate_pass(attempt['id'], item['candidate'])
    _immutable(store, 'bridges/' + item['id'] + '-reopen.json', _reopen_payload(item, reason))
    store.review(attempt['id'], item['candidate'], report, 'fail', blockers=[reason])


def technical_reject(state, item, reason):
    """Record a technical rejection, without claiming semantic visual review."""
    store, attempt = bound_attempt(state, item)
    report = _json({'technical_rejection': reason, 'visual_review': 'not asserted'})
    for output in attempt['outputs']:
        candidate = Path(output['path']).name
        recorded = next((r for r in attempt['reviews'] if r['candidate'] == candidate
                         and r['report']['sha256'] == _digest(report.encode())), None)
        if recorded:
            _check_file(store, recorded['report'])
        else:
            store.review(attempt['id'], candidate, report, 'fail', blockers=[reason])


def upload_receipt(state, item, upload, result):
    store = _store(state)
    payload = {'keyframe_id': item['id'], 'upload_id': upload['id'],
               'image_sha256': item['image']['sha256'], 'result': result}
    return _immutable(store, 'bridges/' + upload['id'] + '.json', payload)


def load_upload(state, item, upload):
    store = _store(state)
    path = store.path / ('bridges/' + upload['id'] + '.json')
    _require(path.is_file(), 'no durable upload receipt; reconcile remotely without resending')
    payload = json.loads(path.read_text())
    _require(payload.get('keyframe_id') == item['id'] and payload.get('upload_id') == upload['id']
             and payload.get('image_sha256') == item['image']['sha256'], 'upload receipt belongs to different image')
    receipt = upload_receipt(state, item, upload, payload['result'])
    if upload.get('receipt'):
        _require(upload['receipt'] == receipt, 'upload receipt identity changed')
    return payload['result'], receipt


def verify_upload(state, item, upload):
    result, receipt = load_upload(state, item, upload)
    _require(upload.get('receipt') == receipt and upload.get('result') == result,
             'uploaded opening frame differs from durable provider receipt')
    _require(result.get('source_sha256') == item['image']['sha256'], 'uploaded bytes differ from accepted image')
    return result


def reauthorize(state, instruction):
    """Renew only the original authorization scope, with a durable batch identity."""
    store = _store(state)
    saved = store.snapshot()
    _require(not any(a['status'] in ('prepared', 'reserved', 'unknown') for a in saved['attempts']),
             'resolve prepared or uncertain shared image requests before renewal')
    number = len(state.get('keyframe_reauthorizations', [])) + 1
    relative = f'bridges/reauthorization-{number:03d}.json'
    path = store.path / relative
    policy = {key: value for key, value in state['keyframe_policy'].items() if key != 'active'}
    if path.exists():
        intent = json.loads(path.read_text())
        _require(intent.get('instruction') == instruction and intent.get('policy') == policy,
                 'pending image authorization renewal changed')
    else:
        intent = {'instruction': instruction, 'policy': policy, 'batch_id': len(saved['batches']) + 1}
    receipt = _immutable(store, relative, intent)
    if len(saved['batches']) == intent['batch_id'] - 1:
        store.new_batch(instruction, limit=policy['max_generations'])
    else:
        _require(len(saved['batches']) == intent['batch_id'], 'image authorization renewal batch changed')
        batch = saved['batches'][-1]
        _require(batch['limit'] == policy['max_generations']
                 and _check_file(store, batch['authorization']).read_text() == instruction,
                 'image authorization renewal does not match the committed batch')
    return receipt


def discard(state, item, reason):
    """Cancel only an unreserved shared draft; uncertain sends cannot enter here."""
    store = prepare(state, item) if not item.get('shared_round') else _store(state)
    _, attempt = bound_attempt(state, item)
    intent = {'keyframe_id': item['id'], 'round_id': attempt['id'], 'request_hash': item['request_hash'], 'reason': reason}
    receipt = _immutable(store, 'bridges/' + item['id'] + '-discard.json', intent)
    store.abandon_prepared(attempt['id'], reason)
    return receipt
