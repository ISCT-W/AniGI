"""Structural contracts for new director-led runs; no semantic judging or API calls."""
import hashlib
import json

AUDIO = {'audio', 'dialogue', 'av_sync', 'lip_sync'}
VISUAL = {'identity', 'appearance', 'visual_artifacts', 'prompt_adherence', 'motion', 'action', 'pacing', 'narrative'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def text(v):
    return isinstance(v, str) and bool(v.strip())


def strings(v):
    return isinstance(v, list) and bool(v) and all(text(x) for x in v)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def timeline(rows, subjects):
    require(isinstance(rows, list) and len(rows) == 5, 'director timeline requires five one-second rows')
    for second, row in enumerate(rows):
        require(isinstance(row, dict) and type(row.get('second')) is int and row['second'] == second,
                'director seconds must cover 0..4 in order')
        require(all(text(row.get(k)) for k in ('scene', 'camera', 'cause', 'end_state')), 'director row needs scene/camera/cause/end_state')
        actors = row.get('actors', {})
        require(isinstance(actors, dict) and set(actors) == set(subjects), 'each second must cover every actor')
        require(all(isinstance(a, dict) and all(text(a.get(k)) for k in ('action', 'attention', 'emotion')) for a in actors.values()),
                'actor needs action/attention/emotion')


def validate(plan):
    c = plan.get('production_contract')
    if c is None:
        return  # Historical plans retain their original requirements.
    require(isinstance(c, dict) and c.get('version') in (1, 2), 'unsupported production contract')
    require(c.get('audio_review') == 'disabled', 'current production contract is visual-only')
    require(strings(c.get('subjects')) and len(set(c['subjects'])) == len(c['subjects']), 'unique actor subjects required')
    require(strings(c.get('allowed_changes')), 'allowed changes required')
    inv = c.get('invariants')
    require(isinstance(inv, list) and inv, 'strict invariants required')
    seen = set()
    for item in inv:
        require(isinstance(item, dict) and all(text(item.get(k)) for k in ('id', 'subject_id', 'attribute', 'requirement')), 'invariant fields required')
        require(item['id'] not in seen and strings(item.get('evidence_ids')), 'unique invariant and evidence IDs required')
        seen.add(item['id'])
    require(plan.get('sequence_mode') == 'reviewed_segments', 'director contract needs reviewed segments')
    for shot in plan['shots']:
        require(shot['duration_s'] == 5 and shot.get('require_audio') is False, 'new director shots are five seconds, audio optional')
        require(VISUAL <= set(shot['required_checks']) and not AUDIO & set(shot['required_checks']), 'visual checks required; audio excluded')
        require(all(set(i['evidence_ids']) <= set(shot['evidence_ids']) for i in inv), 'shot must bind all invariant references')
        timeline(shot.get('director_timeline'), c['subjects'])
    require(VISUAL <= set(plan['final_checks']) and not AUDIO & set(plan['final_checks']), 'final visual checks required; audio excluded')


def verify(state):
    c = state['plan'].get('production_contract')
    if c is not None:
        require(state.get('production_contract_sha256') == fingerprint(c), 'frozen production contract changed')


def prepare(state, data, previous):
    c = state['plan'].get('production_contract')
    if c is None:
        return
    timeline(data.get('director_timeline'), c['subjects'])
    if previous is None:
        return
    revision = data.get('revision', {})
    require(revision.get('previous_attempt_id') == previous['id'], 'revision must reference latest attempt')
    require(strings(revision.get('preserve')) and strings(revision.get('regression_checks')), 'revision needs preserved elements and regression checks')
    feedback = revision.get('feedback')
    require(isinstance(feedback, list) and feedback and all(isinstance(x, dict) and all(text(x.get(k)) for k in ('issue', 'evidence', 'change')) for x in feedback), 'revision needs actionable feedback with evidence')
    fields = ('prompt', 'image_url', 'reference_image_urls', 'reference_video_urls')
    changed = any(data.get('arguments', {}).get(k) != previous.get('arguments', {}).get(k) for k in fields)
    require(changed or previous['status'] == 'failed' and text(revision.get('technical_retry_reason')), 'unchanged quality retry is prohibited')


def review(state, report):
    c = state['plan'].get('production_contract')
    if c is None:
        return
    require(not AUDIO & set(report.get('checks', {})), 'audio review is disabled for this production')
    if report.get('decision') == 'accept':
        rows = report.get('invariant_checks', {})
        require(set(rows) == {x['id'] for x in c['invariants']}, 'review every frozen invariant')
        require(all(isinstance(x, dict) and x.get('status') == 'pass' and text(x.get('reason')) for x in rows.values()), 'strict invariant has not passed')
