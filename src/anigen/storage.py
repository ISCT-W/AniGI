"""Compact completed tasks without pretending discarded caches were rechecked."""
from __future__ import annotations

import json
import fcntl
from contextlib import contextmanager, ExitStack
from pathlib import Path

from .workspace import TaskError, artifact_path, file_hash, load_task, save_json, task_lock

RECEIPT = '.archive.json'


def archived(task):
    return artifact_path(task, RECEIPT).is_file()


def require_active(task):
    if archived(task):
        raise TaskError('Task is compacted: restore evidence or re-review before further production; budgets are unchanged')


def archive_approval(task):
    """Verify retained bytes, including all approval state, against a prior check.

    This is historical approval verification, not a replay of deleted observations.
    Existing files scheduled for deletion must still match until deletion completes.
    """
    path = Path(task).resolve()
    data = json.loads(artifact_path(path, RECEIPT).read_text())
    if data.get('schema') != 1 or data.get('kind') != 'verified-completed-video':
        raise TaskError('Invalid archive receipt')
    _, _, metadata = load_task(path)
    if data.get('task_id') != metadata['id']:
        raise TaskError('Archive belongs to a different task')
    for name, expected in data['retained'].items():
        if file_hash(artifact_path(path, name)) != expected:
            raise TaskError('Retained archive evidence changed')
    for name, expected in data['removed'].items():
        candidate = artifact_path(path, name)
        if candidate.exists() and file_hash(candidate) != expected:
            raise TaskError('Archive cleanup candidate changed')
    from .video import observation
    state_name = f"video/{metadata['video_run_id']}/state.json"
    mandatory = {'task.json', state_name, 'keyframe/state.json'}
    if not mandatory <= data['retained'].keys():
        raise TaskError('Archive is missing required state bindings')
    state = json.loads(artifact_path(path, state_name).read_text())
    final = state.get('final')
    if not final or final.get('status') != 'accepted' or not final.get('reviews'):
        raise TaskError('Archived final approval was withdrawn')
    expected = {**final, 'review_sha256': observation.content_hash(final['reviews'][-1])}
    source = artifact_path(path, final['observation']['media_path'])
    if source.relative_to(path).as_posix() not in data['retained'] or expected != data['approval']:
        raise TaskError('Archive approval differs from retained state')
    if file_hash(source) != final['observation']['media_sha256']:
        raise TaskError('Archived media differs from its reviewed hash')
    return expected


def _candidates(task, state):
    """Only disposable caches and redundant transport inputs; keep actual opinions."""
    removable = set()
    run = task / 'video' / state['id']
    for file in run.rglob('*'):
        if not file.is_file():
            continue
        parts = file.relative_to(run).parts
        if 'frames' in parts or 'visual-only' in parts or parts[0].startswith('window-'):
            removable.add(file)
        if file.name == 'window.mp4' and 'ending' in parts:
            removable.add(file)
        if file.name.startswith('state.pre-'):
            removable.add(file)
    for attempt in state.get('attempts', []):
        # Keep the exact reviewed derivative; source identity remains in state.
        if attempt.get('normalizations'):
            removable.add(run / attempt['id'] / 'video.mp4')
    for file in task.rglob('*'):
        if file.is_file() and file.name == '.DS_Store':
            removable.add(file)
    # The runtime stores executed requests, reports, evidence and plans in state.
    for file in (task / 'director').glob('*-input.json'):
        payload = json.loads(file.read_text())
        if isinstance(payload, dict) and set(payload) == {'attempt_id'} and any(
                a['id'] == payload['attempt_id'] for a in state.get('attempts', [])):
            removable.add(file)
    # Frozen input copies are redundant only when their exact content survives.
    outside = {file_hash(f) for f in task.rglob('*') if f.is_file()
               and f not in removable and 'inputs' not in f.relative_to(task).parts}
    for file in (task / 'keyframe' / 'rounds').glob('*/inputs/*'):
        if file.is_file() and file_hash(file) in outside:
            removable.add(file)
    return removable


@contextmanager
def _image_lock(task):
    lock = artifact_path(task, 'keyframe/.lock')
    if not lock.parent.is_dir():
        raise TaskError('Shared image stage is missing')
    with lock.open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def compact(task, *, extra_remove=()):
    """Seal before deletion; interrupted deletions can resume idempotently.

    extra_remove is an explicit local audit manifest, never inferred from names.
    Prompts, reports, states and final products cannot be removed via that input.
    """
    from .config import use_workspace
    from .video import runtime
    with task_lock(task) as (path, root, metadata), use_workspace(root):
        if metadata['purpose'] != 'video':
            raise TaskError('Compaction currently requires a completed video task')
        with runtime.locked(metadata['video_run_id'], runs=path / 'video') as (_, _, state), ExitStack() as locks:
            receipt = artifact_path(path, RECEIPT)
            if receipt.exists():
                locks.enter_context(_image_lock(path))
                archive_approval(path)
                record = json.loads(receipt.read_text())
            else:
                if any(a.get('status') not in {'accepted', 'rejected', 'failed', 'superseded'}
                       for a in state.get('attempts', [])):
                    raise TaskError('Resolve unfinished generation before compaction')
                image_file = artifact_path(path, 'keyframe/state.json')
                image_bytes = image_file.read_bytes()
                image_state = json.loads(image_bytes)
                if any(a.get('status') not in {'succeeded', 'failed', 'not_sent'} for a in image_state['attempts']):
                    raise TaskError('Resolve unfinished image generation before compaction')
                for collection in (state.get('keyframe_uploads', []), state.get('handoff_uploads', {})):
                    for upload in (collection.values() if isinstance(collection, dict) else collection):
                        if upload.get('status') not in {'completed', 'failed'}:
                            raise TaskError('Resolve unfinished upload before compaction')
                for observation in state.get('av_observations', {}).values():
                    if observation.get('status') not in {'completed', 'failed', 'interrupted'}:
                        raise TaskError('Resolve unfinished observation before compaction')
                    if observation.get('cleanup_status') != 'complete' or observation.get('remote_upload_outcome_unknown') or observation.get('result', {}).get('remote_upload_outcome_unknown'):
                        raise TaskError('Resolve remote cleanup before compaction')
                approval = runtime.validate_final(state)
                locks.enter_context(_image_lock(path))
                if image_file.read_bytes() != image_bytes:
                    raise TaskError("Image state changed during compaction validation")
                removable = _candidates(path, state)
                for name in extra_remove:
                    file = artifact_path(path, name)
                    if not ((file.parent == path / 'references' and file.suffix in {'.png', '.jpg', '.mp4'})
                            or (file.parent == path / 'director' and file.suffix == '.json')
                            or (file.parent == path and file.suffix == '.py')):
                        raise TaskError('Extra cleanup requires explicitly audited reference media or task-local command files')
                    removable.add(file)
                files = [f for f in path.rglob('*') if f.is_file()]
                surviving_hashes = {file_hash(f) for f in files if f not in removable}
                for f in list(removable):
                    if 'inputs' in f.relative_to(path).parts and file_hash(f) not in surviving_hashes:
                        removable.remove(f)
                        surviving_hashes.add(file_hash(f))
                for f in files:
                    artifact_path(path, f.relative_to(path))  # reject symlinks
                excluded = {'README.md', 'feedback.md', 'delivery.json', 'final_output/acceptance.md'}
                retained, removed = {}, {}
                for f in files:
                    name = f.relative_to(path).as_posix()
                    if f in removable:
                        removed[name] = file_hash(f)
                    elif name not in excluded and f.name not in {'.lock', '.task.lock'}:
                        retained[name] = file_hash(f)
                record = {'schema': 1, 'kind': 'verified-completed-video', 'task_id': metadata['id'],
                          'approval': approval, 'retained': retained, 'removed': removed,
                          'note': 'Historical approval checked before cleanup; production requires restored evidence or re-review.'}
                save_json(receipt, record)
            count = size = 0
            for name, expected in record['removed'].items():
                f = artifact_path(path, name)
                if f.exists():
                    if file_hash(f) != expected:
                        raise TaskError('Cleanup candidate changed; stopped')
                    size += f.stat().st_size
                    f.unlink()
                    count += 1
            for folder in sorted((f for f in path.rglob('*') if f.is_dir()), key=lambda f: len(f.parts), reverse=True):
                if not any(folder.iterdir()):
                    folder.rmdir()
            archive_approval(path)
            return {'removed_files': count, 'removed_bytes': size, 'archived': True}
