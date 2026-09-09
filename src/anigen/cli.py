"""One entry point for explicitly authorized, locally recorded production."""

import argparse
import json
from pathlib import Path
import sys

from .config import ConfigError, use_workspace
from . import delivery
from .workspace import TaskError, artifact_path, create_task, load_task, refresh_index, task_lock


def _text(path):
    file = Path(path)
    if file.suffix.lower() != ".md":
        raise TaskError("Text inputs must be Markdown files, never credential files")
    return file.read_text(encoding="utf-8")


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--workspace", type=Path, default=Path.cwd(), help="Root containing local configuration and generation/")
    sub = command.add_subparsers(dest="command", required=True)
    new = sub.add_parser("new", help="Create a named image or video task; no service calls")
    new.add_argument("purpose", choices=["image", "video"])
    new.add_argument("--description", required=True)
    new.add_argument("--backend", choices=["gpt", "gemini"])
    new.add_argument("--brief-file", type=Path, required=True)
    new.add_argument("--mode", choices=["offline", "generation"], default="offline")
    new.add_argument("--authorization-file", type=Path)
    for name in ("image", "video"):
        action = sub.add_parser(name, help=f"Run one {name} workflow action within an existing task")
        action.add_argument("action")
        action.add_argument("--task", type=Path, required=True)
        if name == "video":
            action.add_argument("--input", type=Path)
            action.add_argument("--live", action="store_true")
    compact = sub.add_parser("compact", help="Remove caches from a completed video; retain historical approval, disable production")
    compact.add_argument("--task", type=Path, required=True)
    status = sub.add_parser("status", help="Refresh the local task index without generating")
    status.add_argument("--task", type=Path, required=True)
    output = sub.add_parser("deliver", help="Export the exact accepted final image or full film")
    output.add_argument("--task", type=Path, required=True)
    output.add_argument("--round")
    output.add_argument("--candidate")
    accept = sub.add_parser("accept", help="Record the user's opinion of an exported version")
    accept.add_argument("--task", type=Path, required=True)
    accept.add_argument("--version", required=True)
    accept.add_argument("--status", required=True, choices=["accepted", "changes_requested"])
    accept.add_argument("--comment-file", type=Path, required=True)
    sub.add_parser("backends", help="Show supported image backends without loading credentials")
    return command


def video_action(task, action, data=None, *, live=False, **providers):
    from .video import runtime
    from .storage import require_active
    require_active(task)
    with task_lock(task) as (path, root, record), use_workspace(root):
        if record["purpose"] != "video":
            raise TaskError("Video actions require a video task")
        if live and record["mode"] != "generation" and not providers:
            raise TaskError("An offline task cannot call live services")
        data = {} if data is None else dict(data)
        if action == "init":
            supplied = data.get("reference_scope_id")
            if supplied != record["reference_scope_id"]:
                raise TaskError("Video plan reference scope must match the task's frozen scope")
            if data.get("task_id", record["id"]) != record["id"]:
                raise TaskError("Video plan belongs to another task")
            data["task_id"] = record["id"]
        if action == "keyframe-prepare":
            data.setdefault("backend", record["initial_backend"])
        runs = artifact_path(path, "video")
        runs.mkdir(exist_ok=True)
        state = runtime.dispatch(action, record["video_run_id"], data, live=live, runs=runs, **providers)
        refresh_index(path)
        return state


def image_action(task, action, arguments):
    from .image import cli as image_cli
    with task_lock(task) as (path, root, record), use_workspace(root):
        if record["purpose"] != "image":
            raise TaskError("Use video keyframe actions for a video's shared image stage")
        if action in {"init", "promote", "accept"}:
            raise TaskError("Use new, deliver or accept at the task entry point")
        # The common store path is controlled here; caller options cannot replace it.
        result = image_cli.main([action, str(artifact_path(path, "image")), *arguments])
        refresh_index(path)
        return result


def main(argv=None):
    args, extra = parser().parse_known_args(argv)
    try:
        if extra and args.command != "image":
            raise TaskError("Unrecognized arguments; see --help")
        if args.command == "new":
            path = create_task(args.workspace, args.purpose, args.description, args.backend,
                               _text(args.brief_file), args.mode,
                               _text(args.authorization_file) if args.authorization_file else "")
            print(path)
        elif args.command == "backends":
            from .image.backends import BACKENDS
            for name, spec in BACKENDS.items():
                if spec.factory:
                    print(name)
        elif args.command == "image":
            return image_action(args.task, args.action, extra)
        elif args.command == "compact":
            from .storage import compact
            print(json.dumps(compact(args.task)))
        elif args.command == "video":
            if args.input and str(args.input) != "-" and args.input.suffix.lower() != ".json":
                raise TaskError("Video action input must be JSON, never a credential file")
            data = json.loads(sys.stdin.read() if str(args.input) == "-" else args.input.read_text(encoding="utf-8")) if args.input else {}
            if not isinstance(data, dict):
                raise TaskError("Video action input must be an object")
            state = video_action(args.task, args.action, data, live=args.live)
            print(json.dumps({"action": args.action, "run_id": state.get("id"),
                              "video_attempts": len(state.get("attempts", [])),
                              "index": str(Path(args.task).resolve() / "README.md")}, ensure_ascii=False))
        elif args.command == "status":
            with task_lock(args.task):
                print(refresh_index(args.task))
        elif args.command == "deliver":
            print(delivery.export(args.task, args.round, args.candidate))
        elif args.command == "accept":
            delivery.accept(args.task, args.version, args.status, _text(args.comment_file))
            print("User verdict recorded for the selected version.")
        return 0
    except (TaskError, ConfigError, ValueError, OSError) as exc:
        # Provider errors are sanitized at their boundaries. Do not echo file bodies.
        message = str(exc) if isinstance(exc, (TaskError, ConfigError)) else type(exc).__name__
        print(f"Operation not completed: {message}. Existing request state must be reconciled before retrying.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
