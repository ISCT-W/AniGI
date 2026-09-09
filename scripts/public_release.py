"""Build a fresh public snapshot from an explicitly reviewed file manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from typing import Iterable


class ReleaseError(ValueError):
    """An unsafe or ambiguous input blocked the export."""


EXCLUDED_PARTS = {
    ".git", ".codex", ".private", ".venv", "__pycache__", ".pytest_cache",
    "generation", "generations", "generated", "runs", "plans", "final_output",
    "node_modules", "public_release", "public-release",
}
EXCLUDED_NAMES = {
    "AGENTS.md", ".DS_Store", "transfer_policy.json", "TRANSFER_POLICY.md",
    "state.json",
}
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt", ".lock"}
ROOT_DOTFILES = {".gitignore"}
BASE_PATTERNS = (
    ("personal_path", r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    ("temporary_path", r"/(?:private/)?var/folders/[A-Za-z0-9]+/"),
    ("private_key", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ("credential", r"\b(?:sk-[A-Za-z0-9_-]{24,}|AIza[A-Za-z0-9_-]{30,}|gh[pousr]_[A-Za-z0-9]{30,})\b"),
    ("email", r"\b[A-Za-z0-9._%+-]+@(?!example\.(?:com|org|invalid)\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("authenticated_url", r"https?://[^\s/@]+:[^\s/@]+@"),
    ("signed_url", r"(?i)[?&](?:x-amz-signature|x-goog-signature|access_token|api_key)=[^\s&#]+"),
    ("resource_identifier", r"\b(?!00000000-0000-4000-8000-000000000000\b)[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"),
)


@dataclass(frozen=True)
class SourceFile:
    relative: str
    data: bytes


def _safe_relative(value: str, number: int) -> PurePosixPath:
    if not value or "\\" in value or any(ord(c) < 32 for c in value):
        raise ReleaseError(f"manifest line {number}: invalid path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(p in {".", ".."} for p in path.parts):
        raise ReleaseError(f"manifest line {number}: path must be canonical and relative")
    if any(c in value for c in "*?[]"):
        raise ReleaseError(f"manifest line {number}: patterns are not allowed")
    if any(part in EXCLUDED_PARTS or part.startswith(".env") for part in path.parts):
        raise ReleaseError(f"manifest line {number}: protected file category")
    if path.name in EXCLUDED_NAMES:
        # Empty template scaffolds are reviewed source, while runtime files are private.
        if not ("templates" in path.parts and path.suffix == ".md"):
            raise ReleaseError(f"manifest line {number}: protected file category")
    if path.suffix not in TEXT_SUFFIXES and value not in ROOT_DOTFILES:
        raise ReleaseError(f"manifest line {number}: unsupported file type")
    return path


def _reject_symlinks(root: Path, relative: PurePosixPath, number: int) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ReleaseError(f"manifest line {number}: symbolic links are not allowed")
    if not current.is_file():
        raise ReleaseError(f"manifest line {number}: missing regular file")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ReleaseError(f"manifest line {number}: path leaves source root")
    return current


def candidates(root: Path, manifest: Path | None = None) -> list[Path]:
    """Return only manifest entries; never enumerate private workspace content."""
    root = Path(root).resolve()
    manifest = Path(manifest) if manifest is not None else root / "public-files.txt"
    if manifest.is_symlink() or not manifest.is_file():
        raise ReleaseError("a regular explicit public manifest is required")
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (UnicodeError, OSError):
        raise ReleaseError("manifest is unreadable UTF-8 text") from None
    result, seen = [], set()
    for number, line in enumerate(lines, 1):
        if not line or line.startswith("#"):
            continue
        relative = _safe_relative(line, number)
        if str(relative) in seen:
            raise ReleaseError(f"manifest line {number}: duplicate path")
        seen.add(str(relative))
        result.append(_reject_symlinks(root, relative, number))
    if not result:
        raise ReleaseError("public manifest contains no files")
    return result


def load_patterns(rules: Path | None = None) -> tuple[tuple[str, re.Pattern[str]], ...]:
    patterns = list(BASE_PATTERNS)
    if rules is not None:
        rules = Path(rules)
        if rules.is_symlink() or not rules.is_file():
            raise ReleaseError("local pattern file must be a regular file")
        try:
            config = json.loads(rules.read_text(encoding="utf-8"))
            if set(config) != {"patterns"} or not isinstance(config["patterns"], list):
                raise ValueError
            for item in config["patterns"]:
                if set(item) != {"category", "regex"} or not re.fullmatch("[a-z_]+", item["category"]):
                    raise ValueError
                patterns.append((item["category"], item["regex"]))
        except (ValueError, TypeError, KeyError, UnicodeError, OSError):
            raise ReleaseError("invalid local pattern configuration") from None
    try:
        return tuple((category, re.compile(pattern)) for category, pattern in patterns)
    except (re.error, TypeError):
        raise ReleaseError("invalid local pattern expression") from None


def _scan_text(text: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> list[tuple[int, list[str]]]:
    return [
        (number, kinds)
        for number, line in enumerate(text.splitlines(), 1)
        if (kinds := sorted({category for category, pattern in patterns if pattern.search(line)}))
    ]


def _text(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
        if any(ord(c) < 32 and c not in "\t\n\r" for c in text):
            raise UnicodeError
        return text
    except UnicodeError:
        raise ReleaseError("candidate contains binary or non-UTF-8 content") from None


def _read_candidate(root: Path, relative: str) -> bytes:
    """Open each component without following links, including during replacement."""
    descriptors = []
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        current = os.open(root, directory_flags)
        descriptors.append(current)
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current)
        descriptors.append(file_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ReleaseError("candidate is not a regular file")
        chunks = []
        while chunk := os.read(file_fd, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError:
        raise ReleaseError("candidate could not be opened safely") from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def findings(paths: Iterable[Path], rules: Path | None = None) -> list[tuple[Path, int, list[str]]]:
    patterns = load_patterns(rules)
    result = []
    for path in paths:
        path = Path(path)
        if path.is_symlink():
            raise ReleaseError("symbolic links are not allowed")
        try:
            text = _text(path.read_bytes())
        except OSError:
            raise ReleaseError("candidate could not be read") from None
        result.extend((path, number, kinds) for number, kinds in _scan_text(text, patterns))
        names = {category for category, pattern in patterns if pattern.search(path.name)}
        if names:
            result.append((path, 0, sorted(names | {"filename"})))
    return result


def snapshot(root: Path, manifest: Path | None = None, rules: Path | None = None) -> tuple[list[SourceFile], list[tuple[str, int, list[str]]]]:
    """Read once so the exported bytes are the same bytes that were scanned."""
    root = Path(root).resolve()
    patterns = load_patterns(rules)
    files, issues = [], []
    for index, path in enumerate(candidates(root, manifest), 1):
        relative = path.relative_to(root).as_posix()
        try:
            data = _read_candidate(root, relative)
        except ReleaseError:
            raise ReleaseError(f"candidate {index}: cannot read file safely") from None
        try:
            text = _text(data)
        except ReleaseError:
            raise ReleaseError(f"candidate {index}: binary or non-UTF-8 content") from None
        name_kinds = {category for category, pattern in patterns if pattern.search(relative)}
        location = f"candidate-{index}" if name_kinds else relative
        if name_kinds:
            issues.append((location, 0, sorted(name_kinds | {"filename"})))
        issues.extend((location, line, kinds) for line, kinds in _scan_text(text, patterns))
        files.append(SourceFile(relative, data))
    return files, issues


def export(root: Path, output: Path, manifest: Path | None = None, rules: Path | None = None) -> int:
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ReleaseError("output must be a new directory")
    if any(parent.is_symlink() for parent in [output.parent, *output.parents]):
        raise ReleaseError("output parents must not be symbolic links")
    files, issues = snapshot(root, manifest, rules)
    if issues:
        raise ReleaseError("content findings block export; run the scan for locations")
    output.mkdir(parents=True, exist_ok=False)
    try:
        for source in files:
            target = output / source.relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(source.data)
        patterns = load_patterns(rules)
        for source in files:
            target = output / source.relative
            if target.is_symlink() or target.read_bytes() != source.data or _scan_text(_text(target.read_bytes()), patterns):
                raise ReleaseError("export verification failed")
    except Exception:
        shutil.rmtree(output)
        raise
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    rules = args.rules
    if rules is None and (root / ".private/publication-rules.json").exists():
        rules = root / ".private/publication-rules.json"
    try:
        files, issues = snapshot(root, args.manifest, rules)
        for path, line, kinds in issues:
            print(f"{path}:{line}: {','.join(kinds)}")
        if issues:
            raise ReleaseError("review the listed locations; matched values are hidden")
        if args.output:
            count = export(root, args.output, args.manifest, rules)
            print(f"Exported and verified {count} explicitly listed files.")
        else:
            print(f"Scanned {len(files)} explicitly listed files; no pattern findings.")
        print("Local semantic patterns loaded." if rules else "Generic patterns only; local semantic patterns unavailable.")
        print("Independent semantic review and Git history checks remain necessary before publication.")
        return 0
    except ReleaseError as exc:
        print(f"Release blocked: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
