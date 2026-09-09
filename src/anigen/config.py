"""Read selected configuration without import-time IO or exporting secrets."""

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re
import shlex


class ConfigError(ValueError):
    """A configuration problem described without its value."""


DEFAULT_ENV_FILE = object()
_workspace = ContextVar("anigen_workspace", default=None)
_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
_ALIASES = {
    "GPT_API_KEY": ["OPENAI_API_KEY"],
    "OPENAI_API_KEY": ["GPT_API_KEY"],
    "GPT_IMAGE_MODEL": ["OPENAI_IMAGE_MODEL"],
    "OPENAI_IMAGE_MODEL": ["GPT_IMAGE_MODEL"],
    "GEMINI_API_KEY": ["GOOGLE_API_KEY"],
    "GOOGLE_API_KEY": ["GEMINI_API_KEY"],
}


def workspace_path(workspace=None, environ=None):
    environment = os.environ if environ is None else environ
    value = workspace if workspace is not None else _workspace.get()
    if value is None:
        value = environment.get("ANIGEN_WORKSPACE")
    return Path(value).expanduser().resolve() if value else None


@contextmanager
def use_workspace(workspace):
    """Scope nested library calls to one workspace, including concurrent callers."""
    token = _workspace.set(Path(workspace).expanduser().resolve())
    try:
        yield _workspace.get()
    finally:
        _workspace.reset(token)


def _alias_map(workspace, extra=None):
    result = {name: list(rows) for name, rows in _ALIASES.items()}
    sources = []
    if workspace is not None:
        path = workspace / ".private" / "config-aliases.json"
        if path.is_file():
            try:
                sources.append(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                raise ConfigError("Invalid local configuration alias document") from None
    if extra is not None:
        sources.append(extra)
    for source in sources:
        if not isinstance(source, dict):
            raise ConfigError("Configuration aliases must be an object of field-name lists")
        for name, rows in source.items():
            if (not isinstance(name, str) or not _NAME.fullmatch(name)
                    or not isinstance(rows, list)
                    or any(not isinstance(item, str) or not _NAME.fullmatch(item) for item in rows)):
                raise ConfigError("Configuration aliases contain an invalid field name")
            result[name] = list(dict.fromkeys([*result.get(name, []), *rows]))
    return result


def read_settings(names, env_file=DEFAULT_ENV_FILE, environ=None, workspace=None, aliases=None):
    """Resolve named fields and reject disagreeing aliases without displaying values.

    An explicit process value overrides the file value for that same field.
    Conflicting aliases are rejected, including aliases split across the file and
    process. Passing env_file=None disables file IO, including local alias IO.
    No shell expansion, interpolation, or executable dotenv syntax is supported.
    """
    names = list(names)
    if any(not isinstance(name, str) or not _NAME.fullmatch(name) for name in names):
        raise ConfigError("Invalid requested configuration field")
    environment = os.environ if environ is None else environ
    base = workspace_path(workspace, environment)
    mapping = _alias_map(base if env_file is not None else None, aliases)
    wanted = set(names)
    for name in names:
        wanted.update(mapping.get(name, []))
    if env_file is DEFAULT_ENV_FILE:
        path = base / ".env" if base is not None else None
    else:
        path = Path(env_file) if env_file is not None else None
    values = {}
    if path is not None and path.is_file():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            raise ConfigError("Local configuration file is not readable UTF-8") from None
        for number, line in enumerate(lines, 1):
            match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$", line)
            if not match or match[1] not in wanted or match[1] in environment:
                continue
            try:
                parts = shlex.split(match[2], comments=True, posix=True)
            except ValueError:
                raise ConfigError(f"Invalid quoting for {match[1]} at line {number}; value withheld") from None
            if len(parts) > 1:
                raise ConfigError(f"Quote the value for {match[1]} at line {number}; value withheld")
            value = parts[0] if parts else ""
            if match[1] in values and values[match[1]] != value:
                raise ConfigError(f"Conflicting repeated configuration field {match[1]}; values withheld")
            values[match[1]] = value
    values.update({name: environment[name] for name in wanted if name in environment})
    resolved = {}
    for name in names:
        candidates = [name, *mapping.get(name, [])]
        present = {field: values[field] for field in candidates if values.get(field, "").strip()}
        if len(set(present.values())) > 1:
            raise ConfigError(f"Conflicting aliases for {name}: {', '.join(present)}; values withheld")
        if present:
            resolved[name] = next(iter(present.values()))
    return resolved
