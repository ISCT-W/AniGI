"""Image settings delegated to the explicit workspace configuration reader."""

import re

from ..config import ConfigError, DEFAULT_ENV_FILE, read_settings as _read_settings
from .task_store import StoreError


DOTENV = DEFAULT_ENV_FILE


def read_settings(names, env_file=DOTENV, environ=None):
    """Read only requested fields, without executing dotenv text."""
    try:
        return _read_settings(names, env_file=env_file, environ=environ)
    except ConfigError as exc:
        raise StoreError(str(exc)) from None


def image_model(explicit=None, env_file=DOTENV):
    if explicit is not None:
        return explicit
    settings = read_settings(["GEMINI_IMAGE_MODEL"], env_file)
    return settings.get("GEMINI_IMAGE_MODEL") or "gemini-2.5-flash-image"


def gpt_image_model(explicit=None, env_file=DOTENV):
    if explicit is not None:
        return explicit
    settings = read_settings(["GPT_IMAGE_MODEL", "OPENAI_IMAGE_MODEL"], env_file)
    model = _alias(settings, "GPT_IMAGE_MODEL", "OPENAI_IMAGE_MODEL")
    if not model:
        raise StoreError("GPT 需要显式型号或 GPT_IMAGE_MODEL / OPENAI_IMAGE_MODEL；不自动选择型号")
    return model


def reference_project(mode, env_file=DOTENV):
    """Bind records to the configured generic reference scope.

    Offline records use synthetic evidence and never read local configuration.
    """
    if mode == "offline":
        return "offline-fixture"
    if mode != "generation":
        raise StoreError("模式须为 offline/generation")
    scope = read_settings(["REFERENCE_SCOPE_ID"], env_file).get("REFERENCE_SCOPE_ID", "")
    if not scope:
        raise StoreError("正式任务需要在本地配置 REFERENCE_SCOPE_ID")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", scope):
        raise StoreError("REFERENCE_SCOPE_ID 格式无效；值未输出")
    return scope


def _alias(settings, primary, alternate):
    first, second = settings.get(primary, ""), settings.get(alternate, "")
    if first and second and first != second:
        raise StoreError(f"配置字段 {primary} 与 {alternate} 冲突；值未输出")
    return first or second


def _key(settings, primary, alternate):
    key = _alias(settings, primary, alternate)
    if not key or not key.strip():
        raise StoreError(f"缺少 {primary} 或 {alternate}；请在本地配置")
    if any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise StoreError("API key 包含无效的 HTTP 头字符；值未输出")
    return key


def gemini_key(env_file=DOTENV):
    settings = read_settings(["GEMINI_API_KEY", "GOOGLE_API_KEY"], env_file)
    return _key(settings, "GEMINI_API_KEY", "GOOGLE_API_KEY")


def gpt_key(env_file=DOTENV):
    settings = read_settings(["GPT_API_KEY", "OPENAI_API_KEY"], env_file)
    return _key(settings, "GPT_API_KEY", "OPENAI_API_KEY")
