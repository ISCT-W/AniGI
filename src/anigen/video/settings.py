"""Explicit provider-field reads; importing this module never loads configuration."""
from ..config import read_settings


def get_setting(name):
    if name not in {'FAL_KEY', 'GEMINI_API_KEY', 'OPENAI_API_KEY', 'OPENAI_IMAGE_MODEL', 'GEMINI_IMAGE_MODEL', 'IMAGE_BACKEND'}:
        raise ValueError('Unsupported provider configuration field')
    return read_settings({name}).get(name, '')
