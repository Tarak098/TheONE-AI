"""Tests for API credential sanitization."""

import pytest
from backend.config import Settings


def test_gnani_key_with_name_and_quotes_and_newline():
    s = Settings(GNANI_API_KEY='GNANI_API_KEY="vach_test123"\n')
    assert s.GNANI_API_KEY == "vach_test123"
    assert s.prisma_api_key == "vach_test123"
    assert s.timbre_api_key == "vach_test123"


def test_gemini_key_with_quotes():
    s = Settings(GEMINI_API_KEY='"AIzaSyTestKey123"')
    assert s.GEMINI_API_KEY == "AIzaSyTestKey123"


def test_key_with_whitespace():
    s = Settings(GNANI_API_KEY="   vach_clean_key \r\n\t")
    assert s.GNANI_API_KEY == "vach_clean_key"

