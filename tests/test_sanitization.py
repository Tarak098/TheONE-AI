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


def test_render_exact_pasted_gnani_key():
    raw_pasted = 'GNANI_API_KEY="vach_1ytE2CY5X2PMmjL8sLkFsDrvBgFCTVwnvbJyxN42h1NdYaiCAmZumyXMjeb5bKfqq370bvpqSkPc8PCoujPP4ij6o1hJeCLV_2433e7180d00b9e032a8fa7606c1d7fb"\n'
    s = Settings(GNANI_API_KEY=raw_pasted)
    expected = "vach_1ytE2CY5X2PMmjL8sLkFsDrvBgFCTVwnvbJyxN42h1NdYaiCAmZumyXMjeb5bKfqq370bvpqSkPc8PCoujPP4ij6o1hJeCLV_2433e7180d00b9e032a8fa7606c1d7fb"
    assert s.GNANI_API_KEY == expected
    assert s.prisma_api_key == expected
    assert s.timbre_api_key == expected
    # Ensure no return character or quotes in headers
    assert "\n" not in s.prisma_api_key
    assert "\r" not in s.prisma_api_key
    assert '"' not in s.prisma_api_key


