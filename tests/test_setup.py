"""Tests for the OpenMem setup wizard."""

from __future__ import annotations

import json
import os
import stat
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openmem.mcp.setup import (
    _ask,
    _ask_choice,
    _ask_confirm,
    _ask_secret,
    _load_existing_config,
    _mask_key,
    _save_config,
    _validate_openai_embedding,
    _validate_ollama,
)
import openmem.mcp.providers as providers_module


# --- Helper tests ---


class TestAsk:
    def test_returns_user_input(self):
        with patch("builtins.input", return_value="my_value"):
            assert _ask("prompt") == "my_value"

    def test_returns_default_on_empty(self):
        with patch("builtins.input", return_value=""):
            assert _ask("prompt", default="/tmp/test.db") == "/tmp/test.db"

    def test_strips_whitespace(self):
        with patch("builtins.input", return_value="  spaced  "):
            assert _ask("prompt") == "spaced"


class TestAskSecret:
    def test_returns_user_input(self):
        with patch("getpass.getpass", return_value="secret123"):
            assert _ask_secret("key") == "secret123"

    def test_returns_default_on_empty(self):
        with patch("getpass.getpass", return_value=""):
            assert _ask_secret("key", default="existing") == "existing"


class TestAskChoice:
    def test_returns_default_on_empty(self):
        with patch("builtins.input", return_value=""):
            assert _ask_choice("pick", ["a", "b", "c"], default=2) == 2

    def test_returns_user_choice(self):
        with patch("builtins.input", return_value="3"):
            assert _ask_choice("pick", ["a", "b", "c"]) == 3

    def test_retries_on_invalid_then_succeeds(self):
        inputs = iter(["invalid", "5", "2"])
        with patch("builtins.input", side_effect=inputs):
            assert _ask_choice("pick", ["a", "b", "c"]) == 2


class TestAskConfirm:
    def test_default_yes(self):
        with patch("builtins.input", return_value=""):
            assert _ask_confirm("ok?", default=True) is True

    def test_default_no(self):
        with patch("builtins.input", return_value=""):
            assert _ask_confirm("ok?", default=False) is False

    def test_explicit_yes(self):
        with patch("builtins.input", return_value="y"):
            assert _ask_confirm("ok?", default=False) is True

    def test_explicit_no(self):
        with patch("builtins.input", return_value="n"):
            assert _ask_confirm("ok?", default=True) is False


class TestMaskKey:
    def test_short_key(self):
        assert _mask_key("abc") == "****"

    def test_long_key(self):
        assert _mask_key("sk-1234567890abcdef") == "sk-12345..."


# --- Config file I/O ---


class TestSaveConfig:
    def test_writes_config(self, tmp_path):
        config = {
            "OPENMEM_STORAGE_PATH": "~/.openmem/memory.db",
            "OPENMEM_EMBEDDING_PROVIDER": "openai",
            "OPENMEM_EMBEDDING_API_KEY": "sk-test123",
        }
        with patch("openmem.mcp.setup.CONFIG_DIR", tmp_path), \
             patch("openmem.mcp.setup.CONFIG_FILE", tmp_path / "config.env"):
            path = _save_config(config)

        content = path.read_text()
        assert "OPENMEM_STORAGE_PATH=~/.openmem/memory.db" in content
        assert "OPENMEM_EMBEDDING_PROVIDER=openai" in content
        assert "OPENMEM_EMBEDDING_API_KEY=sk-test123" in content

    def test_skips_empty_values(self, tmp_path):
        config = {"KEY_A": "value", "KEY_B": ""}
        with patch("openmem.mcp.setup.CONFIG_DIR", tmp_path), \
             patch("openmem.mcp.setup.CONFIG_FILE", tmp_path / "config.env"):
            path = _save_config(config)

        content = path.read_text()
        assert "KEY_A=value" in content
        assert "KEY_B" not in content

    def test_has_header_comment(self, tmp_path):
        with patch("openmem.mcp.setup.CONFIG_DIR", tmp_path), \
             patch("openmem.mcp.setup.CONFIG_FILE", tmp_path / "config.env"):
            path = _save_config({"KEY": "val"})

        content = path.read_text()
        assert content.startswith("# OpenMem configuration")

    def test_file_permissions(self, tmp_path):
        with patch("openmem.mcp.setup.CONFIG_DIR", tmp_path), \
             patch("openmem.mcp.setup.CONFIG_FILE", tmp_path / "config.env"):
            path = _save_config({"KEY": "val"})

        mode = path.stat().st_mode
        assert mode & stat.S_IRUSR  # owner can read
        assert mode & stat.S_IWUSR  # owner can write
        assert not (mode & stat.S_IRGRP)  # group cannot read
        assert not (mode & stat.S_IROTH)  # others cannot read


class TestLoadExistingConfig:
    def test_loads_config(self, tmp_path):
        config_file = tmp_path / "config.env"
        config_file.write_text(textwrap.dedent("""\
            # Comment line
            OPENMEM_STORAGE_PATH=~/test.db
            OPENMEM_EMBEDDING_PROVIDER=openai
        """))
        with patch("openmem.mcp.setup.CONFIG_FILE", config_file):
            result = _load_existing_config()

        assert result["OPENMEM_STORAGE_PATH"] == "~/test.db"
        assert result["OPENMEM_EMBEDDING_PROVIDER"] == "openai"

    def test_skips_comments_and_blanks(self, tmp_path):
        config_file = tmp_path / "config.env"
        config_file.write_text("# comment\n\nKEY=val\n")
        with patch("openmem.mcp.setup.CONFIG_FILE", config_file):
            result = _load_existing_config()

        assert result == {"KEY": "val"}

    def test_missing_file(self, tmp_path):
        with patch("openmem.mcp.setup.CONFIG_FILE", tmp_path / "nonexistent"):
            result = _load_existing_config()

        assert result == {}

    def test_malformed_lines(self, tmp_path):
        config_file = tmp_path / "config.env"
        config_file.write_text("no_equals_here\nGOOD=value\n")
        with patch("openmem.mcp.setup.CONFIG_FILE", config_file):
            result = _load_existing_config()

        assert result == {"GOOD": "value"}


# --- Validation ---


class TestValidateOpenAI:
    def test_success(self):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            ok, msg = _validate_openai_embedding("key", "http://localhost/v1", "model")

        assert ok is True
        assert "3-dimensional" in msg

    def test_http_error(self):
        import urllib.error
        error = urllib.error.HTTPError(
            "http://test", 401, "Unauthorized", {}, None
        )
        with patch("urllib.request.urlopen", side_effect=error):
            ok, msg = _validate_openai_embedding("bad-key", "http://localhost/v1", "model")

        assert ok is False
        assert "401" in msg

    def test_connection_error(self):
        import urllib.error
        error = urllib.error.URLError("Connection refused")
        with patch("urllib.request.urlopen", side_effect=error):
            ok, msg = _validate_openai_embedding("key", "http://nowhere", "model")

        assert ok is False
        assert "Connection failed" in msg


class TestValidateOllama:
    def test_success(self):
        mock_response = MagicMock()
        mock_response.read.return_value = b"Ollama is running"
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            ok, msg = _validate_ollama("http://localhost:11434")

        assert ok is True
        assert "running" in msg

    def test_connection_refused(self):
        import urllib.error
        error = urllib.error.URLError("Connection refused")
        with patch("urllib.request.urlopen", side_effect=error):
            ok, msg = _validate_ollama("http://localhost:11434")

        assert ok is False
        assert "Cannot connect" in msg


# --- Config loading in providers.py ---


class TestLoadConfigEnv:
    def test_loads_values_when_env_absent(self, tmp_path):
        config_file = tmp_path / "config.env"
        config_file.write_text("TEST_SETUP_KEY=from_file\n")

        # Reset the loader state
        providers_module._config_loaded = False
        # Remove from env if present
        os.environ.pop("TEST_SETUP_KEY", None)

        with patch("openmem.mcp.providers.Path.home", return_value=tmp_path.parent):
            # The config path is Path.home() / ".openmem" / "config.env"
            # We need to match that structure
            pass

        # Simpler: directly test the function with a patched path
        providers_module._config_loaded = False
        config_dir = tmp_path / ".openmem"
        config_dir.mkdir()
        config_env = config_dir / "config.env"
        config_env.write_text("TEST_OPENMEM_LOAD=loaded_value\n")
        os.environ.pop("TEST_OPENMEM_LOAD", None)

        with patch("pathlib.Path.home", return_value=tmp_path):
            providers_module._config_loaded = False
            providers_module.load_config_env()

        assert os.environ.get("TEST_OPENMEM_LOAD") == "loaded_value"

        # Cleanup
        del os.environ["TEST_OPENMEM_LOAD"]

    def test_does_not_override_existing_env(self, tmp_path):
        config_dir = tmp_path / ".openmem"
        config_dir.mkdir()
        config_env = config_dir / "config.env"
        config_env.write_text("TEST_OPENMEM_OVERRIDE=from_file\n")
        os.environ["TEST_OPENMEM_OVERRIDE"] = "from_env"

        with patch("pathlib.Path.home", return_value=tmp_path):
            providers_module._config_loaded = False
            providers_module.load_config_env()

        assert os.environ["TEST_OPENMEM_OVERRIDE"] == "from_env"

        # Cleanup
        del os.environ["TEST_OPENMEM_OVERRIDE"]

    def test_handles_missing_file(self, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            providers_module._config_loaded = False
            providers_module.load_config_env()  # should not raise

    def test_idempotent(self, tmp_path):
        config_dir = tmp_path / ".openmem"
        config_dir.mkdir()
        config_env = config_dir / "config.env"
        config_env.write_text("TEST_OPENMEM_IDEM=value1\n")
        os.environ.pop("TEST_OPENMEM_IDEM", None)

        with patch("pathlib.Path.home", return_value=tmp_path):
            providers_module._config_loaded = False
            providers_module.load_config_env()
            assert os.environ.get("TEST_OPENMEM_IDEM") == "value1"

            # Change file and call again — should NOT reload
            config_env.write_text("TEST_OPENMEM_IDEM=value2\n")
            os.environ.pop("TEST_OPENMEM_IDEM", None)
            providers_module.load_config_env()
            # Still not set because loader didn't run again
            assert os.environ.get("TEST_OPENMEM_IDEM") is None


# --- Full wizard integration ---


class TestWizardIntegration:
    def test_openai_flow(self, tmp_path):
        """Full OpenAI setup flow with mocked input and HTTP."""
        from openmem.mcp.setup import _run_wizard

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "data": [{"embedding": [0.1] * 256}]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        inputs = iter([
            "~/.openmem/test.db",  # storage path
            "1",                   # OpenAI
            "",                    # model default
            "",                    # dimensions default
            "y",                   # overwrite
        ])

        with patch("builtins.input", side_effect=inputs), \
             patch("getpass.getpass", return_value="sk-test-key-1234"), \
             patch("urllib.request.urlopen", return_value=mock_response), \
             patch("openmem.mcp.setup.CONFIG_DIR", tmp_path), \
             patch("openmem.mcp.setup.CONFIG_FILE", tmp_path / "config.env"):
            _run_wizard()

        config = (tmp_path / "config.env").read_text()
        assert "OPENMEM_STORAGE_PATH=~/.openmem/test.db" in config
        assert "OPENMEM_EMBEDDING_PROVIDER=openai" in config
        assert "OPENMEM_EMBEDDING_API_KEY=sk-test-key-1234" in config

    def test_none_flow(self, tmp_path):
        """Setup with no embeddings."""
        from openmem.mcp.setup import _run_wizard

        inputs = iter([
            "",   # storage path default
            "4",  # None
            "",   # confirm default (yes)
            "y",  # overwrite
        ])

        with patch("builtins.input", side_effect=inputs), \
             patch("openmem.mcp.setup.CONFIG_DIR", tmp_path), \
             patch("openmem.mcp.setup.CONFIG_FILE", tmp_path / "config.env"):
            _run_wizard()

        config = (tmp_path / "config.env").read_text()
        assert "OPENMEM_EMBEDDING_PROVIDER=none" in config

    def test_keyboard_interrupt(self):
        """Ctrl+C exits cleanly."""
        from openmem.mcp.setup import main

        with patch("builtins.input", side_effect=KeyboardInterrupt), \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
