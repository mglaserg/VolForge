import os

from volforge.config import get_env, load_project_env, require_env


def test_load_project_env_reads_explicit_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("VOLFORGE_TEST_VALUE=from_dotenv\n")
    monkeypatch.delenv("VOLFORGE_TEST_VALUE", raising=False)

    loaded = load_project_env(env_file)

    assert loaded == env_file
    assert os.getenv("VOLFORGE_TEST_VALUE") == "from_dotenv"


def test_process_environment_wins_over_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("VOLFORGE_TEST_VALUE=from_dotenv\n")
    monkeypatch.setenv("VOLFORGE_TEST_VALUE", "from_process")

    load_project_env(env_file)

    assert os.getenv("VOLFORGE_TEST_VALUE") == "from_process"


def test_get_env_loads_repo_style_dotenv_from_cwd(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("VOLFORGE_CWD_VALUE=hello\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VOLFORGE_CWD_VALUE", raising=False)

    assert get_env("VOLFORGE_CWD_VALUE") == "hello"


def test_require_env_has_clear_error(monkeypatch):
    monkeypatch.delenv("VOLFORGE_MISSING_VALUE", raising=False)
    try:
        require_env("VOLFORGE_MISSING_VALUE", hint="Put it in .env.")
    except RuntimeError as exc:
        assert "VOLFORGE_MISSING_VALUE" in str(exc)
        assert "Put it in .env." in str(exc)
    else:  # pragma: no cover
        raise AssertionError("require_env should fail for a missing value")
