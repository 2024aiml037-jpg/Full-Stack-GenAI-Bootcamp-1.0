import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

CALCULATOR_PATH = REPO_ROOT / "Class-02-28-Mar-system-setup-genai-intro-by-sunny" / "test.py"
LLM_APP_PATH = REPO_ROOT / "Class-16-17-16&17-May-2026-LLM-Acess-Via-APIs" / "app.py"
MAIN_PATH = REPO_ROOT / "main.py"


def load_module(path: Path, name: str) -> ModuleType:
    """Import a module from an arbitrary path under an explicit module name.

    The bootcamp sources live in directories that are not importable packages
    (spaces, dashes, ampersands), so they are loaded by file location.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def calculator_module() -> ModuleType:
    return load_module(CALCULATOR_PATH, "bootcamp_calculator")


@pytest.fixture(scope="session")
def llm_app() -> ModuleType:
    pytest.importorskip("streamlit")
    pytest.importorskip("langchain_core")
    return load_module(LLM_APP_PATH, "bootcamp_llm_app")


@pytest.fixture(scope="session")
def main_module() -> ModuleType:
    return load_module(MAIN_PATH, "bootcamp_main")
