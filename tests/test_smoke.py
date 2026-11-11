import importlib

import pytest


@pytest.mark.parametrize("package", ["service", "llm", "agent", "eval"])
def test_packages_import(package: str) -> None:
    assert importlib.import_module(package)
