"""The package version must agree between the installed metadata and __version__.

This guards against the version drifting between pyproject.toml and
cavsqueeze/__init__.py, which are maintained by hand.
"""
from importlib.metadata import version

import cavsqueeze


def test_version_consistent():
    assert cavsqueeze.__version__ == version("cavsqueeze")


def test_version_matches_citation_file():
    import pathlib
    cff = pathlib.Path(__file__).resolve().parents[1] / "CITATION.cff"
    if not cff.exists():  # installed from a wheel, nothing to check
        return
    lines = [l for l in cff.read_text().splitlines() if l.startswith("version:")]
    assert lines == [f"version: {cavsqueeze.__version__}"]
