"""The README and example notebook may only use functions and parameters that exist.

Every ``eh.<name>`` reference in a Python code block of README.md or a code cell
of examples/quickstart.ipynb must exist in ``exohab``; every keyword argument
passed to it must be a parameter of its signature, and positional arguments
must not exceed the number of positional parameters.
"""

import ast
import inspect
import json
import re
from pathlib import Path

import pytest

import exohab

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
NOTEBOOK = ROOT / "examples" / "quickstart.ipynb"


def _readme_blocks():
    text = README.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def _notebook_blocks():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _check_source(src: str, where: str) -> int:
    """Validate all ``eh.X`` uses in ``src``; return how many calls were checked."""
    tree = ast.parse(src)
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "eh":
            assert hasattr(exohab, node.attr), f"{where}: exohab.{node.attr} does not exist"
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "eh"
        ):
            continue
        assert hasattr(exohab, node.func.attr), f"{where}: exohab.{node.func.attr} does not exist"
        obj = getattr(exohab, node.func.attr)
        if not callable(obj):
            continue
        sig = inspect.signature(obj)
        params = sig.parameters
        has_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
        for kw in node.keywords:
            if kw.arg is not None and not has_var_kw:
                assert kw.arg in params, f"{where}: {node.func.attr}() has no parameter '{kw.arg}'"
                assert params[kw.arg].kind is not inspect.Parameter.POSITIONAL_ONLY
        positional = [p for p in params.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if not any(p.kind is p.VAR_POSITIONAL for p in params.values()):
            assert len(node.args) <= len(positional), (
                f"{where}: too many positional arguments to {node.func.attr}()"
            )
        checked += 1
    return checked


def test_readme_snippets_use_real_api():
    blocks = _readme_blocks()
    assert len(blocks) >= 3  # students, researchers, ML engineers
    total = sum(_check_source(b, f"README block {i}") for i, b in enumerate(blocks))
    assert total >= 15


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebook not present (sdist without examples)")
def test_notebook_uses_real_api_and_is_valid():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    errors = [o for c in nb.cells if c.cell_type == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, "notebook was saved with error outputs"
    total = sum(_check_source(b, f"notebook cell {i}") for i, b in enumerate(_notebook_blocks()))
    assert total >= 10


DOCS = ROOT / "docs"


@pytest.mark.skipif(not DOCS.exists(), reason="docs/ not present (sdist without docs)")
def test_website_snippets_use_real_api():
    pages = sorted(DOCS.rglob("*.md"))
    assert len(pages) >= 10
    total = 0
    for page in pages:
        blocks = re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), flags=re.S)
        for i, b in enumerate(blocks):
            total += _check_source(b, f"{page.relative_to(ROOT)} block {i}")
    assert total >= 40


def test_checker_catches_mistakes():
    with pytest.raises(AssertionError, match="does not exist"):
        _check_source("eh.compute_esi_5p(1, 2)", "t")
    with pytest.raises(AssertionError, match="no parameter"):
        _check_source("eh.compute_esi_2p(radius=1, insolation=1)", "t")
    with pytest.raises(AssertionError, match="too many positional"):
        _check_source("eh.compute_esi_2p(1, 1, 1)", "t")
