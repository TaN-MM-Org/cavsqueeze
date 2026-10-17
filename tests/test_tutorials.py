"""1.18: each tutorial in examples/ runs, and prints the numbers quoted in
its docstring (and in the README) to the printed precision."""
import pathlib
import re
import subprocess
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
NUMBER = re.compile(r"[-+]?\d+\.\d+(?:e[-+]?\d+)?|[-+]?\d+(?:e[-+]?\d+)?")


def _quoted_output(path):
    """The block after 'Printed output' in the module docstring."""
    doc = path.read_text().split('"""')[1]
    block = doc.split("Printed output")[1].split("\n", 1)[1]
    return [line.strip() for line in block.strip().splitlines()]


def _numbers(line):
    return NUMBER.findall(line)


@pytest.mark.parametrize("name", ["tutorial_decays_to_model.py",
                                  "tutorial_tomography_to_clock.py"])
def test_tutorial_prints_its_quoted_numbers(name):
    path = EXAMPLES / name
    if not path.exists():           # installed from a wheel: nothing to check
        pytest.skip("examples/ is not part of the installed package")
    run = subprocess.run([sys.executable, str(path)], capture_output=True, text=True,
                         timeout=300, check=True)
    got = [line.strip() for line in run.stdout.strip().splitlines()]
    want = _quoted_output(path)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        # same words; each number equal to within one unit of its last digit
        assert NUMBER.sub("#", g) == NUMBER.sub("#", w), (g, w)
        for a, b in zip(_numbers(g), _numbers(w)):
            mant = b.split("e")[0]
            last = 10.0 ** -(len(mant.split(".")[1]) if "." in mant else 0)
            scale = 10.0 ** int(b.split("e")[1]) if "e" in b else 1.0
            assert abs(float(a) - float(b)) <= 1.0001 * last * scale, (g, w)
