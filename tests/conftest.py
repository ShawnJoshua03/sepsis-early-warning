import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from load import load_psv_dir  # noqa: E402


@pytest.fixture(scope="session")
def sample_long_df():
    sample_dir = ROOT / "data" / "sample_psv"
    if not any(sample_dir.glob("*.psv")):  # regenerate the seeded synthetic sample
        sys.path.insert(0, str(ROOT / "data"))
        from make_sample import make_sample
        make_sample(out_dir=str(sample_dir))
    return load_psv_dir(str(sample_dir))
