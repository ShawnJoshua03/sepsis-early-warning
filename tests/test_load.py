import pytest

from load import load_psv_dir


def test_unlabeled_file_fails_the_load(tmp_path):
    (tmp_path / "p1.psv").write_text("HR|ICULOS|SepsisLabel\n80|1|0\n")
    (tmp_path / "p2.psv").write_text("HR|ICULOS\n80|1\n")
    with pytest.raises(ValueError, match="1 of 2 files"):
        load_psv_dir(str(tmp_path))
