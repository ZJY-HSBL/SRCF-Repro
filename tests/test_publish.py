from pathlib import Path
import pytest
from srcf_repro.publish import _rows, _table


def test_report_does_not_allow_missing_baselines():
    with pytest.raises(ValueError, match="missing model"):
        _table([{"mode": "sr", "k": "10", "mae": "1"}], "k")


def test_report_rejects_empty_data(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("mode,k,mae,coverage,n_test\n")
    with pytest.raises(ValueError, match="empty measured"):
        _rows(p)
