import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from viz.mirror_viz import make_mirror_export_filename


def test_stage_and_final_exports_have_expected_names(tmp_path: Path):
    assert make_mirror_export_filename(kind="stage", stage_index=1) == "mirror_stage_01.png"
    assert make_mirror_export_filename(kind="stage", stage_index=2) == "mirror_stage_02.png"
    assert make_mirror_export_filename(kind="final") == "mirror_final.png"
    assert (
        make_mirror_export_filename(kind="final", case_index=3, order=[2, 0, 1])
        == "mirror_final_case_03_order_312.png"
    )
