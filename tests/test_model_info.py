from pathlib import Path

import mujoco

from robo_trot.model_info import describe_model, format_indexed_names


def test_format_indexed_names_prints_indices_and_names():
    text = format_indexed_names("body", ["world", "trunk", "FR_foot"])
    assert "body names" in text
    assert "0: world" in text
    assert "2: FR_foot" in text


def test_describe_model_reports_foot_geom_candidates_when_assets_exist():
    xml_path = Path("assets/mujoco_menagerie/unitree_a1/scene.xml")
    if not xml_path.exists():
        return
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    text = describe_model(model)

    assert "likely foot geom candidates:" in text
    assert "body=FR_calf" in text
    assert "body=FL_calf" in text
    assert "body=RR_calf" in text
    assert "body=RL_calf" in text
    assert "detected foot contact geoms:" in text
    assert "body=FR_calf type=mjGEOM_SPHERE" in text
