import importlib


def test_core_domains_are_importable_from_robo_trot_package():
    modules = [
        "robo_trot.robot.a1",
        "robo_trot.robot.kinematics",
        "robo_trot.robot.model_info",
        "robo_trot.sim.a1_teacher_env",
        "robo_trot.teachers.base",
        "robo_trot.teachers.footspace_cpg_ik",
        "robo_trot.demos.dataset_writer",
        "robo_trot.demos.record_teacher_demos",
        "robo_trot.demos.sharded_generation",
        "robo_trot.demos.manifest",
        "robo_trot.demos.validation",
        "robo_trot.policies",
        "robo_trot.training",
    ]

    for module in modules:
        assert importlib.import_module(module)


def test_legacy_cli_modules_forward_to_robo_trot_implementations():
    from robo_trot.demos.manifest import build_manifest
    from robo_trot.demos.dataset_writer import DatasetWriter
    from robo_trot.demos.sharded_generation import shards_for_total
    from robo_trot.demos.record_teacher_demos import make_teacher

    assert build_manifest.__module__ == "robo_trot.demos.manifest"
    assert DatasetWriter.__module__ == "robo_trot.demos.dataset_writer"
    assert shards_for_total.__module__ == "robo_trot.demos.sharded_generation"
    assert make_teacher.__module__ == "robo_trot.demos.record_teacher_demos"
