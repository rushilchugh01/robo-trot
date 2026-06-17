import numpy as np
from pathlib import Path

from envs.a1_teacher_env import A1TeacherEnv, build_actor_obs
from robo_trot.a1_constants import OBS_DIM_NO_CONTACTS, OBS_DIM_WITH_CONTACTS, Q_HOME
from robo_trot.kinematics import quat_to_rotmat, rotate_world_to_body


XML_PATH = Path("assets/mujoco_menagerie/unitree_a1/scene.xml")


def test_identity_quaternion_leaves_vector_unchanged():
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    np.testing.assert_allclose(rotate_world_to_body(quat, vec), vec, atol=1e-6)


def test_quaternion_rotation_matrix_shape():
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert quat_to_rotmat(quat).shape == (3, 3)


def test_build_actor_obs_has_56_values_with_contacts():
    obs = build_actor_obs(
        projected_gravity_body=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        base_ang_vel_body=np.zeros(3, dtype=np.float32),
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
        command=np.array([0.3, 0.0, 0.1], dtype=np.float32),
        q_minus_home=np.zeros(12, dtype=np.float32),
        qdot=np.zeros(12, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        phase=0.25,
        previous_reward=0.5,
        reset_flag=True,
        foot_contacts=np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        use_contacts=True,
    )
    assert obs.dtype == np.float32
    assert obs.shape == (OBS_DIM_WITH_CONTACTS,)


def test_build_actor_obs_has_52_values_without_contacts():
    obs = build_actor_obs(
        projected_gravity_body=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        base_ang_vel_body=np.zeros(3, dtype=np.float32),
        base_lin_vel_body=np.zeros(3, dtype=np.float32),
        command=np.array([0.3, 0.0, 0.1], dtype=np.float32),
        q_minus_home=np.zeros(12, dtype=np.float32),
        qdot=np.zeros(12, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        phase=0.25,
        previous_reward=0.5,
        reset_flag=False,
        foot_contacts=np.zeros(4, dtype=np.float32),
        use_contacts=False,
    )
    assert obs.shape == (OBS_DIM_NO_CONTACTS,)


def test_step_q_des_records_post_step_actuator_force_when_assets_exist():
    if not XML_PATH.exists():
        return
    env = A1TeacherEnv(XML_PATH, {"use_contacts": True})
    env.reset(seed=0)

    _reward, _done, info = env.step_q_des(Q_HOME + np.array([0.05, 0.0, 0.0] * 4, dtype=np.float32))

    np.testing.assert_allclose(info["torque"], env.data.actuator_force[:12], rtol=1e-5, atol=1e-5)
