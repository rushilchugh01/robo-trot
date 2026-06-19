from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco

FOOT_TOKENS = ("foot", "toe", "ankle")
LEG_TOKENS = ("fr", "fl", "rr", "rl")


@dataclass(frozen=True)
class ActuatorJointMap:
    """Mapping from one MuJoCo actuator to its driven joint addresses."""

    actuator_id: int
    actuator_name: str
    joint_id: int
    joint_name: str
    qposadr: int
    dofadr: int


def mj_names(model: mujoco.MjModel, obj_type: int, count: int) -> list[str]:
    """Return MuJoCo object names, substituting placeholders for unnamed objects."""
    names: list[str] = []
    for idx in range(count):
        name = mujoco.mj_id2name(model, obj_type, idx)
        names.append(name if name is not None else f"<unnamed_{idx}>")
    return names


def format_indexed_names(label: str, names: Iterable[str]) -> str:
    """Format an indexed list of MuJoCo names for inspection output."""
    lines = [f"{label} names:"]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    return "\n".join(lines)


def actuator_joint_maps(model: mujoco.MjModel) -> list[ActuatorJointMap]:
    """Return actuator-to-joint mappings in MuJoCo actuator order."""
    maps: list[ActuatorJointMap] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        maps.append(
            ActuatorJointMap(
                actuator_id=actuator_id,
                actuator_name=actuator_name or f"<unnamed_actuator_{actuator_id}>",
                joint_id=joint_id,
                joint_name=joint_name or f"<unnamed_joint_{joint_id}>",
                qposadr=int(model.jnt_qposadr[joint_id]),
                dofadr=int(model.jnt_dofadr[joint_id]),
            )
        )
    return maps


def likely_foot_names(names: Iterable[str]) -> list[str]:
    """Filter names that likely refer to A1 feet or lower legs."""
    out: list[str] = []
    for name in names:
        lower = name.lower()
        if any(token in lower for token in FOOT_TOKENS):
            out.append(name)
        elif any(token in lower for token in LEG_TOKENS) and ("calf" in lower or "lower" in lower):
            out.append(name)
    return out


def likely_foot_geom_candidates(model: mujoco.MjModel) -> list[str]:
    """Return formatted candidate foot geoms based on calf body membership."""
    candidates: list[str] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"<unnamed_body_{body_id}>"
        lower_body = body_name.lower()
        if not any(token in lower_body for token in LEG_TOKENS) or "calf" not in lower_body:
            continue
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"<unnamed_geom_{geom_id}>"
        geom_type = mujoco.mjtGeom(int(model.geom_type[geom_id])).name
        candidates.append(f"geom_id={geom_id} name={geom_name} body={body_name} type={geom_type}")
    return candidates


def detected_foot_contact_geoms(model: mujoco.MjModel) -> list[str]:
    """Return likely spherical foot contact geoms from candidate foot geoms."""
    contacts: list[str] = []
    for candidate in likely_foot_geom_candidates(model):
        if "type=mjGEOM_SPHERE" in candidate:
            contacts.append(candidate)
    return contacts


def format_indexed_block(title: str, values: Iterable[str]) -> str:
    """Format a titled indexed block of arbitrary string values."""
    lines = [f"{title}:"]
    lines.extend(f"  {idx}: {value}" for idx, value in enumerate(values))
    return "\n".join(lines)


def describe_model(model: mujoco.MjModel) -> str:
    """Return a human-readable MuJoCo model inspection report."""
    actuator_names = mj_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    joint_names = mj_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    body_names = mj_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    site_names = mj_names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    geom_names = mj_names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)

    lines = [
        format_indexed_names("actuator", actuator_names),
        "",
        "actuator -> joint mapping:",
    ]
    for item in actuator_joint_maps(model):
        lines.append(
            f"  {item.actuator_id}: {item.actuator_name} -> "
            f"{item.joint_name} (joint_id={item.joint_id}, qposadr={item.qposadr}, dofadr={item.dofadr})"
        )
    lines.extend(
        [
            "",
            "joint qpos/qvel addresses:",
        ]
    )
    for joint_id, name in enumerate(joint_names):
        lines.append(
            f"  {joint_id}: {name} qposadr={int(model.jnt_qposadr[joint_id])} "
            f"dofadr={int(model.jnt_dofadr[joint_id])}"
        )
    lines.extend(
        [
            "",
            format_indexed_names("body", body_names),
            "",
            format_indexed_names("site", site_names),
            "",
            format_indexed_names("geom", geom_names),
            "",
            format_indexed_names("likely foot body/site/geom", sorted(set(
                likely_foot_names(body_names) + likely_foot_names(site_names) + likely_foot_names(geom_names)
            ))),
            "",
            format_indexed_block("likely foot geom candidates", likely_foot_geom_candidates(model)),
            "",
            format_indexed_block("detected foot contact geoms", detected_foot_contact_geoms(model)),
        ]
    )
    return "\n".join(lines)
