import numpy as np
from typing import Any

MOVABLE_JOINT_TYPES = ["revolute", "planar"]


def urdf_movable_joints_idx(urdf: Any):
    urdf_joints = urdf.findall("joint")
    movable_joints_idx = []
    for idx, urdf_joint in enumerate(urdf_joints):
        if (urdf_joint.attrib["type"] in MOVABLE_JOINT_TYPES):
            movable_joints_idx.append(idx)
    return movable_joints_idx


def urdf_limits_to_numpy(urdf: Any):
    # Return  URDF defined limits as lower and upper np.ndarray
    lower_limits, upper_limits = [], []
    urdf_joints = urdf.findall("joint")
    for urdf_joint in urdf_joints:
        if (urdf_joint.attrib["type"] in MOVABLE_JOINT_TYPES):
            limit = urdf_joint.find("limit")
            if limit is None:
                raise ValueError("Limit is not defined for tag: {} "
                                 .format(urdf_joint.attrib["name"]))
            lower_limits.append(float(limit.attrib["lower"]))
            lower_limits.append(-float(limit.attrib["velocity"]))
            upper_limits.append(float(limit.attrib["upper"]))
            upper_limits.append(float(limit.attrib["velocity"]))
    return (np.array(lower_limits), np.array(upper_limits))
