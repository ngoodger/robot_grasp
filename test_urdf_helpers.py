import xml.etree.ElementTree as ET
import numpy as np
import pytest
import urdf_helpers
import xml


@pytest.fixture
def urdf_with_4_joints() -> xml.etree.ElementTree.Element:
    root = ET.Element("root")
    # Joint 0: fixed.
    joint3 = ET.SubElement(root, "joint")
    joint3.attrib = {"name": "joint3", "type": "fixed"}
    # Joint 1: revolute.
    joint0 = ET.SubElement(root, "joint")
    joint0.attrib = {"name": "joint0", "type": "revolute"}
    limit0 = ET.SubElement(joint0, "limit")
    limit0.attrib = {"effort": "1", "lower": "-2.57",
                     "upper": "2.57", "velocity": "1.0"}
    # Joint 2: revolute.
    joint1 = ET.SubElement(root, "joint")
    joint1.attrib = {"name": "joint1", "type": "revolute"}
    limit1 = ET.SubElement(joint1, "limit")
    limit1.attrib = {"effort": "0.5", "lower": "-1.57",
                     "upper": "1.57", "velocity": "0.5"}
    # Joint 3: planer.
    joint2 = ET.SubElement(root, "joint")
    joint2.attrib = {"name": "joint2", "type": "planar"}
    limit2 = ET.SubElement(joint2, "limit")
    limit2.attrib = {"effort": "0.1", "lower": "-0.57",
                     "upper": "0.57", "velocity": "0.1"}
    return root


def test_urdf_limits_to_numpy(urdf_with_4_joints):
    lower_correct = np.array([-2.57, -1., -1.57, -0.5, -0.57, -0.1])
    upper_correct = np.array([2.57, 1., 1.57, 0.5, 0.57, 0.1])
    lower, upper = urdf_helpers.urdf_limits_to_numpy(urdf_with_4_joints)
    assert(np.array_equal(lower_correct, lower))
    assert(np.array_equal(upper_correct, upper))


def test_urdf_movable_joints_idx(urdf_with_4_joints):
    movable_joints_correct = [1, 2, 3]
    movable_joints = urdf_helpers.urdf_movable_joints_idx(urdf_with_4_joints)
    assert(movable_joints_correct == movable_joints)
