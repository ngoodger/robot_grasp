import xml.etree.ElementTree as ET
import numpy as np
import niryo_env
import pytest


@pytest.fixture
def urdf_with_4_joints():
    root = ET.Element("root")
    joint0 = ET.SubElement(root, "joint")
    joint0.attrib = {"name": "joint0", "type": "revolute"}
    limit0 = ET.SubElement(joint0, "limit")
    limit0.attrib = {"effort": "1", "lower": "-2.57",
                     "upper": "2.57", "velocity": "1.0"}
    joint1 = ET.SubElement(root, "joint")
    joint1.attrib = {"name": "joint1", "type": "revolute"}
    limit1 = ET.SubElement(joint1, "limit")
    limit1.attrib = {"effort": "0.5", "lower": "-1.57",
                     "upper": "1.57", "velocity": "0.5"}
    joint2 = ET.SubElement(root, "joint")
    joint2.attrib = {"name": "joint2", "type": "planar"}
    limit2 = ET.SubElement(joint2, "limit")
    limit2.attrib = {"effort": "0.1", "lower": "-0.57",
                     "upper": "0.57", "velocity": "0.1"}
    joint3 = ET.SubElement(root, "joint")
    joint3.attrib = {"name": "joint3", "type": "fixed"}
    return root


def test_urdf_state_limits_to_numpy(urdf_with_4_joints):
    lower_correct = np.array([-2.57, -1., -1.57, -0.5, -0.57, -0.1])
    upper_correct = np.array([2.57, 1., 1.57, 0.5, 0.57, 0.1])
    lower, upper = niryo_env.urdf_state_limits_to_numpy(urdf_with_4_joints)
    assert(np.array_equal(lower_correct, lower))
    assert(np.array_equal(upper_correct, upper))
