from typing import Any
from niryo_env import NiryoRobotEnv
import pytest


@pytest.fixture
def robot_test_env() -> None:
    test_env = NiryoRobotEnv(render=False)
    test_env.action_space
    test_env.reset()
    return test_env


def test_compute_observation(robot_test_env: Any) -> Any:
    # action = [0.] * test_env.action_space.n
    correct_observation = [0., 0.,
                           0., 0.,
                           0., 0.,
                           0., 0.,
                           0., 0.,
                           0., 0.,
                           0., 0.,
                           0.5, 0., 0.001,
                           0., 0., 0.,
                           0., 0., 0.,
                           0., 0., 0.]
    observation = robot_test_env._compute_observation()
    print(observation)
    print(correct_observation)
    assert (correct_observation == observation) 
