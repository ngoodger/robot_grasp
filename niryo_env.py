import gym
from gym import spaces
import math
import numpy as np
import os
import pybullet
from typing import List, Tuple, Any
import xml
import xml.etree.ElementTree as ET
import urdf_helpers
from functools import reduce
import logging

# Initial State 
ROBOT_START_POS = [0.0, 0.0, 0.001]
ROBOT_START_ORIENTATION = [0.0, 0.0, 0.0]
PART_START_POS = [0.5, 0.0, 0.001]
PART_START_ORIENTATION = [0.0, 0.0, 0.0]


# Robot Parameters  

# Env Parameters 
TIMESTEP_SECONDS = 0.01
GRAVITY_ACCELERATION_Z = -9.81
PART_TARGET_Z = 0.03
PART_TARGET_Z_DONE_TOL_Z = 0.003
PART_TARGET_Z_DONE_TOL_STEPS = 100
MAX_STEPS_DONE = 10000


class NiryoRobotEnv(gym.Env):
    metadata = {
                'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50
               }

    def __init__(self, render: bool=True) -> None:
        logging.basicConfig(filename= "file.log", level=logging.DEBUG)
        # self._observation = [] # type: List[np.ndarray]
        path = os.path.abspath(os.path.dirname("__file__"))
        self.robot_urdf_abs_path = os.path.join(path, "niryo_one_description"
                                                      "/urdf/niryo_one.urdf")
        self.part_urdf_abs_path = os.path.join(path,
                                               "niryo_one_description"
                                               "/urdf/part.urdf")
        robot_urdf_tree = ET.parse(self.robot_urdf_abs_path)
        robot_urdf_root = robot_urdf_tree.getroot()
        lower, upper = urdf_helpers.urdf_limits_to_numpy(robot_urdf_root)
        self.observation_space = spaces.Box(low=lower,
                                            high=upper,
                                            dtype=np.float32)
        self.movable_joints_idx_list = (urdf_helpers.
                                        urdf_movable_joints_idx
                                        (robot_urdf_root))
        self.action_space = spaces.Discrete(len(self.movable_joints_idx_list))
        if (render):
            self.pysicsClient = pybullet.connect(pybullet.GUI)
        else:
            self.pysicsClient = pybullet.connect(pybullet.DIRECT)
        logging.debug('Niro Env Initialized')
        # self._seed()

    def _step(self, action: List[float]) -> Tuple[np.ndarray,
                                                  float, bool, dict]:
        self._assign_force(action)
        pybullet.stepSimulation()
        self._observation = self._compute_observation()
        reward = self._compte_reward()
        done = self._compute_done()
        self._envStepCounter += 1
        self._env_done_tolerance_step_counter = 0
        logging.debug('Niro Env Step')
        return (self._observation, reward, done, {})

    def reset(self) -> np.ndarray:
        self._envStepCounter = 0
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, GRAVITY_ACCELERATION_Z)
        pybullet.setTimeStep(TIMESTEP_SECONDS)
        pybullet.loadURDF("plane.urdf")
        robot_start_pos = ROBOT_START_POS
        robot_start_orientation = (pybullet.
                                   getQuaternionFromEuler
                                   (ROBOT_START_ORIENTATION))
        self.bot_id = pybullet.loadURDF(self.robot_urdf_abs_path,
                                        robot_start_pos,
                                        robot_start_orientation,
                                        useFixedBase=True)
        logging.debug('Niro Env Robot loaded')
        part_start_pos = PART_START_POS
        part_start_orientation = (pybullet
                                  .getQuaternionFromEuler
                                  (PART_START_ORIENTATION))
        self.part_id = pybullet.loadURDF(self.part_urdf_abs_path,
                                         part_start_pos,
                                         part_start_orientation)
        logging.debug('Niro Env Part loaded')
        self._observation = self._compute_observation()
        logging.debug('Niro Env Reset')
        return np.array(self._observation)

    def _assign_force(self,  action: List[float]):
        pybullet.setJointMotorControlArray(bodyUniqueId=self.bot_id,
                                           jointIndices=list(range(ACTION_SPACE_SIZE)),
                                           controlMode=pybullet.TORQUE_CONTROL,
                                           forces=action)
        logging.debug('Niro Env Force Assigned')

    def _compute_observation(self) -> np.ndarray:
        joint_states = pybullet.getJointStates(self.bot_id,
                                               self.movable_joints_idx_list)
        # Extract only the position and velocity from the joint states.
        filtered_joint_states = list(list(joint_state[0:2])
                                     for joint_state in joint_states)
        robot_joint_state = list(reduce(lambda x, y: x + y[0:2],
                                 filtered_joint_states, []))
        part_pos, part_orientation = (pybullet
                                      .getBasePositionAndOrientation
                                      (self.part_id))
        part_orientation_euler = (pybullet
                                  .getEulerFromQuaternion
                                  (part_orientation))
        part_velocity, part_angular_velocity = (pybullet
                                                .getBaseVelocity(self.part_id))
        observation = (robot_joint_state + list(part_pos) +
                       list(part_orientation_euler) +
                       list(part_velocity) +
                       list(part_angular_velocity))
        logging.debug('Niro Env Observation complete')
        return observation

    def _compute_reward(self) -> float:
        # CONFIRM THIS IS Z?
        part_z = pybullet.getBasePositionAndOrientation(self.part_id)[0][2]
        reward = 1.0 - abs(PART_TARGET_Z - part_z)
        logging.debug('Niro Env Reward Computed')
        return reward

    def _compute_done(self) -> bool:
        part_z = pybullet.getBasePositionAndOrientation(self.part_id)[0][2]
        diff_z = abs(PART_TARGET_Z - part_z)
        if diff_z <= PART_TARGET_Z_DONE_TOL_Z:
            self._env_done_tolerance_step_counter += 1
        else:                                                    
            self._env_done_tolerance_step_counter = 0
        done = (self._env_done_tolerance_step_counter >= PART_TARGET_Z_DONE_TOL_STEPS or self.env_step_counter >= MAX_STEPS_DONE)
              
        logging.debug('Niro Env Computed done')
        return done
        # return cubepybullet.s[2] < 0.15 or self._envStepCounter >= 1500
        
    def _render(self, mode='human', close=False) -> None:
        pass
