import gym
from gym import spaces
import math
import numpy as np
import os
import pybullet
from typing import List, Tuple, Any

# Initial State 
ROBOT_START_POS = [0.0, 0.0, 0.001]
ROBOT_START_ORIENTATION = [0.0, 0.0, 0.0]
PART_START_POS = [0.5, 0.0, 0.001]
PART_START_ORIENTATION = [0.0, 0.0, 0.0]


# Robot Parameters  
ACTION_SPACE_SIZE = 9

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
                'video.frames_per_second' : 50
               }

        
    def __init__(self, render: bool=True) -> None:
        #self._observation = [] # type: List[np.ndarray]
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(low=np.array([0.0, # ground_joint
                                                          -3.05433, # joint_1
                                                          -1.5707963268, # joint_2
                                                          -1.401, # joint_3
                                                          -2.61799, # joint_4
                                                          -2.26893, #joint_5
                                                          -2.57 , #joint_6
                                                          ]),
                                            high=np.array([0.0, # ground_joint
                                                           3.05433, #joint_1
                                                           0.628319, #joint_2
                                                           0.994838, #joint_3
                                                           2.61799, #joint_4
                                                           2.57, #joint_5
                                                           ]),
                                            dtype=np.float32)
        if (render):
            self.pysicsClient = pybullet.connect(pybullet.GUI)
        else:
            self.pysicsClient = pybullet.connect(pybullet.DIRECT)
        #self._seed()
        
        
    def _step(self, action: List[float]) -> Tuple[np.ndarray, float, bool, dict]:
        self._assign_force(action)
        pybullet.stepSimulation()
        self._observation = self._compute_observation()
        reward = self._compte_reward()
        done = self._compute_done()
        self._envStepCounter += 1
        self._env_done_tolerance_step_counter = 0
        return (np.array(self._observation), reward, done, {})

    
    def reset(self) -> np.ndarray:
        self._envStepCounter = 0
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, GRAVITY_ACCELERATION_Z)
        pybullet.setTimeStep(TIMESTEP_SECONDS)
        pyane_id = pybullet.loadURDF("plane.urdf")
        robot_start_pos = ROBOT_START_POS 
        robot_start_orientation = pybullet.getQuaternionFromEuler(ROBOT_START_ORIENTATION)
        path = os.path.abspath(os.path.dirname("__file__"))
        robot_urdf_abs_path = os.path.join(path, "niryo_one_description/urdf/niryo_one.urdf")
        self.bot_id = pybullet.loadURDF(robot_urdf_abs_path,
                                        robot_start_pos,
                                        robot_start_orientation,
                                        useFixedBase=True)
        part_start_pos = PART_START_POS 
        part_start_orientation = pybullet.getQuaternionFromEuler(PART_START_ORIENTATION)
        part_urdf_abs_path = os.path.join(path, "niryo_one_description/urdf/part.urdf")
        self.part_id = pybullet.loadURDF(part_urdf_abs_path,
                                                part_start_pos,
                                                part_start_orientation)
        self._observation = self._compute_observation()
        return np.array(self._observation)


    def _assign_force(self,  action: List[float]):
        pybullet.setJointMotorControlArray(bodyUniqueId=self.bot_id,
                                                  jointIndices = list(range(ACTION_SPACE_SIZE)),
                                                  controlMode=pybullet.TORQUE_CONTROL,
                                                  forces=action)


    def _compute_observation(self) -> np.ndarray:
        bot_pos, bot_orientation = pybullet.getBasePositionAndOrientation(self.bot_id)
        bot_orientation_euler = pybullet.getEulerFromQuaternion(bot_orientation)
        bot_velocity, bot_angular_velocity = pybullet.getBaseVelocity(self.bot_id)
        part_pos, part_orientation = pybullet.getBasePositionAndOrientation(self.part_id)
        part_orientation_euler = pybullet.getEulerFromQuaternion(part_orientation)
        part_velocity, part_angular_velocity = pybullet.getBaseVelocity(self.part_id)
        observation = (bot_pos + bot_orientation_euler + bot_velocity + bot_angular_velocity +part_pos + part_orientation_euler + part_velocity + part_angular_velocity)
                       
        return observation

    def _compute_reward(self) -> float:
        #CONFIRM THIS IS Z?
        part_z = pybullet.getBasePositionAndOrientation(self.part_id)[0][2]
        reward = 1.0 - abs(PART_TARGET_Z - part_z)
        return reward

    def _compute_done(self) -> bool:
        part_z = pybullet.getBasePositionAndOrientation(self.part_id)[0][2]
        diff_z = abs(PART_TARGET_Z - part_z)
        if diff_z <= PART_TARGET_Z_DONE_TOL_Z: 
            self._env_done_tolerance_step_counter += 1
        else:                                                     
            self._env_done_tolerance_step_counter = 0
        done = (self._env_done_tolerance_step_counter >= PART_TARGET_Z_DONE_TOL_STEPS or self.env_step_counter >= MAX_STEPS_DONE)
              
        return done 
        # return cubepybullet.s[2] < 0.15 or self._envStepCounter >= 1500
        
        
    def _render(self, mode='human', close=False) -> None:
        pass
