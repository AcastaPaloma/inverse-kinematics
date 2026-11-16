import numpy as np

from kinematics import forward_kinematics
from jacobian import jacobian


def solve_ik(q0: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    q0:     (5,) starting joint angles
    target: (3,) desired gripper-tip position in meters
    Returns q: (5,) joint angles that put p_end at target.
    """
    raise NotImplementedError
