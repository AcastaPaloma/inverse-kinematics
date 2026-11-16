import numpy as np

from kinematics import forward_kinematics


def jacobian(q: np.ndarray) -> np.ndarray:
    """
    q: (5,) joint angles.
    Returns J: (3, 5), column i = d p_end / d q_i.
    """
    raise NotImplementedError
