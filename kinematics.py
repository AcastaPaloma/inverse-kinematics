import numpy as np

from robot import D1, A1, L1, L2, L3, L_GRIP


def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def forward_kinematics(q: np.ndarray):
    """
    q: (5,) joint angles in radians  [J0 yaw, J1 pitch, J2 pitch, J3 pitch, J4 roll]

    Returns (positions, rotations):
      positions : (7, 3) world coordinates in meters
                  [0] base origin  (0, 0, 0)
                  [1] J0           [2] J1   [3] J2   [4] J3   [5] J4
                  [6] gripper tip  <- this is p_end
      rotations : list of 5 (3, 3) world rotation matrices R_0 .. R_4
                  column 0/1/2 = joint's local x/y/z axis in world coordinates

    Sanity check at q = 0 (arm straight out along +X at height D1):
      positions[6] == [A1 + L1 + L2 + L3 + L_GRIP, 0, D1]

    Convention implied by that home pose:
      J0 rotates about world z, J1..J3 pitch about y, J4 rolls about the link (x).

    Raise NotImplementedError until done; the visualizer then shows a placeholder.
    """
    d_base = np.array([0.0, 0.0, D1]) # base plate -> J0
    d = [
        np.array([A1,     0.0, 0.0]),   # J0 -> J1
        np.array([L1,     0.0, 0.0]),   # J1 -> J2
        np.array([L2,     0.0, 0.0]),   # J2 -> J3
        np.array([L3,     0.0, 0.0]),   # J3 -> J4
        np.array([L_GRIP, 0.0, 0.0]),   # J4 -> tip
    ]

    # (x, y, z) at each of the seven joints || [0]: base; [1 - 5]: J0 - J4; [6]: Tip;
    positions = np.zeros((7, 3))
    rotations = []
    R = np.identity(3)

    # pj0 = pbase + dbase = <0, 0, D1> (D1 = length of the first shaft)
    positions[1] = positions[0] + d_base




    raise NotImplementedError
