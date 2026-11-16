"""SO-101 arm constants. Pure data — no math, no plotting."""

# Link dimensions in meters
D1 = 0.062      # base plate -> J0 (shoulder height)
A1 = 0.030      # J0 -> J1 forward offset
L1 = 0.116      # J1 -> J2 (upper arm)
L2 = 0.135      # J2 -> J3 (forearm)
L3 = 0.060      # J3 -> J4 (wrist)
L_GRIP = 0.055  # J4 -> gripper tip

N_JOINTS = 5
JOINT_NAMES = [
    "J0 base yaw",
    "J1 shoulder pitch",
    "J2 elbow pitch",
    "J3 wrist pitch",
    "J4 wrist roll",
]
