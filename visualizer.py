import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from kinematics import forward_kinematics
from ik_solver import solve_ik

# Link dimensions (matches kinematics.py spec)
D1, A1, L1, L2, L3, L_GRIP = 0.062, 0.030, 0.116, 0.135, 0.060, 0.055

def mock_home_pose():
    """Static q = 0 pose along +X so you can visualize & drag before implementing FK."""
    positions = np.array([
        [0.0, 0.0, 0.0],                                   # Base origin
        [0.0, 0.0, D1],                                    # J0: Base Yaw
        [A1,  0.0, D1],                                    # J1: Shoulder Pitch
        [A1 + L1, 0.0, D1],                                # J2: Elbow Pitch
        [A1 + L1 + L2, 0.0, D1],                           # J3: Wrist Pitch
        [A1 + L1 + L2 + L3, 0.0, D1],                      # J4: Wrist Roll
        [A1 + L1 + L2 + L3 + L_GRIP, 0.0, D1],             # Gripper Tip
    ], dtype=float)
    rotations = [np.eye(3) for _ in range(5)]
    return positions, rotations


class ArmVisualizer:
    def __init__(self, ik_enabled: bool = False):
        self.q = np.zeros(5)
        self.positions, self.rotations = mock_home_pose()
        self.target = np.array([0.22, 0.08, 0.18])
        self.ik_enabled = ik_enabled

        # Drag state: None, "target", or int index (1..6) for dragging joints manually
        self.dragging = None
        self.last_mouse_xy = None

        self.fig = plt.figure(figsize=(9, 8))
        plt.subplots_adjust(bottom=0.16)
        self.ax = self.fig.add_subplot(111, projection="3d")

        # "Run IK" button (greyed out when ik_enabled=False)
        btn_ax = plt.axes([0.38, 0.04, 0.24, 0.05])
        self.btn_ik = Button(
            btn_ax,
            "Run IK" if ik_enabled else "Run IK (Locked)",
            color="#d0e8ff" if ik_enabled else "#e0e0e0",
            hovercolor="#a6d4ff" if ik_enabled else "#e0e0e0",
        )
        self.btn_ik.label.set_color("black" if ik_enabled else "#888888")
        self.btn_ik.on_clicked(self._on_run_ik)

        # Mouse event hooks for 3D dragging
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        self.render()

    def _project_to_screen(self, pt3d: np.ndarray) -> np.ndarray:
        """Projects a 3D world point to 2D display pixel coordinates."""
        from mpl_toolkits.mplot3d import proj3d
        x2, y2, _ = proj3d.proj_transform(pt3d[0], pt3d[1], pt3d[2], self.ax.get_proj())
        return self.ax.transData.transform((x2, y2))

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        mouse_px = np.array([event.x, event.y])

        # Check if user clicked near the target marker (within 18 pixels)
        target_px = self._project_to_screen(self.target)
        if np.linalg.norm(mouse_px - target_px) < 18:
            self.dragging = "target"
            self.last_mouse_xy = mouse_px
            self.ax.disable_mouse_rotation()  # Lock camera orbit while dragging point
            return

        # Check if user clicked near any movable joint/tip (indices 1..6)
        for idx in range(1, len(self.positions)):
            joint_px = self._project_to_screen(self.positions[idx])
            if np.linalg.norm(mouse_px - joint_px) < 15:
                self.dragging = idx
                self.last_mouse_xy = mouse_px
                self.ax.disable_mouse_rotation()
                return

    def _on_motion(self, event):
        if self.dragging is None or event.inaxes != self.ax or event.x is None:
            return

        curr_px = np.array([event.x, event.y])
        delta_px = curr_px - self.last_mouse_xy
        self.last_mouse_xy = curr_px

        # Map 2D screen drag onto camera right & up vectors in 3D world space
        azim = np.deg2rad(self.ax.azim)
        elev = np.deg2rad(self.ax.elev)
        cam_right = np.array([-np.sin(azim), np.cos(azim), 0.0])
        cam_up = np.array([
            -np.cos(azim) * np.sin(elev),
            -np.sin(azim) * np.sin(elev),
            np.cos(elev),
        ])

        scale = 0.0012  # meters per pixel
        delta_3d = (delta_px[0] * cam_right + delta_px[1] * cam_up) * scale

        if self.dragging == "target":
            self.target += delta_3d
        else:
            # Dragging a joint moves it and everything downstream so you can pose manually
            self.positions[self.dragging:] += delta_3d

        self.render()

    def _on_release(self, _):
        if self.dragging is not None:
            self.dragging = None
            self.ax.mouse_init()  # Restore normal 3D camera orbit

    def _on_run_ik(self, _):
        if not self.ik_enabled:
            return
        # TODO: Once kinematics.py, jacobian.py, and ik_solver.py are ready:
        # self.q = solve_ik(self.q, self.target)
        # self.positions, self.rotations = forward_kinematics(self.q)

        self.render()

    def render(self):
        self.ax.clear()

        # Links & joints
        self.ax.plot(
            self.positions[:, 0], self.positions[:, 1], self.positions[:, 2],
            "-o", linewidth=4, markersize=7, color="#1f77b4", label="SO-101 Arm (Drag joints)"
        )

        # RGB local axes at each joint
        axis_len = 0.025
        for pos, rot in zip(self.positions[1:-1], self.rotations):
            for idx, col in enumerate(["r", "g", "b"]):
                vec = rot[:, idx] * axis_len
                self.ax.quiver(pos[0], pos[1], pos[2], vec[0], vec[1], vec[2], color=col, lw=1.2)

        # Gripper tip & Target
        tip = self.positions[-1]
        self.ax.scatter(*tip, color="green", s=90, label="Gripper Tip")
        self.ax.scatter(*self.target, color="red", s=130, marker="x", lw=2.5, label="Target (Drag me)")

        err_mm = np.linalg.norm(self.target - tip) * 1000.0
        self.ax.set_title(
            f"SO-101 Sandbox | Target: ({self.target[0]:.2f}, {self.target[1]:.2f}, {self.target[2]:.2f}) | Err: {err_mm:.1f} mm"
        )

        limit = 0.38
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([0.0, limit * 1.3])
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.legend(loc="upper right")
        self.fig.canvas.draw_idle()

        print("JOIN CONFIGURATION: ", self.q)


if __name__ == "__main__":
    viz = ArmVisualizer(ik_enabled=False)
    plt.show()