import subprocess
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

from kinematics import forward_kinematics
from ik_solver import solve_ik

# Link dimensions (matches kinematics.py spec)
D1, A1, L1, L2, L3, L_GRIP = 0.062, 0.030, 0.116, 0.135, 0.060, 0.055

HOME_TARGET = np.array([0.22, 0.08, 0.18])
LIMIT = 0.38
VIEWS = {  # key: (elev, azim)
    "1": (22, -58),   # iso
    "2": (0, -90),    # front
    "3": (0, 0),      # side
    "4": (90, -90),   # top
}


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


# ── Theme ────────────────────────────────────────────────────────────────────

def _macos_dark_mode() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        out = subprocess.run(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True, text=True, timeout=1,
        )
        return out.stdout.strip().lower() == "dark"
    except Exception:
        return False


# Apple system colors
THEMES = {
    "dark": dict(
        bg="#1c1c1e", fg="#f2f2f7", muted="#8e8e93", grid="#3a3a3c",
        accent="#0a84ff", tip="#30d158", target="#ff453a",
        x="#ff453a", y="#30d158", z="#0a84ff",
        btn="#0a84ff", btn_fg="#ffffff", btn_off="#2c2c2e", btn_off_fg="#636366",
    ),
    "light": dict(
        bg="#fbfbfd", fg="#1d1d1f", muted="#86868b", grid="#e5e5ea",
        accent="#007aff", tip="#34c759", target="#ff3b30",
        x="#ff3b30", y="#34c759", z="#007aff",
        btn="#007aff", btn_fg="#ffffff", btn_off="#e5e5ea", btn_off_fg="#aeaeb2",
    ),
}


def _setup_matplotlib():
    mpl.rcParams.update({
        "toolbar": "None",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.monospace": ["SF Mono", "Menlo", "DejaVu Sans Mono"],
        "figure.dpi": 100,
    })
    # Free up keys we use for our own shortcuts
    for k in ("keymap.home", "keymap.back", "keymap.forward"):
        mpl.rcParams[k] = [key for key in mpl.rcParams[k] if key not in ("r", "left", "right")]


class ArmVisualizer:
    PICK_RADIUS_PX = 16

    def __init__(self, ik_enabled: bool = False, theme: str | None = None):
        _setup_matplotlib()
        self.c = THEMES[theme or ("dark" if _macos_dark_mode() else "light")]

        self.q = np.zeros(5)
        self.positions, self.rotations = mock_home_pose()
        self.target = HOME_TARGET.copy()
        self.ik_enabled = ik_enabled

        # Interaction state: None, "target", or int joint index (1..6)
        self.dragging = None
        self.hover = None
        self.last_mouse_xy = None
        self.zoom = 1.0

        self._build_figure()
        self._build_artists()
        self._connect_events()
        self.render()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _build_figure(self):
        c = self.c
        self.fig = plt.figure(figsize=(9, 7.5), facecolor=c["bg"])
        try:
            self.fig.canvas.manager.set_window_title("SO-101 Sandbox")
        except Exception:
            pass

        self.ax = self.fig.add_axes([0.0, 0.08, 1.0, 0.82], projection="3d")
        self.ax.set_facecolor(c["bg"])
        self.ax.set_axis_off()
        self.ax.set_box_aspect((1, 1, 0.5), zoom=1.4)
        self.ax.view_init(*VIEWS["1"])
        self._apply_limits()

        # Header
        self.fig.text(0.04, 0.955, "SO-101", fontsize=15, weight="bold", color=c["fg"], va="top")
        self.fig.text(0.04, 0.918, "Inverse kinematics sandbox", fontsize=10.5, color=c["muted"], va="top")

        # Readout (top right)
        mono = dict(family="monospace", fontsize=10, va="top", ha="right")
        self.txt_target = self.fig.text(0.96, 0.955, "", color=c["fg"], **mono)
        self.txt_err = self.fig.text(0.96, 0.918, "", color=c["muted"], **mono)

        # Footer hint
        self.fig.text(
            0.04, 0.035,
            "Drag  target or joints     Drag empty space  orbit     Scroll  zoom\n"
            "1–4  views     R  reset     Space  run IK",
            fontsize=9, color=c["muted"], va="bottom", linespacing=1.7,
        )

        # Pill button (bottom right)
        self.btn = self.fig.text(
            0.96, 0.045, "Run IK" if self.ik_enabled else "IK locked",
            ha="right", va="bottom", fontsize=11, weight="bold",
            color=c["btn_fg"] if self.ik_enabled else c["btn_off_fg"],
            bbox=dict(
                boxstyle="round,pad=0.7,rounding_size=0.9",
                fc=c["btn"] if self.ik_enabled else c["btn_off"], ec="none",
            ),
        )

    def _build_artists(self):
        c, ax = self.c, self.ax

        # Floor: concentric rings + cross, gives depth cues without clutter
        t = np.linspace(0, 2 * np.pi, 120)
        for r in (0.1, 0.2, 0.3):
            ax.plot(r * np.cos(t), r * np.sin(t), 0 * t, color=c["grid"], lw=0.8, zorder=0)
        ax.plot([-0.3, 0.3], [0, 0], [0, 0], color=c["grid"], lw=0.8)
        ax.plot([0, 0], [-0.3, 0.3], [0, 0], color=c["grid"], lw=0.8)
        # World axis hint at origin
        for vec, col in zip(np.eye(3) * 0.05, (c["x"], c["y"], c["z"])):
            ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color=col, lw=1.6, alpha=0.6)
        # Base plate
        ax.plot(0.035 * np.cos(t), 0.035 * np.sin(t), 0 * t, color=c["muted"], lw=2.2)

        # Arm
        (self.ln_links,) = ax.plot([], [], [], color=c["accent"], lw=6,
                                   solid_capstyle="round", solid_joinstyle="round")
        (self.ln_joints,) = ax.plot([], [], [], ls="none", marker="o", ms=8,
                                    mfc=c["bg"], mec=c["accent"], mew=2)
        (self.ln_tip,) = ax.plot([], [], [], ls="none", marker="o", ms=9,
                                 mfc=c["tip"], mec=c["tip"])

        # Local joint frames
        self.ln_frames = [
            [ax.plot([], [], [], color=col, lw=1.3, alpha=0.85)[0]
             for col in (c["x"], c["y"], c["z"])]
            for _ in self.rotations
        ]

        # Target, its floor shadow, and the error line to the tip
        (self.ln_err,) = ax.plot([], [], [], color=c["muted"], lw=1, ls=(0, (2, 3)))
        (self.ln_drop,) = ax.plot([], [], [], color=c["target"], lw=1, ls=(0, (2, 3)), alpha=0.5)
        (self.ln_shadow,) = ax.plot([], [], [], ls="none", marker="o", ms=5,
                                    mfc=c["target"], mec="none", alpha=0.35)
        (self.ln_target,) = ax.plot([], [], [], ls="none", marker="X", ms=13,
                                    mfc=c["target"], mec=c["bg"], mew=1.2)

        # Hover halo
        (self.ln_halo,) = ax.plot([], [], [], ls="none", marker="o", ms=20,
                                  mfc="none", mec=c["fg"], mew=1.2, alpha=0.5)

    def _connect_events(self):
        cv = self.fig.canvas
        cv.mpl_connect("button_press_event", self._on_press)
        cv.mpl_connect("motion_notify_event", self._on_motion)
        cv.mpl_connect("button_release_event", self._on_release)
        cv.mpl_connect("scroll_event", self._on_scroll)
        cv.mpl_connect("key_press_event", self._on_key)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _apply_limits(self):
        lim = LIMIT * self.zoom
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(0.0, lim)

    def _project_to_screen(self, pt3d: np.ndarray) -> np.ndarray:
        """Projects a 3D world point to 2D display pixel coordinates."""
        x2, y2, _ = proj3d.proj_transform(pt3d[0], pt3d[1], pt3d[2], self.ax.get_proj())
        return self.ax.transData.transform((x2, y2))

    def _pick(self, mouse_px):
        """Return what's under the cursor: 'target', a joint index, or None."""
        if np.linalg.norm(mouse_px - self._project_to_screen(self.target)) < self.PICK_RADIUS_PX + 2:
            return "target"
        best, best_d = None, self.PICK_RADIUS_PX
        for idx in range(1, len(self.positions)):
            d = np.linalg.norm(mouse_px - self._project_to_screen(self.positions[idx]))
            if d < best_d:
                best, best_d = idx, d
        return best

    def _button_hit(self, event) -> bool:
        patch = self.btn.get_bbox_patch()
        return patch is not None and patch.get_window_extent().contains(event.x, event.y)

    def _set_cursor(self, kind):
        try:
            from matplotlib.backend_tools import Cursors
            self.fig.canvas.set_cursor(getattr(Cursors, kind))
        except Exception:
            pass

    # ── Events ───────────────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.button != 1 or event.x is None:
            return
        if self._button_hit(event):
            self._on_run_ik()
            return
        if event.inaxes != self.ax:
            return

        mouse_px = np.array([event.x, event.y])
        hit = self._pick(mouse_px)
        if hit is not None:
            self.dragging = hit
            self.last_mouse_xy = mouse_px
            self.ax.disable_mouse_rotation()  # Lock camera orbit while dragging a point
            self._set_cursor("MOVE")

    def _on_motion(self, event):
        if event.x is None:
            return
        curr_px = np.array([event.x, event.y])

        if self.dragging is None:
            # Hover feedback only
            hit = self._pick(curr_px) if event.inaxes == self.ax else None
            over_btn = self.ik_enabled and self._button_hit(event)
            self._set_cursor("HAND" if (hit is not None or over_btn) else "POINTER")
            if hit != self.hover:
                self.hover = hit
                self.render()
            return

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

        # Meters per pixel, scaled with zoom so drags track the cursor
        scale = (2 * LIMIT * self.zoom) / max(self.ax.bbox.width, 1) * 1.35
        delta_3d = (delta_px[0] * cam_right + delta_px[1] * cam_up) * scale

        if self.dragging == "target":
            self.target += delta_3d
            self.target[2] = max(self.target[2], 0.0)
        else:
            # Dragging a joint moves it and everything downstream so you can pose manually
            self.positions[self.dragging:] += delta_3d

        self.render()

    def _on_release(self, _):
        if self.dragging is not None:
            self.dragging = None
            self.ax.mouse_init()  # Restore normal 3D camera orbit
            self._set_cursor("HAND" if self.hover is not None else "POINTER")

    def _on_scroll(self, event):
        step = event.step if event.step else (1 if event.button == "up" else -1)
        self.zoom = float(np.clip(self.zoom * (0.92 ** step), 0.3, 2.5))
        self._apply_limits()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key in VIEWS:
            self.ax.view_init(*VIEWS[key])
            self.fig.canvas.draw_idle()
        elif key == "r":
            self.positions, self.rotations = mock_home_pose()
            self.target = HOME_TARGET.copy()
            self.q = np.zeros(5)
            self.zoom = 1.0
            self.ax.view_init(*VIEWS["1"])
            self._apply_limits()
            self.render()
        elif key in (" ", "enter"):
            self._on_run_ik()

    def _on_run_ik(self, _=None):
        if not self.ik_enabled:
            return
        # TODO: Once kinematics.py, jacobian.py, and ik_solver.py are ready:
        # self.q = solve_ik(self.q, self.target)
        # self.positions, self.rotations = forward_kinematics(self.q)

        self.render()

    # ── Drawing ──────────────────────────────────────────────────────────────

    def render(self):
        p, tgt = self.positions, self.target
        tip = p[-1]

        self.ln_links.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
        self.ln_joints.set_data_3d(p[1:-1, 0], p[1:-1, 1], p[1:-1, 2])
        self.ln_tip.set_data_3d([tip[0]], [tip[1]], [tip[2]])

        axis_len = 0.025
        for pos, rot, lines in zip(p[1:-1], self.rotations, self.ln_frames):
            for i, ln in enumerate(lines):
                end = pos + rot[:, i] * axis_len
                ln.set_data_3d([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]])

        self.ln_target.set_data_3d([tgt[0]], [tgt[1]], [tgt[2]])
        self.ln_shadow.set_data_3d([tgt[0]], [tgt[1]], [0.0])
        self.ln_drop.set_data_3d([tgt[0], tgt[0]], [tgt[1], tgt[1]], [0.0, tgt[2]])
        self.ln_err.set_data_3d([tip[0], tgt[0]], [tip[1], tgt[1]], [tip[2], tgt[2]])

        active = self.dragging if self.dragging is not None else self.hover
        if active is None:
            self.ln_halo.set_data_3d([], [], [])
        else:
            pt = tgt if active == "target" else p[active]
            self.ln_halo.set_data_3d([pt[0]], [pt[1]], [pt[2]])

        err_mm = np.linalg.norm(tgt - tip) * 1000.0
        self.txt_target.set_text(f"target  {tgt[0]:+.3f}  {tgt[1]:+.3f}  {tgt[2]:+.3f} m")
        self.txt_err.set_text(f"error  {err_mm:7.1f} mm")
        self.txt_err.set_color(self.c["tip"] if err_mm < 2.0 else self.c["muted"])

        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    viz = ArmVisualizer(ik_enabled=False)
    plt.show()
