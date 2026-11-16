"""
Display only. No kinematics math lives here.

It calls three functions and draws whatever they return:
    kinematics.forward_kinematics(q)  -> (positions (7,3), rotations [5 x (3,3)])
    ik_solver.solve_ik(q, target)     -> q (5,)
Any of them may raise NotImplementedError; the UI shows a notice instead of crashing.
"""
import subprocess
import sys
import traceback

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

from robot import D1, A1, L1, L2, L3, L_GRIP, N_JOINTS, JOINT_NAMES
from kinematics import forward_kinematics
from ik_solver import solve_ik

HOME_TARGET = np.array([0.22, 0.08, 0.18])
LIMIT = 0.44
STEP_DEG, FINE_STEP_DEG = 5.0, 1.0
VIEWS = {  # key: (elev, azim)
    "1": (22, -58),   # iso
    "2": (0, -90),    # front
    "3": (0, 0),      # side
    "4": (90, -90),   # top
}


def placeholder_pose():
    """Straight-out pose drawn while forward_kinematics is not implemented (ignores q)."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, D1],
        [A1, 0.0, D1],
        [A1 + L1, 0.0, D1],
        [A1 + L1 + L2, 0.0, D1],
        [A1 + L1 + L2 + L3, 0.0, D1],
        [A1 + L1 + L2 + L3 + L_GRIP, 0.0, D1],
    ], dtype=float)
    return positions, [np.eye(3) for _ in range(N_JOINTS)]


# ── Theme ────────────────────────────────────────────────────────────────────

def _macos_dark_mode() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        out = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"],
                             capture_output=True, text=True, timeout=1)
        return out.stdout.strip().lower() == "dark"
    except Exception:
        return False


THEMES = {  # Apple system colors
    "dark": dict(
        bg="#1c1c1e", fg="#f2f2f7", muted="#8e8e93", grid="#3a3a3c",
        accent="#0a84ff", tip="#30d158", target="#ff453a", warn="#ff9f0a",
        x="#ff453a", y="#30d158", z="#0a84ff",
        btn="#0a84ff", btn_fg="#ffffff",
    ),
    "light": dict(
        bg="#fbfbfd", fg="#1d1d1f", muted="#86868b", grid="#e5e5ea",
        accent="#007aff", tip="#34c759", target="#ff3b30", warn="#ff9500",
        x="#ff3b30", y="#34c759", z="#007aff",
        btn="#007aff", btn_fg="#ffffff",
    ),
}


def _setup_matplotlib():
    mpl.rcParams.update({
        "toolbar": "None",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.monospace": ["SF Mono", "Menlo", "DejaVu Sans Mono"],
    })
    # Free up keys we use for our own shortcuts
    for k in ("keymap.home", "keymap.back", "keymap.forward"):
        mpl.rcParams[k] = [key for key in mpl.rcParams[k] if key not in ("r", "left", "right")]


class ArmVisualizer:
    PICK_RADIUS_PX = 16

    def __init__(self, theme: str | None = None):
        _setup_matplotlib()
        self.c = THEMES[theme or ("dark" if _macos_dark_mode() else "light")]

        self.q = np.zeros(N_JOINTS)
        self.target = HOME_TARGET.copy()
        self.selected = 0          # joint index controlled by arrow keys
        self.status = ""           # one-line notice (FK/IK not implemented, errors)

        # Interaction state
        self.dragging = False      # dragging the target
        self.hover = None          # None, "target", or joint index 0..4
        self.last_mouse_xy = None
        self.zoom = 1.0

        self._build_figure()
        self._build_artists()
        self._connect_events()
        self.update_pose()

    # ── Calling into the math ────────────────────────────────────────────────

    def update_pose(self):
        """Run forward_kinematics(q) and redraw. Falls back to a placeholder."""
        try:
            positions, rotations = forward_kinematics(self.q.copy())
            positions = np.asarray(positions, dtype=float)
            rotations = [np.asarray(R, dtype=float) for R in rotations]
            if positions.shape != (7, 3) or len(rotations) != N_JOINTS \
                    or any(R.shape != (3, 3) for R in rotations):
                raise ValueError(
                    f"expected positions (7,3) and 5 rotations (3,3), got "
                    f"{positions.shape} and {[R.shape for R in rotations]}")
            self.positions, self.rotations = positions, rotations
            self.status = ""
        except NotImplementedError:
            self.positions, self.rotations = placeholder_pose()
            self.status = "forward_kinematics not implemented · showing placeholder"
        except Exception as e:
            traceback.print_exc()
            self.positions, self.rotations = placeholder_pose()
            self.status = f"forward_kinematics error: {e}"
        self.render()

    def run_ik(self):
        try:
            q = np.asarray(solve_ik(self.q.copy(), self.target.copy()), dtype=float)
            if q.shape != (N_JOINTS,):
                raise ValueError(f"solve_ik must return shape ({N_JOINTS},), got {q.shape}")
            self.q = q
            self.update_pose()
        except NotImplementedError:
            self.status = "solve_ik not implemented yet"
            self.render()
        except Exception as e:
            traceback.print_exc()
            self.status = f"solve_ik error: {e}"
            self.render()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _build_figure(self):
        c = self.c
        self.fig = plt.figure(figsize=(9.5, 7.5), facecolor=c["bg"])
        try:
            self.fig.canvas.manager.set_window_title("SO-101 Sandbox")
        except Exception:
            pass

        self.ax = self.fig.add_axes([0.12, 0.08, 0.88, 0.82], projection="3d")
        self.ax.set_facecolor(c["bg"])
        self.ax.set_axis_off()
        self.ax.set_box_aspect((1, 1, 0.5), zoom=1.4)
        self.ax.view_init(*VIEWS["1"])
        self._apply_limits()

        # Header
        self.fig.text(0.04, 0.955, "SO-101", fontsize=15, weight="bold", color=c["fg"], va="top")
        self.fig.text(0.04, 0.918, "Inverse kinematics sandbox", fontsize=10.5,
                      color=c["muted"], va="top")
        self.txt_status = self.fig.text(0.04, 0.885, "", fontsize=10, color=c["warn"], va="top")

        # Readout (top right)
        mono = dict(family="monospace", fontsize=10, va="top", ha="right")
        self.txt_tip = self.fig.text(0.96, 0.955, "", color=c["fg"], **mono)
        self.txt_target = self.fig.text(0.96, 0.925, "", color=c["muted"], **mono)
        self.txt_err = self.fig.text(0.96, 0.895, "", color=c["muted"], **mono)

        # Joint list (left)
        self.txt_joints = [
            self.fig.text(0.04, 0.74 - i * 0.042, "", family="monospace", fontsize=10,
                          va="top", color=c["muted"])
            for i in range(N_JOINTS)
        ]

        # Footer hint
        self.fig.text(
            0.04, 0.035,
            "Up / Down  select joint     Left / Right  rotate (Shift = fine)     0  zero joints\n"
            "Drag target  move it     Drag empty space  orbit     Scroll  zoom     1–4  views     R  reset     Space  run IK",
            fontsize=9, color=c["muted"], va="bottom", linespacing=1.7,
        )

        # Pill button (bottom right)
        self.btn = self.fig.text(
            0.96, 0.045, "Run IK", ha="right", va="bottom", fontsize=11, weight="bold",
            color=c["btn_fg"],
            bbox=dict(boxstyle="round,pad=0.7,rounding_size=0.9", fc=c["btn"], ec="none"),
        )

    def _build_artists(self):
        c, ax = self.c, self.ax

        # Floor rings + cross, world axes at origin, base plate
        t = np.linspace(0, 2 * np.pi, 120)
        for r in (0.1, 0.2, 0.3):
            ax.plot(r * np.cos(t), r * np.sin(t), 0 * t, color=c["grid"], lw=0.8)
        self._build_world_axes()
        ax.plot(0.035 * np.cos(t), 0.035 * np.sin(t), 0 * t, color=c["muted"], lw=2.2)

        # Arm
        (self.ln_links,) = ax.plot([], [], [], color=c["accent"], lw=6,
                                   solid_capstyle="round", solid_joinstyle="round")
        (self.ln_joints,) = ax.plot([], [], [], ls="none", marker="o", ms=8,
                                    mfc=c["bg"], mec=c["accent"], mew=2)
        (self.ln_selected,) = ax.plot([], [], [], ls="none", marker="o", ms=9,
                                      mfc=c["accent"], mec=c["fg"], mew=1.5)
        (self.ln_tip,) = ax.plot([], [], [], ls="none", marker="o", ms=9,
                                 mfc=c["tip"], mec=c["tip"])

        # Local joint frames (columns of R_i)
        self.ln_frames = [
            [ax.plot([], [], [], color=col, lw=1.3, alpha=0.85)[0]
             for col in (c["x"], c["y"], c["z"])]
            for _ in range(N_JOINTS)
        ]

        # Target, floor shadow, error line
        (self.ln_err,) = ax.plot([], [], [], color=c["muted"], lw=1, ls=(0, (2, 3)))
        (self.ln_drop,) = ax.plot([], [], [], color=c["target"], lw=1, ls=(0, (2, 3)), alpha=0.5)
        (self.ln_shadow,) = ax.plot([], [], [], ls="none", marker="o", ms=5,
                                    mfc=c["target"], mec="none", alpha=0.35)
        (self.ln_target,) = ax.plot([], [], [], ls="none", marker="X", ms=13,
                                    mfc=c["target"], mec=c["bg"], mew=1.2)

        # Hover halo
        (self.ln_halo,) = ax.plot([], [], [], ls="none", marker="o", ms=20,
                                  mfc="none", mec=c["fg"], mew=1.2, alpha=0.5)

    def _build_world_axes(self, length=0.34, ticks=(0.1, 0.2, 0.3), tick_len=0.008):
        """World x / y / z axes from the origin, with labels and 10 cm ticks (meters)."""
        c, ax = self.c, self.ax
        tick_style = dict(fontsize=7.5, color=c["muted"], ha="center", va="center")
        for k, (name, col) in enumerate(zip("xyz", (c["x"], c["y"], c["z"]))):
            e = np.eye(3)[k]
            # Negative half (floor axes only), faint dashed
            if k < 2:
                neg = -e * length
                ax.plot([0, neg[0]], [0, neg[1]], [0, neg[2]], color=c["grid"], lw=0.9, ls=(0, (3, 3)))
            # Positive half, colored, with an end dot and a label
            end = e * length
            ax.plot([0, end[0]], [0, end[1]], [0, end[2]], color=col, lw=1.8, alpha=0.9)
            ax.plot([end[0]], [end[1]], [end[2]], ls="none", marker="o", ms=4, mfc=col, mec=col)
            lbl = e * (length + 0.03)
            ax.text(*lbl, name, color=col, fontsize=12, weight="bold", ha="center", va="center")
            # Ticks: short marks perpendicular to the axis, plus distance labels
            side = np.eye(3)[1] if k == 0 else np.eye(3)[0]   # x ticks along y; y, z ticks along x
            for t in ticks:
                p = e * t
                a, b = p - side * tick_len, p + side * tick_len
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=col, lw=1.2, alpha=0.9)
                q = p - side * 0.03
                ax.text(*q, f"{t:.1f}", **tick_style)

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

    def _project_to_screen(self, pt3d):
        x2, y2, _ = proj3d.proj_transform(pt3d[0], pt3d[1], pt3d[2], self.ax.get_proj())
        return self.ax.transData.transform((x2, y2))

    def _pick(self, mouse_px):
        """'target', a joint index 0..4, or None."""
        if np.linalg.norm(mouse_px - self._project_to_screen(self.target)) < self.PICK_RADIUS_PX + 2:
            return "target"
        best, best_d = None, self.PICK_RADIUS_PX
        for j in range(N_JOINTS):
            d = np.linalg.norm(mouse_px - self._project_to_screen(self.positions[j + 1]))
            if d < best_d:
                best, best_d = j, d
        return best

    def _button_hit(self, event):
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
            self.run_ik()
            return
        if event.inaxes != self.ax:
            return
        hit = self._pick(np.array([event.x, event.y]))
        if hit == "target":
            self.dragging = True
            self.last_mouse_xy = np.array([event.x, event.y])
            self.ax.disable_mouse_rotation()
            self._set_cursor("MOVE")
        elif hit is not None:
            self.selected = hit       # click a joint to select it
            self.render()

    def _on_motion(self, event):
        if event.x is None:
            return
        curr_px = np.array([event.x, event.y])

        if not self.dragging:
            hit = self._pick(curr_px) if event.inaxes == self.ax else None
            self._set_cursor("HAND" if (hit is not None or self._button_hit(event)) else "POINTER")
            if hit != self.hover:
                self.hover = hit
                self.render()
            return

        delta_px = curr_px - self.last_mouse_xy
        self.last_mouse_xy = curr_px

        # Map 2D screen drag onto camera right & up vectors in world space
        azim, elev = np.deg2rad(self.ax.azim), np.deg2rad(self.ax.elev)
        cam_right = np.array([-np.sin(azim), np.cos(azim), 0.0])
        cam_up = np.array([-np.cos(azim) * np.sin(elev), -np.sin(azim) * np.sin(elev), np.cos(elev)])
        scale = (2 * LIMIT * self.zoom) / max(self.ax.bbox.width, 1) * 1.35
        self.target += (delta_px[0] * cam_right + delta_px[1] * cam_up) * scale
        self.target[2] = max(self.target[2], 0.0)
        self.render()

    def _on_release(self, _):
        if self.dragging:
            self.dragging = False
            self.ax.mouse_init()
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
        elif key == "up":
            self.selected = (self.selected - 1) % N_JOINTS
            self.render()
        elif key == "down":
            self.selected = (self.selected + 1) % N_JOINTS
            self.render()
        elif key in ("left", "right", "shift+left", "shift+right"):
            step = FINE_STEP_DEG if key.startswith("shift") else STEP_DEG
            sign = 1 if key.endswith("right") else -1
            self.q[self.selected] += sign * np.deg2rad(step)
            self.update_pose()
        elif key == "0":
            self.q[:] = 0.0
            self.update_pose()
        elif key == "r":
            self.q[:] = 0.0
            self.target = HOME_TARGET.copy()
            self.zoom = 1.0
            self.ax.view_init(*VIEWS["1"])
            self._apply_limits()
            self.update_pose()
        elif key in (" ", "enter"):
            self.run_ik()

    # ── Drawing ──────────────────────────────────────────────────────────────

    def render(self):
        c, p, tgt = self.c, self.positions, self.target
        tip = p[-1]

        self.ln_links.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
        self.ln_joints.set_data_3d(p[1:-1, 0], p[1:-1, 1], p[1:-1, 2])
        self.ln_tip.set_data_3d([tip[0]], [tip[1]], [tip[2]])
        sel = p[self.selected + 1]
        self.ln_selected.set_data_3d([sel[0]], [sel[1]], [sel[2]])

        axis_len = 0.025
        for pos, rot, lines in zip(p[1:-1], self.rotations, self.ln_frames):
            for i, ln in enumerate(lines):
                end = pos + rot[:, i] * axis_len
                ln.set_data_3d([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]])

        self.ln_target.set_data_3d([tgt[0]], [tgt[1]], [tgt[2]])
        self.ln_shadow.set_data_3d([tgt[0]], [tgt[1]], [0.0])
        self.ln_drop.set_data_3d([tgt[0], tgt[0]], [tgt[1], tgt[1]], [0.0, tgt[2]])
        self.ln_err.set_data_3d([tip[0], tgt[0]], [tip[1], tgt[1]], [tip[2], tgt[2]])

        if self.dragging:
            active = "target"
        else:
            active = self.hover
        if active is None:
            self.ln_halo.set_data_3d([], [], [])
        else:
            pt = tgt if active == "target" else p[active + 1]
            self.ln_halo.set_data_3d([pt[0]], [pt[1]], [pt[2]])

        # Text
        for j, txt in enumerate(self.txt_joints):
            is_sel = j == self.selected
            txt.set_text(f"{'›' if is_sel else ' '} {JOINT_NAMES[j]:<18}{np.rad2deg(self.q[j]):+7.1f}°")
            txt.set_color(c["fg"] if is_sel else c["muted"])
            txt.set_weight("bold" if is_sel else "normal")

        err_mm = np.linalg.norm(tgt - tip) * 1000.0
        self.txt_tip.set_text(f"p_end   {tip[0]:+.3f}  {tip[1]:+.3f}  {tip[2]:+.3f} m")
        self.txt_target.set_text(f"target  {tgt[0]:+.3f}  {tgt[1]:+.3f}  {tgt[2]:+.3f} m")
        self.txt_err.set_text(f"error  {err_mm:7.1f} mm")
        self.txt_err.set_color(c["tip"] if err_mm < 2.0 else c["muted"])
        self.txt_status.set_text(self.status)

        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    ArmVisualizer()
    plt.show()
