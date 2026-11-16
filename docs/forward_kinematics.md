# Forward Kinematics — step by step

**Goal:** given joint angles $q$, compute every joint position $p_i$, every joint orientation $R_i$, and the gripper tip $p_{\text{end}}$.

$$
q = \begin{bmatrix} q_0 & q_1 & q_2 & q_3 & q_4 \end{bmatrix} \quad \text{(radians)}
\qquad\longrightarrow\qquad
\text{positions} \in \mathbb{R}^{7\times 3},\;\; \text{rotations} = [R_0,\dots,R_4],\; R_i \in \mathbb{R}^{3\times 3}
$$

---

## Step 1 — The three basic rotation matrices

Rotation by angle $\theta$ about a single axis ($c = \cos\theta,\ s = \sin\theta$):

$$
R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & c & -s \\ 0 & s & c \end{bmatrix}
\qquad
R_y(\theta) = \begin{bmatrix} c & 0 & s \\ 0 & 1 & 0 \\ -s & 0 & c \end{bmatrix}
\qquad
R_z(\theta) = \begin{bmatrix} c & -s & 0 \\ s & c & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

```python
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
```

---

## Step 2 — Which axis each joint turns about

| Joint | Name | Axis | Basic rotation |
|---|---|---|---|
| J0 | base yaw | $z$ | $R_z(q_0)$ |
| J1 | shoulder pitch | $y$ | $R_y(q_1)$ |
| J2 | elbow pitch | $y$ | $R_y(q_2)$ |
| J3 | wrist pitch | $y$ | $R_y(q_3)$ |
| J4 | wrist roll | $x$ | $R_x(q_4)$ |

```python
axis_rot = [rot_z, rot_y, rot_y, rot_y, rot_x]
```

---

## Step 3 — The link offsets (the joint lengths)

Each offset $d_i$ is a **fixed** vector, written in the frame of the joint *before* it. Only its direction in the world changes; its length never does.

$$
d_{\text{base}} = \begin{bmatrix} 0 \\ 0 \\ D_1 \end{bmatrix}
\quad
d_0 = \begin{bmatrix} A_1 \\ 0 \\ 0 \end{bmatrix}
\quad
d_1 = \begin{bmatrix} L_1 \\ 0 \\ 0 \end{bmatrix}
\quad
d_2 = \begin{bmatrix} L_2 \\ 0 \\ 0 \end{bmatrix}
\quad
d_3 = \begin{bmatrix} L_3 \\ 0 \\ 0 \end{bmatrix}
\quad
d_4 = \begin{bmatrix} L_{\text{grip}} \\ 0 \\ 0 \end{bmatrix}
$$

- $d_{\text{base}}$: base plate → J0
- $d_i$: joint $i$ → next joint (and $d_4$: J4 → gripper tip)

```python
d_base = np.array([0.0, 0.0, D1])
d = [
    np.array([A1,     0.0, 0.0]),   # J0 -> J1
    np.array([L1,     0.0, 0.0]),   # J1 -> J2
    np.array([L2,     0.0, 0.0]),   # J2 -> J3
    np.array([L3,     0.0, 0.0]),   # J3 -> J4
    np.array([L_GRIP, 0.0, 0.0]),   # J4 -> tip
]
```

---

## Step 4 — Start at the base

$$
p_{\text{base}} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix},
\qquad
R_{-1} = I_{3\times 3}
$$

$R_{-1}$ is "the orientation before any joint has turned" — just the world frame.

```python
positions = np.zeros((7, 3))   # [0] base, [1..5] J0..J4, [6] tip
rotations = []
R = np.eye(3)
```

---

## Step 5 — Go up to J0

J0 hasn't rotated anything yet, so the offset goes straight up:

$$
p_{J0} = p_{\text{base}} + d_{\text{base}} = \begin{bmatrix} 0 \\ 0 \\ D_1 \end{bmatrix}
$$

```python
positions[1] = positions[0] + d_base
```

---

## Step 6 — The chain loop (the core of FK)

For each joint $i = 0, 1, 2, 3, 4$, do two things:

**6a. Accumulate the orientation.** Joint $i$ rotates about its *own* axis, on top of everything before it:

$$
\boxed{\,R_i = R_{i-1}\; R_{\text{axis}_i}(q_i)\,}
$$

**6b. Step to the next joint.** Take the fixed offset, rotate it into the world, add it on:

$$
\boxed{\,p_{i+1} = p_i + R_i\, d_i\,}
$$

```python
for i in range(5):
    R = R @ axis_rot[i](q[i])                       # 6a
    rotations.append(R)
    positions[i + 2] = positions[i + 1] + R @ d[i]  # 6b
```

Written out in full, the loop computes:

$$
\begin{aligned}
R_0 &= R_z(q_0) \\
R_1 &= R_z(q_0)\,R_y(q_1) \\
R_2 &= R_z(q_0)\,R_y(q_1)\,R_y(q_2) \\
R_3 &= R_z(q_0)\,R_y(q_1)\,R_y(q_2)\,R_y(q_3) \\
R_4 &= R_z(q_0)\,R_y(q_1)\,R_y(q_2)\,R_y(q_3)\,R_x(q_4)
\end{aligned}
$$

$$
p_{\text{end}} = p_{J0} + R_0 d_0 + R_1 d_1 + R_2 d_2 + R_3 d_3 + R_4 d_4
$$

### ⚠️ Why multiply on the **right**?

- $R_{i-1}\,R_{\text{axis}}(q_i)$ → rotate about the joint's **local** axis (the axis that has already been turned by earlier joints). ✅ This is what a physical joint does.
- $R_{\text{axis}}(q_i)\,R_{i-1}$ → rotate about the **world** axis. ❌ Most common FK bug: the elbow would always pitch about world $y$ even after the base has yawed.

---

## Step 7 — Return

```python
return positions, rotations
```

- `positions[6]` is $p_{\text{end}}$.
- The third column of $R_i$ is the joint's local $z$-axis, `rotations[i][:, 2]`; in general, **the joint's rotation axis in world coordinates** is the column matching its axis: $z$ for J0, $y$ for J1–J3, $x$ for J4. You'll need this for the Jacobian:

$$
\hat a_0 = R_0[:,2],\quad \hat a_{1,2,3} = R_{1,2,3}[:,1],\quad \hat a_4 = R_4[:,0]
$$

---

## Step 8 — Check your answer

Hand-computed values (with $D_1 = 0.062$, $A_1 = 0.030$, $L_1 = 0.116$, $L_2 = 0.135$, $L_3 = 0.060$, $L_{\text{grip}} = 0.055$).
Horizontal reach at home: $A_1 + L_1 + L_2 + L_3 + L_{\text{grip}} = 0.396$.

| $q$ (degrees) | Expected $p_{\text{end}}$ | What it tests |
|---|---|---|
| $(0, 0, 0, 0, 0)$ | $(0.396,\ 0,\ 0.062)$ | All $R = I$, offsets just add up |
| $(90, 0, 0, 0, 0)$ | $(0,\ 0.396,\ 0.062)$ | Base yaw swings arm to $+y$, height unchanged |
| $(0, 90, 0, 0, 0)$ | $(0.030,\ 0,\ -0.304)$ | Shoulder pitch: arm points **down** (see sign note) |
| $(0, -90, 0, 0, 0)$ | $(0.030,\ 0,\ 0.428)$ | Arm points straight **up** |
| $(0, -90, 90, 0, 0)$ | $(0.280,\ 0,\ 0.178)$ | "L" shape: upper arm up, forearm forward again |
| $(0, 0, 0, 0, \text{anything})$ | $(0.396,\ 0,\ 0.062)$ | Wrist roll spins about the link, **tip doesn't move** (only the triad spins) |

```python
q = np.deg2rad([0, -90, 90, 0, 0])
positions, rotations = forward_kinematics(q)
assert np.allclose(positions[6], [0.280, 0.0, 0.178], atol=1e-3)
```

### Sign note

With the standard right-handed $R_y$, a **positive** pitch tips $+x$ toward $-z$:

$$
R_y(\theta)\begin{bmatrix}1\\0\\0\end{bmatrix} = \begin{bmatrix}\cos\theta\\0\\-\sin\theta\end{bmatrix}
$$

So $q_1 > 0$ points the arm **down**. If you'd rather have positive = up, use $R_y(-q_i)$ for the pitch joints. Either is fine — just pick one and match it to the real servos later.

### Visual checks in the sandbox

Run `.venv/bin/python visualizer.py`, then:

1. At $q = 0$, the readout `p_end` must be `+0.396 +0.000 +0.062`.
2. Select J0, press ←/→: whole arm swings around the vertical; `p_end` $z$ stays $0.062$.
3. Select J1: arm tilts up/down as one piece.
4. Bend J1 then J2: the elbow bends relative to the upper arm, not relative to the world.
5. Select J4: arm doesn't move at all, only the little axis triad at J4 spins.
6. The RGB triads at each joint must turn *with* their links. If they don't, check Step 6a.

---

## Appendix — same thing with 4×4 homogeneous transforms

Many textbooks pack $R$ and $p$ into one matrix:

$$
T = \begin{bmatrix} R & p \\ 0\;0\;0 & 1 \end{bmatrix} \in \mathbb{R}^{4\times4}
$$

and chain $T_{i} = T_{i-1}\; \text{Rot}(q_i)\; \text{Trans}(d_i)$. It computes exactly the same numbers as Step 6 — the 3×3 version above is just easier to read and debug. Switch to 4×4 later if you want DH parameters or a URDF-style chain.
