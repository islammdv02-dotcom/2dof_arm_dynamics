import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Project setup
# ============================================================

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# ============================================================
# Arm parameters
# ============================================================

l1 = 0.30  # upper arm length [m]
l2 = 0.30  # forearm length [m]

m1 = 1.5   # upper arm mass [kg]
m2 = 1.0   # forearm mass [kg]



# Center of mass distances from each joint
lc1 = l1 / 2.0
lc2 = l2 / 2.0

# Moments of inertia for uniform slender rods about their centers of mass
I1 = (1.0 / 12.0) * m1 * l1**2
I2 = (1.0 / 12.0) * m2 * l2**2

# Gravity choice
USE_GRAVITY = False
g = 9.81


# ============================================================
# Movement parameters
# ============================================================

T = 1.0       # movement duration [s]
dt = 0.001    # timestep [s]
t = np.arange(0, T + dt, dt)


# ============================================================
# Target grid
# ============================================================

x_vals = np.array([0.27, 0.35, 0.43])
y_vals = np.array([-0.08, 0.00, 0.08])

targets = np.array([[x, y] for y in y_vals for x in x_vals])


# ============================================================
# Forward kinematics
# ============================================================

def forward_kinematics(q1, q2, l1=0.30, l2=0.30):
    """
    Given shoulder angle q1 and elbow angle q2,
    compute endpoint position x, y.
    """
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

    return x, y


# ============================================================
# Inverse kinematics
# ============================================================

def inverse_kinematics(x, y, l1=0.30, l2=0.30, elbow="down"):
    """
    Given endpoint position x, y,
    compute shoulder angle q1 and elbow angle q2.

    We use one consistent elbow configuration.
    """
    r2 = x**2 + y**2

    cos_q2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)

    if elbow == "down":
        q2 = np.arccos(cos_q2)
    elif elbow == "up":
        q2 = -np.arccos(cos_q2)
    else:
        raise ValueError("elbow must be either 'down' or 'up'")

    q1 = np.arctan2(y, x) - np.arctan2(
        l2 * np.sin(q2),
        l1 + l2 * np.cos(q2)
    )

    return q1, q2


# ============================================================
# Minimum-jerk trajectory
# ============================================================

def minimum_jerk_profile(t, T):
    """
    Minimum-jerk scalar trajectory.

    s(0) = 0
    s(T) = 1

    This creates smooth motion with zero velocity
    at the beginning and end.
    """
    tau = t / T
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5

    return s


def generate_endpoint_trajectory(start_point, end_point, t, T):
    """
    Generate a straight-line endpoint trajectory
    using minimum-jerk timing.
    """
    start_point = np.array(start_point)
    end_point = np.array(end_point)

    s = minimum_jerk_profile(t, T)

    x = start_point[0] + s * (end_point[0] - start_point[0])
    y = start_point[1] + s * (end_point[1] - start_point[1])

    return x, y, s


def compute_endpoint_speed(x, y, dt):
    """
    Compute endpoint speed from x(t), y(t).
    """
    x_dot = np.gradient(x, dt)
    y_dot = np.gradient(y, dt)

    speed = np.sqrt(x_dot**2 + y_dot**2)

    return speed


# ============================================================
# Plot target grid
# ============================================================

def plot_target_grid():
    plt.figure(figsize=(5, 5))
    plt.scatter(targets[:, 0], targets[:, 1], s=80)

    for i, (x, y) in enumerate(targets):
        plt.text(x + 0.005, y + 0.005, str(i + 1))

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("3 x 3 Target Grid")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_grid.png", dpi=300)
    plt.show()


# ============================================================
# Plot one example endpoint movement
# ============================================================

def plot_example_endpoint_trajectory():
    """
    Plot one example movement: Target 1 to Target 9.
    """
    start = targets[0]
    end = targets[8]

    x, y, s = generate_endpoint_trajectory(start, end, t, T)
    speed = compute_endpoint_speed(x, y, dt)

    plt.figure(figsize=(5, 5))
    plt.plot(x, y, linewidth=2, label="Minimum-jerk path")
    plt.scatter(targets[:, 0], targets[:, 1], s=60, label="Targets")
    plt.scatter(start[0], start[1], s=120, marker="o", label="Start")
    plt.scatter(end[0], end[1], s=120, marker="x", label="End")

    for i, (x_target, y_target) in enumerate(targets):
        plt.text(x_target + 0.005, y_target + 0.005, str(i + 1))

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Example Straight-Line Endpoint Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "example_endpoint_trajectory.png", dpi=300)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(t, speed, linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Endpoint speed [m/s]")
    plt.title("Bell-Shaped Endpoint Speed Profile")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "example_speed_profile.png", dpi=300)
    plt.show()


# ============================================================
# Plot all 72 endpoint movements
# ============================================================

def plot_all_endpoint_trajectories():
    """
    Plot all ordered target-to-target movements.
    There are 9 targets and 8 possible destinations from each target.

    Total:
    9 * 8 = 72 directed movements.
    """
    movement_count = 0

    plt.figure(figsize=(6, 6))

    for i, start in enumerate(targets):
        for j, end in enumerate(targets):
            if i == j:
                continue

            x, y, s = generate_endpoint_trajectory(start, end, t, T)
            plt.plot(x, y, linewidth=0.8, alpha=0.6)

            movement_count += 1

    plt.scatter(targets[:, 0], targets[:, 1], s=80, zorder=5)

    for i, (x_target, y_target) in enumerate(targets):
        plt.text(x_target + 0.005, y_target + 0.005, str(i + 1))

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"All {movement_count} Directed Endpoint Trajectories")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "all_72_endpoint_trajectories.png", dpi=300)
    plt.show()

    print(f"\nTotal number of directed movements: {movement_count}")


# ============================================================
# Test inverse kinematics
# ============================================================

def test_inverse_kinematics():
    print("\nInverse kinematics test:")

    for i, (x, y) in enumerate(targets):
        q1, q2 = inverse_kinematics(x, y)

        x_check, y_check = forward_kinematics(q1, q2)

        error = np.sqrt((x - x_check)**2 + (y - y_check)**2)

        print(
            f"Target {i + 1}: "
            f"q1 = {np.rad2deg(q1):7.2f} deg, "
            f"q2 = {np.rad2deg(q2):7.2f} deg, "
            f"error = {error:.2e} m"
        )


# ============================================================
# Singularity check
# ============================================================

def check_singularities_all_movements(epsilon=1e-3):
    """
    Check Jacobian determinant for all movements.

    det(J) = l1 * l2 * sin(q2)

    The arm is singular when q2 is near 0 or pi.
    """
    min_abs_detJ = np.inf
    worst_movement = None

    for i, start in enumerate(targets):
        for j, end in enumerate(targets):
            if i == j:
                continue

            x, y, s = generate_endpoint_trajectory(start, end, t, T)

            q2_values = []

            for x_k, y_k in zip(x, y):
                q1, q2 = inverse_kinematics(x_k, y_k)
                q2_values.append(q2)

            q2_values = np.array(q2_values)

            detJ = l1 * l2 * np.sin(q2_values)
            movement_min = np.min(np.abs(detJ))

            if movement_min < min_abs_detJ:
                min_abs_detJ = movement_min
                worst_movement = (i + 1, j + 1)

    print("\nSingularity check:")
    print(f"Minimum |det(J)| across all movements: {min_abs_detJ:.6f}")
    print(f"Worst movement: Target {worst_movement[0]} -> Target {worst_movement[1]}")

    if min_abs_detJ > epsilon:
        print(f"PASS: all movements satisfy |det(J)| > {epsilon}")
    else:
        print(f"WARNING: some movement is close to singularity, |det(J)| <= {epsilon}")



# ============================================================
# Joint trajectory from endpoint trajectory
# ============================================================

def compute_joint_trajectory_from_endpoint(x, y, dt):
    """
    Convert endpoint trajectory x(t), y(t)
    into joint angles q1(t), q2(t), joint velocities, and accelerations.

    We use inverse kinematics at every time step.
    Then we use numerical differentiation to get qdot and qddot.
    """

    q1_values = []
    q2_values = []

    for x_k, y_k in zip(x, y):
        q1, q2 = inverse_kinematics(x_k, y_k, elbow="down")
        q1_values.append(q1)
        q2_values.append(q2)

    q1_values = np.array(q1_values)
    q2_values = np.array(q2_values)

    # Unwrap angles to avoid artificial jumps near +/- pi
    q1_values = np.unwrap(q1_values)
    q2_values = np.unwrap(q2_values)

    q = np.column_stack((q1_values, q2_values))

    # Numerical derivatives
    qdot = np.gradient(q, dt, axis=0)
    qddot = np.gradient(qdot, dt, axis=0)

    return q, qdot, qddot


# ============================================================
# Plot example joint kinematics
# ============================================================

def plot_example_joint_kinematics():
    """
    For one example movement, compute and plot:
    q(t), qdot(t), qddot(t).

    Example movement: Target 1 to Target 9.
    """

    start = targets[0]
    end = targets[8]

    x, y, s = generate_endpoint_trajectory(start, end, t, T)

    q, qdot, qddot = compute_joint_trajectory_from_endpoint(x, y, dt)

    q1 = q[:, 0]
    q2 = q[:, 1]

    q1_dot = qdot[:, 0]
    q2_dot = qdot[:, 1]

    q1_ddot = qddot[:, 0]
    q2_ddot = qddot[:, 1]

    # -------------------------
    # Joint angles
    # -------------------------

    plt.figure(figsize=(7, 4))
    plt.plot(t, np.rad2deg(q1), linewidth=2, label="q1 shoulder")
    plt.plot(t, np.rad2deg(q2), linewidth=2, label="q2 elbow")

    plt.xlabel("Time [s]")
    plt.ylabel("Joint angle [deg]")
    plt.title("Example Joint Angle Trajectories")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "example_joint_angles.png", dpi=300)
    plt.show()

    # -------------------------
    # Joint velocities
    # -------------------------

    plt.figure(figsize=(7, 4))
    plt.plot(t, q1_dot, linewidth=2, label="q1_dot shoulder")
    plt.plot(t, q2_dot, linewidth=2, label="q2_dot elbow")

    plt.xlabel("Time [s]")
    plt.ylabel("Joint velocity [rad/s]")
    plt.title("Example Joint Velocity Trajectories")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "example_joint_velocities.png", dpi=300)
    plt.show()

    # -------------------------
    # Joint accelerations
    # -------------------------

    plt.figure(figsize=(7, 4))
    plt.plot(t, q1_ddot, linewidth=2, label="q1_ddot shoulder")
    plt.plot(t, q2_ddot, linewidth=2, label="q2_ddot elbow")

    plt.xlabel("Time [s]")
    plt.ylabel("Joint acceleration [rad/s²]")
    plt.title("Example Joint Acceleration Trajectories")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "example_joint_accelerations.png", dpi=300)
    plt.show()

    print("\nExample joint kinematics:")
    print(f"Movement: Target 1 -> Target 9")
    print(f"Initial q1 = {np.rad2deg(q1[0]):.2f} deg")
    print(f"Initial q2 = {np.rad2deg(q2[0]):.2f} deg")
    print(f"Final q1   = {np.rad2deg(q1[-1]):.2f} deg")
    print(f"Final q2   = {np.rad2deg(q2[-1]):.2f} deg")



# ============================================================
# Arm dynamics
# ============================================================

def inertia_matrix(q):
    """
    Compute the 2x2 inertia matrix M(q).

    q[0] = q1 shoulder angle
    q[1] = q2 elbow angle

    Model:
    - two uniform rigid links
    - center of mass at half of each link
    - moments of inertia I1 and I2
    """
    q1, q2 = q

    M11 = (
        I1
        + I2
        + m1 * lc1**2
        + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(q2))
    )

    M12 = I2 + m2 * (lc2**2 + l1 * lc2 * np.cos(q2))

    M22 = I2 + m2 * lc2**2

    M = np.array([
        [M11, M12],
        [M12, M22]
    ])

    return M


def coriolis_centrifugal_vector(q, qdot):
    """
    Compute C(q, qdot) qdot.

    This vector contains Coriolis and centrifugal effects.
    """
    q1, q2 = q
    q1_dot, q2_dot = qdot

    h = m2 * l1 * lc2 * np.sin(q2)

    c1 = -h * (2 * q1_dot * q2_dot + q2_dot**2)
    c2 = h * q1_dot**2

    return np.array([c1, c2])


def gravity_vector(q):
    """
    Compute gravity vector G(q).

    For this project we neglect gravity by default.
    This corresponds to a horizontal-plane arm.
    """
    if not USE_GRAVITY:
        return np.array([0.0, 0.0])

    q1, q2 = q

    G1 = (
        (m1 * lc1 + m2 * l1) * g * np.cos(q1)
        + m2 * lc2 * g * np.cos(q1 + q2)
    )

    G2 = m2 * lc2 * g * np.cos(q1 + q2)

    return np.array([G1, G2])


def inverse_dynamics(q, qdot, qddot):
    """
    Compute joint torque:

    tau = M(q) qddot + C(q, qdot) qdot + G(q)
    """
    M = inertia_matrix(q)
    Cqd = coriolis_centrifugal_vector(q, qdot)
    G = gravity_vector(q)

    tau = M @ qddot + Cqd + G

    return tau


def compute_torque_trajectory(q, qdot, qddot):
    """
    Compute torque trajectory for all time steps.
    """
    tau_values = []

    for q_k, qdot_k, qddot_k in zip(q, qdot, qddot):
        tau_k = inverse_dynamics(q_k, qdot_k, qddot_k)
        tau_values.append(tau_k)

    tau_values = np.array(tau_values)

    return tau_values



# ============================================================
# Plot example torque trajectory
# ============================================================

def plot_example_joint_torques():
    """
    Compute and plot inverse dynamics torques
    for one example movement.

    Example movement: Target 1 to Target 9.
    """
    start = targets[0]
    end = targets[8]

    x, y, s = generate_endpoint_trajectory(start, end, t, T)

    q, qdot, qddot = compute_joint_trajectory_from_endpoint(x, y, dt)

    tau = compute_torque_trajectory(q, qdot, qddot)

    tau1 = tau[:, 0]
    tau2 = tau[:, 1]

    plt.figure(figsize=(7, 4))
    plt.plot(t, tau1, linewidth=2, label="tau1 shoulder")
    plt.plot(t, tau2, linewidth=2, label="tau2 elbow")

    plt.xlabel("Time [s]")
    plt.ylabel("Joint torque [Nm]")
    plt.title("Example Inverse Dynamics Joint Torques")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "example_joint_torques.png", dpi=300)
    plt.show()

    print("\nExample inverse dynamics torques:")
    print("Movement: Target 1 -> Target 9")
    print(f"Max |tau1| = {np.max(np.abs(tau1)):.4f} Nm")
    print(f"Max |tau2| = {np.max(np.abs(tau2)):.4f} Nm")



# ============================================================
# Main script
# ============================================================

if __name__ == "__main__":
    print("2-DOF arm reaching project started.")
    print(f"Number of targets: {len(targets)}")
    print(targets)

    plot_target_grid()
    test_inverse_kinematics()

    plot_example_endpoint_trajectory()
    plot_all_endpoint_trajectories()
    check_singularities_all_movements()

    plot_example_joint_kinematics()
    plot_example_joint_torques()

    print("\nFinished.")
    print("Check the 'figures' folder. It should now contain:")
    print("1. target_grid.png")
    print("2. example_endpoint_trajectory.png")
    print("3. example_speed_profile.png")
    print("4. all_72_endpoint_trajectories.png")
    print("5. example_joint_angles.png")
    print("6. example_joint_velocities.png")
    print("7. example_joint_accelerations.png")
    print("8. example_joint_torques.png")



    