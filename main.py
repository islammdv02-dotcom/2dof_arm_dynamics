import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



# 2-DOF arm reaching simulation
# I kept this file intentionally as one script so it is easy
# to run, inspect, and review. After running it one time, it will generate all the figures in the figures/ folder.



# Basic project setup


figures_folder = Path("figures")
figures_folder.mkdir(exist_ok=True)


# Arm parameters

upper_arm_length = 0.30     # l1 [m]
forearm_length = 0.30       # l2 [m]

upper_arm_mass = 1.5        # m1 [kg]
forearm_mass = 1.0          # m2 [kg]

# I assume each link is a simple uniform rod.
# Not a perfect human arm, but good enough for this project.
upper_arm_com = upper_arm_length / 2.0
forearm_com = forearm_length / 2.0

upper_arm_inertia = (1.0 / 12.0) * upper_arm_mass * upper_arm_length**2
forearm_inertia = (1.0 / 12.0) * forearm_mass * forearm_length**2

# Gravity is optional in the assignment. I keep it off and treat the arm
# as moving in a horizontal plane, like sliding on a table.
use_gravity = False
gravity_acceleration = 9.81



# Timing parameters

movement_duration = 1.0     # T [s]
time_step = 0.001           # dt [s]
time_points = np.arange(0, movement_duration + time_step, time_step)



# Target grid


# 9 targets arranged in a small reachable square in front of the shoulder.
x_grid_values = np.array([0.27, 0.35, 0.43])
y_grid_values = np.array([-0.08, 0.00, 0.08])

target_points = np.array([[x, y] for y in y_grid_values for x in x_grid_values])


# Kinematics


def forward_kinematics(shoulder_angle, elbow_angle,
                       upper_arm_len=upper_arm_length,
                       forearm_len=forearm_length):
    """
    Convert joint angles into endpoint position.

    shoulder_angle = q1
    elbow_angle    = q2
    """
    x = upper_arm_len * np.cos(shoulder_angle) + forearm_len * np.cos(shoulder_angle + elbow_angle)
    y = upper_arm_len * np.sin(shoulder_angle) + forearm_len * np.sin(shoulder_angle + elbow_angle)

    return x, y


def inverse_kinematics(x, y,
                       upper_arm_len=upper_arm_length,
                       forearm_len=forearm_length,
                       elbow_choice="down"):
    """
    Convert endpoint position into joint angles.

    I use one elbow branch consistently so the arm does not randomly flip
    between two possible postures. No acrobatics today.
    """
    distance_squared = x**2 + y**2

    cos_elbow = (
        distance_squared - upper_arm_len**2 - forearm_len**2
    ) / (2 * upper_arm_len * forearm_len)

    # Small numerical safety net. Sometimes floating point gives 1.0000000002.
    cos_elbow = np.clip(cos_elbow, -1.0, 1.0)

    if elbow_choice == "down":
        elbow_angle = np.arccos(cos_elbow)
    elif elbow_choice == "up":
        elbow_angle = -np.arccos(cos_elbow)
    else:
        raise ValueError("elbow_choice must be either 'down' or 'up'")

    shoulder_angle = np.arctan2(y, x) - np.arctan2(
        forearm_len * np.sin(elbow_angle),
        upper_arm_len + forearm_len * np.cos(elbow_angle)
    )

    return shoulder_angle, elbow_angle



# Endpoint trajectory generation


def minimum_jerk_profile(time_array, total_time):
    """
    Smooth scalar motion profile from 0 to 1.

    This is the reason the endpoint starts gently, speeds up in the middle,
    and slows down again at the end.
    """
    normalized_time = time_array / total_time
    progress = 10 * normalized_time**3 - 15 * normalized_time**4 + 6 * normalized_time**5

    return progress


def generate_endpoint_trajectory(start_point, end_point, time_array, total_time):
    """
    Build a straight-line endpoint path from one target to another.
    The path is straight; the timing is smooth.
    """
    start_point = np.array(start_point)
    end_point = np.array(end_point)

    progress = minimum_jerk_profile(time_array, total_time)

    x_path = start_point[0] + progress * (end_point[0] - start_point[0])
    y_path = start_point[1] + progress * (end_point[1] - start_point[1])

    return x_path, y_path, progress


def compute_endpoint_speed(x_path, y_path, dt):
    """Compute endpoint speed from the x and y coordinates."""
    x_velocity = np.gradient(x_path, dt)
    y_velocity = np.gradient(y_path, dt)

    speed = np.sqrt(x_velocity**2 + y_velocity**2)

    return speed



# Plotting target grid and endpoint paths


def plot_target_grid():
    """Plot the 3 x 3 grid of targets."""
    plt.figure(figsize=(5, 5))
    plt.scatter(target_points[:, 0], target_points[:, 1], s=80)

    for target_number, (x, y) in enumerate(target_points, start=1):
        plt.text(x + 0.005, y + 0.005, str(target_number))

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("3 x 3 Target Grid")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_folder / "target_grid.png", dpi=300)
    plt.show()


def plot_example_endpoint_trajectory():
    """Plot one example reach: Target 1 to Target 9."""
    start_target = target_points[0]
    end_target = target_points[8]

    x_path, y_path, _ = generate_endpoint_trajectory(
        start_target,
        end_target,
        time_points,
        movement_duration
    )
    speed = compute_endpoint_speed(x_path, y_path, time_step)

    plt.figure(figsize=(5, 5))
    plt.plot(x_path, y_path, linewidth=2, label="Minimum-jerk path")
    plt.scatter(target_points[:, 0], target_points[:, 1], s=60, label="Targets")
    plt.scatter(start_target[0], start_target[1], s=120, marker="o", label="Start")
    plt.scatter(end_target[0], end_target[1], s=120, marker="x", label="End")

    for target_number, (x, y) in enumerate(target_points, start=1):
        plt.text(x + 0.005, y + 0.005, str(target_number))

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Example Straight-Line Endpoint Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_folder / "example_endpoint_trajectory.png", dpi=300)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(time_points, speed, linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Endpoint speed [m/s]")
    plt.title("Bell-Shaped Endpoint Speed Profile")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_folder / "example_speed_profile.png", dpi=300)
    plt.show()


def plot_all_endpoint_trajectories():
    """Plot all 72 ordered target-to-target movements."""
    movement_count = 0

    plt.figure(figsize=(6, 6))

    for start_index, start_target in enumerate(target_points):
        for end_index, end_target in enumerate(target_points):
            if start_index == end_index:
                continue

            x_path, y_path, _ = generate_endpoint_trajectory(
                start_target,
                end_target,
                time_points,
                movement_duration
            )

            plt.plot(x_path, y_path, linewidth=0.8, alpha=0.6)
            movement_count += 1

    plt.scatter(target_points[:, 0], target_points[:, 1], s=80, zorder=5)

    for target_number, (x, y) in enumerate(target_points, start=1):
        plt.text(x + 0.005, y + 0.005, str(target_number))

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"All {movement_count} Directed Endpoint Trajectories")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_folder / "all_72_endpoint_trajectories.png", dpi=300)
    plt.show()

    print(f"\nTotal number of directed movements: {movement_count}")



# Basic checks


def test_inverse_kinematics():
    """Check that IK followed by FK gives back the original target."""
    print("\nInverse kinematics test:")

    for target_number, (x, y) in enumerate(target_points, start=1):
        shoulder_angle, elbow_angle = inverse_kinematics(x, y)
        x_check, y_check = forward_kinematics(shoulder_angle, elbow_angle)

        error = np.sqrt((x - x_check)**2 + (y - y_check)**2)

        print(
            f"Target {target_number}: "
            f"q1 = {np.rad2deg(shoulder_angle):7.2f} deg, "
            f"q2 = {np.rad2deg(elbow_angle):7.2f} deg, "
            f"error = {error:.2e} m"
        )


def check_singularities_all_movements(epsilon=1e-3):
    """
    Check whether the arm gets too close to a singular configuration.

    If the elbow is almost fully straight or fully folded, the Jacobian gets
    close to singular. Basically, the arm starts behaving awkwardly.
    """
    smallest_abs_det_jacobian = np.inf
    worst_movement = None

    for start_index, start_target in enumerate(target_points):
        for end_index, end_target in enumerate(target_points):
            if start_index == end_index:
                continue

            x_path, y_path, _ = generate_endpoint_trajectory(
                start_target,
                end_target,
                time_points,
                movement_duration
            )

            elbow_angles = []

            for x_now, y_now in zip(x_path, y_path):
                _, elbow_angle = inverse_kinematics(x_now, y_now)
                elbow_angles.append(elbow_angle)

            elbow_angles = np.array(elbow_angles)
            det_jacobian = upper_arm_length * forearm_length * np.sin(elbow_angles)
            movement_minimum = np.min(np.abs(det_jacobian))

            if movement_minimum < smallest_abs_det_jacobian:
                smallest_abs_det_jacobian = movement_minimum
                worst_movement = (start_index + 1, end_index + 1)

    print("\nSingularity check:")
    print(f"Minimum |det(J)| across all movements: {smallest_abs_det_jacobian:.6f}")
    print(f"Worst movement: Target {worst_movement[0]} -> Target {worst_movement[1]}")

    if smallest_abs_det_jacobian > epsilon:
        print(f"PASS: all movements satisfy |det(J)| > {epsilon}")
    else:
        print(f"WARNING: at least one movement is close to singularity, |det(J)| <= {epsilon}")


# Joint trajectories


def compute_joint_trajectory_from_endpoint(x_path, y_path, dt):
    """
    Convert endpoint motion into joint motion.

    First I run inverse kinematics at every point. Then I use numerical
    differentiation to estimate velocities and accelerations.
    """
    shoulder_angles = []
    elbow_angles = []

    for x_now, y_now in zip(x_path, y_path):
        shoulder_angle, elbow_angle = inverse_kinematics(x_now, y_now, elbow_choice="down")
        shoulder_angles.append(shoulder_angle)
        elbow_angles.append(elbow_angle)

    shoulder_angles = np.unwrap(np.array(shoulder_angles))
    elbow_angles = np.unwrap(np.array(elbow_angles))

    joint_angles = np.column_stack((shoulder_angles, elbow_angles))

    # Derivatives from samples. Tiny numerical noise is expected here.
    joint_velocities = np.gradient(joint_angles, dt, axis=0)
    joint_accelerations = np.gradient(joint_velocities, dt, axis=0)

    return joint_angles, joint_velocities, joint_accelerations


def plot_example_joint_kinematics():
    """Plot q, qdot, and qddot for one example movement."""
    start_target = target_points[0]
    end_target = target_points[8]

    x_path, y_path, _ = generate_endpoint_trajectory(
        start_target,
        end_target,
        time_points,
        movement_duration
    )

    joint_angles, joint_velocities, joint_accelerations = compute_joint_trajectory_from_endpoint(
        x_path,
        y_path,
        time_step
    )

    shoulder_angle = joint_angles[:, 0]
    elbow_angle = joint_angles[:, 1]

    shoulder_velocity = joint_velocities[:, 0]
    elbow_velocity = joint_velocities[:, 1]

    shoulder_acceleration = joint_accelerations[:, 0]
    elbow_acceleration = joint_accelerations[:, 1]

    plt.figure(figsize=(7, 4))
    plt.plot(time_points, np.rad2deg(shoulder_angle), linewidth=2, label="q1 shoulder")
    plt.plot(time_points, np.rad2deg(elbow_angle), linewidth=2, label="q2 elbow")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint angle [deg]")
    plt.title("Example Joint Angle Trajectories")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_folder / "example_joint_angles.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(time_points, shoulder_velocity, linewidth=2, label="q1_dot shoulder")
    plt.plot(time_points, elbow_velocity, linewidth=2, label="q2_dot elbow")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint velocity [rad/s]")
    plt.title("Example Joint Velocity Trajectories")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_folder / "example_joint_velocities.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(time_points, shoulder_acceleration, linewidth=2, label="q1_ddot shoulder")
    plt.plot(time_points, elbow_acceleration, linewidth=2, label="q2_ddot elbow")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint acceleration [rad/s²]")
    plt.title("Example Joint Acceleration Trajectories")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_folder / "example_joint_accelerations.png", dpi=300)
    plt.show()

    print("\nExample joint kinematics:")
    print("Movement: Target 1 -> Target 9")
    print(f"Initial q1 = {np.rad2deg(shoulder_angle[0]):.2f} deg")
    print(f"Initial q2 = {np.rad2deg(elbow_angle[0]):.2f} deg")
    print(f"Final q1   = {np.rad2deg(shoulder_angle[-1]):.2f} deg")
    print(f"Final q2   = {np.rad2deg(elbow_angle[-1]):.2f} deg")


# Dynamics model


def inertia_matrix(joint_angles):
    """Compute the 2 x 2 inertia matrix for the arm."""
    _, elbow_angle = joint_angles

    m11 = (
        upper_arm_inertia
        + forearm_inertia
        + upper_arm_mass * upper_arm_com**2
        + forearm_mass * (
            upper_arm_length**2
            + forearm_com**2
            + 2 * upper_arm_length * forearm_com * np.cos(elbow_angle)
        )
    )

    m12 = forearm_inertia + forearm_mass * (
        forearm_com**2 + upper_arm_length * forearm_com * np.cos(elbow_angle)
    )

    m22 = forearm_inertia + forearm_mass * forearm_com**2

    return np.array([
        [m11, m12],
        [m12, m22]
    ])


def coriolis_centrifugal_vector(joint_angles, joint_velocities):
    """
    Compute the Coriolis/centrifugal vector.

    This is the part that couples the joints together when the arm moves.
    In other words, moving one joint can affect the other one. Fun physics.
    """
    _, elbow_angle = joint_angles
    shoulder_velocity, elbow_velocity = joint_velocities

    coupling_term = forearm_mass * upper_arm_length * forearm_com * np.sin(elbow_angle)

    shoulder_term = -coupling_term * (
        2 * shoulder_velocity * elbow_velocity + elbow_velocity**2
    )
    elbow_term = coupling_term * shoulder_velocity**2

    return np.array([shoulder_term, elbow_term])


def gravity_vector(joint_angles):
    """Compute the gravity vector, or return zero if gravity is switched off."""
    if not use_gravity:
        return np.array([0.0, 0.0])

    shoulder_angle, elbow_angle = joint_angles

    shoulder_gravity = (
        (upper_arm_mass * upper_arm_com + forearm_mass * upper_arm_length)
        * gravity_acceleration
        * np.cos(shoulder_angle)
        + forearm_mass
        * forearm_com
        * gravity_acceleration
        * np.cos(shoulder_angle + elbow_angle)
    )

    elbow_gravity = (
        forearm_mass
        * forearm_com
        * gravity_acceleration
        * np.cos(shoulder_angle + elbow_angle)
    )

    return np.array([shoulder_gravity, elbow_gravity])


def inverse_dynamics(joint_angles, joint_velocities, joint_accelerations):
    """
    Compute the torques needed to produce the desired joint motion.
    """
    mass_matrix = inertia_matrix(joint_angles)
    velocity_terms = coriolis_centrifugal_vector(joint_angles, joint_velocities)
    gravity_terms = gravity_vector(joint_angles)

    joint_torques = mass_matrix @ joint_accelerations + velocity_terms + gravity_terms

    return joint_torques


def compute_torque_trajectory(joint_angles, joint_velocities, joint_accelerations):
    """Compute torque at every time step."""
    torque_values = []

    for angles_now, velocities_now, accelerations_now in zip(
        joint_angles,
        joint_velocities,
        joint_accelerations
    ):
        torque_now = inverse_dynamics(angles_now, velocities_now, accelerations_now)
        torque_values.append(torque_now)

    return np.array(torque_values)


def plot_example_joint_torques():
    """Plot inverse-dynamics torques for one example movement."""
    start_target = target_points[0]
    end_target = target_points[8]

    x_path, y_path, _ = generate_endpoint_trajectory(
        start_target,
        end_target,
        time_points,
        movement_duration
    )

    joint_angles, joint_velocities, joint_accelerations = compute_joint_trajectory_from_endpoint(
        x_path,
        y_path,
        time_step
    )

    torque_profile = compute_torque_trajectory(
        joint_angles,
        joint_velocities,
        joint_accelerations
    )

    shoulder_torque = torque_profile[:, 0]
    elbow_torque = torque_profile[:, 1]

    plt.figure(figsize=(7, 4))
    plt.plot(time_points, shoulder_torque, linewidth=2, label="shoulder torque")
    plt.plot(time_points, elbow_torque, linewidth=2, label="elbow torque")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint torque [Nm]")
    plt.title("Example Inverse Dynamics Joint Torques")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_folder / "example_joint_torques.png", dpi=300)
    plt.show()

    print("\nExample inverse dynamics torques:")
    print("Movement: Target 1 -> Target 9")
    print(f"Max |shoulder torque| = {np.max(np.abs(shoulder_torque)):.4f} Nm")
    print(f"Max |elbow torque|    = {np.max(np.abs(elbow_torque)):.4f} Nm")


# Forward dynamics

def forward_dynamics_acceleration(joint_angles, joint_velocities, joint_torques):
    """Compute acceleration from the current state and torques."""
    mass_matrix = inertia_matrix(joint_angles)
    velocity_terms = coriolis_centrifugal_vector(joint_angles, joint_velocities)
    gravity_terms = gravity_vector(joint_angles)

    # Better than explicitly inverting the matrix. Numpy handles the solve nicely.
    joint_accelerations = np.linalg.solve(
        mass_matrix,
        joint_torques - velocity_terms - gravity_terms
    )

    return joint_accelerations


def simulate_forward_dynamics(torque_profile, initial_angles, initial_velocities,
                              dt, method="semi_implicit"):
    """
    Roll the arm forward in time using the provided torques.

    semi_implicit is the main method used in the report because it is a bit
    more stable than plain explicit Euler.
    """
    number_of_steps = torque_profile.shape[0]

    simulated_angles = np.zeros((number_of_steps, 2))
    simulated_velocities = np.zeros((number_of_steps, 2))
    simulated_accelerations = np.zeros((number_of_steps, 2))

    simulated_angles[0] = initial_angles
    simulated_velocities[0] = initial_velocities

    for step in range(number_of_steps - 1):
        torque_now = torque_profile[step]

        acceleration_now = forward_dynamics_acceleration(
            simulated_angles[step],
            simulated_velocities[step],
            torque_now
        )

        simulated_accelerations[step] = acceleration_now

        if method == "semi_implicit":
            simulated_velocities[step + 1] = simulated_velocities[step] + acceleration_now * dt
            simulated_angles[step + 1] = simulated_angles[step] + simulated_velocities[step + 1] * dt

        elif method == "explicit":
            simulated_angles[step + 1] = simulated_angles[step] + simulated_velocities[step] * dt
            simulated_velocities[step + 1] = simulated_velocities[step] + acceleration_now * dt

        else:
            raise ValueError("method must be 'semi_implicit' or 'explicit'")

    simulated_accelerations[-1] = forward_dynamics_acceleration(
        simulated_angles[-1],
        simulated_velocities[-1],
        torque_profile[-1]
    )

    return simulated_angles, simulated_velocities, simulated_accelerations


def endpoint_from_joint_trajectory(joint_angles):
    """Convert a whole joint trajectory back into endpoint coordinates."""
    x_values = []
    y_values = []

    for angles_now in joint_angles:
        x_now, y_now = forward_kinematics(angles_now[0], angles_now[1])
        x_values.append(x_now)
        y_values.append(y_now)

    return np.array(x_values), np.array(y_values)


def compute_cartesian_error(x_desired, y_desired, x_simulated, y_simulated):
    """Compute RMS and max endpoint error."""
    pointwise_error = np.sqrt(
        (x_desired - x_simulated)**2 + (y_desired - y_simulated)**2
    )

    rms_error = np.sqrt(np.mean(pointwise_error**2))
    max_error = np.max(pointwise_error)

    return pointwise_error, rms_error, max_error


# Forward dynamics verification


def plot_forward_dynamics_verification():
    """
    Use computed torques as input and check whether the arm follows the path.
    This is the main sanity check for the dynamics part.
    """
    start_target = target_points[0]
    end_target = target_points[8]

    x_desired, y_desired, _ = generate_endpoint_trajectory(
        start_target,
        end_target,
        time_points,
        movement_duration
    )

    desired_angles, desired_velocities, desired_accelerations = compute_joint_trajectory_from_endpoint(
        x_desired,
        y_desired,
        time_step
    )

    torque_profile = compute_torque_trajectory(
        desired_angles,
        desired_velocities,
        desired_accelerations
    )

    initial_angles = desired_angles[0]
    initial_velocities = desired_velocities[0]

    simulated_angles, _, _ = simulate_forward_dynamics(
        torque_profile,
        initial_angles,
        initial_velocities,
        time_step,
        method="semi_implicit"
    )

    x_simulated, y_simulated = endpoint_from_joint_trajectory(simulated_angles)

    error, rms_error, max_error = compute_cartesian_error(
        x_desired,
        y_desired,
        x_simulated,
        y_simulated
    )

    plt.figure(figsize=(5, 5))
    plt.plot(x_desired, y_desired, linewidth=2, label="Desired endpoint path")
    plt.plot(x_simulated, y_simulated, "--", linewidth=2, label="Forward dynamics reconstruction")
    plt.scatter(start_target[0], start_target[1], s=100, marker="o", label="Start")
    plt.scatter(end_target[0], end_target[1], s=100, marker="x", label="End")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Forward Dynamics Verification")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_folder / "forward_dynamics_verification.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(time_points, error, linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Endpoint error [m]")
    plt.title("Forward Dynamics Endpoint Reconstruction Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_folder / "forward_dynamics_error_time.png", dpi=300)
    plt.show()

    print("\nForward dynamics verification:")
    print("Movement: Target 1 -> Target 9")
    print(f"RMS endpoint error = {rms_error:.6e} m")
    print(f"Max endpoint error = {max_error:.6e} m")


def compute_forward_dynamics_errors_all_movements():
    """Run the reconstruction error check for all 72 movements."""
    error_results = []
    movement_number = 0

    for start_index, start_target in enumerate(target_points):
        for end_index, end_target in enumerate(target_points):
            if start_index == end_index:
                continue

            movement_number += 1

            x_desired, y_desired, _ = generate_endpoint_trajectory(
                start_target,
                end_target,
                time_points,
                movement_duration
            )

            desired_angles, desired_velocities, desired_accelerations = compute_joint_trajectory_from_endpoint(
                x_desired,
                y_desired,
                time_step
            )

            torque_profile = compute_torque_trajectory(
                desired_angles,
                desired_velocities,
                desired_accelerations
            )

            simulated_angles, _, _ = simulate_forward_dynamics(
                torque_profile,
                desired_angles[0],
                desired_velocities[0],
                time_step,
                method="semi_implicit"
            )

            x_simulated, y_simulated = endpoint_from_joint_trajectory(simulated_angles)

            _, rms_error, max_error = compute_cartesian_error(
                x_desired,
                y_desired,
                x_simulated,
                y_simulated
            )

            error_results.append({
                "movement_id": movement_number,
                "start_target": start_index + 1,
                "end_target": end_index + 1,
                "rms_error_m": rms_error,
                "max_error_m": max_error
            })

    return error_results


def plot_error_summary_all_movements():
    """Plot RMS and maximum endpoint error for all 72 movements."""
    error_results = compute_forward_dynamics_errors_all_movements()

    movement_ids = np.array([item["movement_id"] for item in error_results])
    rms_errors = np.array([item["rms_error_m"] for item in error_results])
    max_errors = np.array([item["max_error_m"] for item in error_results])

    plt.figure(figsize=(9, 4))
    plt.bar(movement_ids, rms_errors)
    plt.xlabel("Movement number")
    plt.ylabel("RMS endpoint error [m]")
    plt.title("Forward Dynamics RMS Error Across All 72 Movements")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(figures_folder / "all_movements_rms_error.png", dpi=300)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.bar(movement_ids, max_errors)
    plt.xlabel("Movement number")
    plt.ylabel("Maximum endpoint error [m]")
    plt.title("Forward Dynamics Maximum Error Across All 72 Movements")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(figures_folder / "all_movements_max_error.png", dpi=300)
    plt.show()

    worst_index = np.argmax(max_errors)
    worst_case = error_results[worst_index]

    print("\nForward dynamics error summary across all 72 movements:")
    print(f"Mean RMS error = {np.mean(rms_errors):.6e} m")
    print(f"Max RMS error  = {np.max(rms_errors):.6e} m")
    print(f"Mean max error = {np.mean(max_errors):.6e} m")
    print(f"Max error      = {np.max(max_errors):.6e} m")
    print(
        "Worst movement: "
        f"Target {worst_case['start_target']} -> Target {worst_case['end_target']}, "
        f"max error = {worst_case['max_error_m']:.6e} m"
    )

    return error_results


# Force-bump perturbation

def gaussian_torque_bump(time_array, amplitude=0.15, center_time=0.5, width=0.04):
    """
    Small temporary torque kick applied to the elbow.
    This is the planned little troublemaker in the simulation.
    """
    return amplitude * np.exp(-((time_array - center_time) ** 2) / (2 * width**2))


def plot_force_bump_perturbation():
    """Apply the elbow torque bump and compare nominal vs perturbed motion."""
    start_target = target_points[0]
    end_target = target_points[8]

    x_desired, y_desired, _ = generate_endpoint_trajectory(
        start_target,
        end_target,
        time_points,
        movement_duration
    )

    desired_angles, desired_velocities, desired_accelerations = compute_joint_trajectory_from_endpoint(
        x_desired,
        y_desired,
        time_step
    )

    nominal_torque = compute_torque_trajectory(
        desired_angles,
        desired_velocities,
        desired_accelerations
    )

    elbow_bump = gaussian_torque_bump(
        time_points,
        amplitude=0.15,
        center_time=0.5,
        width=0.04
    )

    perturbed_torque = nominal_torque.copy()
    perturbed_torque[:, 1] += elbow_bump

    initial_angles = desired_angles[0]
    initial_velocities = desired_velocities[0]

    nominal_angles, nominal_velocities, _ = simulate_forward_dynamics(
        nominal_torque,
        initial_angles,
        initial_velocities,
        time_step,
        method="semi_implicit"
    )

    perturbed_angles, perturbed_velocities, _ = simulate_forward_dynamics(
        perturbed_torque,
        initial_angles,
        initial_velocities,
        time_step,
        method="semi_implicit"
    )

    x_nominal, y_nominal = endpoint_from_joint_trajectory(nominal_angles)
    x_perturbed, y_perturbed = endpoint_from_joint_trajectory(perturbed_angles)

    perturbation_error = np.sqrt(
        (x_perturbed - x_nominal)**2 + (y_perturbed - y_nominal)**2
    )

    max_deviation = np.max(perturbation_error)
    final_deviation = perturbation_error[-1]

    plt.figure(figsize=(7, 4))
    plt.plot(time_points, elbow_bump, linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Perturbation torque [Nm]")
    plt.title("Gaussian Torque Bump Applied to Joint 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_folder / "force_bump_torque_profile.png", dpi=300)
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.plot(x_desired, y_desired, linewidth=2, label="Desired straight path")
    plt.plot(x_nominal, y_nominal, "--", linewidth=2, label="Nominal forward simulation")
    plt.plot(x_perturbed, y_perturbed, linewidth=2, label="Perturbed simulation")
    plt.scatter(start_target[0], start_target[1], s=100, marker="o", label="Start")
    plt.scatter(end_target[0], end_target[1], s=100, marker="x", label="End")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Endpoint Path With Joint 2 Torque Bump")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_folder / "force_bump_endpoint_path.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(time_points, perturbation_error, linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Endpoint deviation [m]")
    plt.title("Endpoint Deviation Caused by Force Bump")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_folder / "force_bump_endpoint_deviation.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(time_points, np.rad2deg(nominal_angles[:, 0]), "--", linewidth=2, label="Nominal q1")
    plt.plot(time_points, np.rad2deg(nominal_angles[:, 1]), "--", linewidth=2, label="Nominal q2")
    plt.plot(time_points, np.rad2deg(perturbed_angles[:, 0]), linewidth=2, label="Perturbed q1")
    plt.plot(time_points, np.rad2deg(perturbed_angles[:, 1]), linewidth=2, label="Perturbed q2")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint angle [deg]")
    plt.title("Joint Angles Under Joint 2 Torque Bump")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_folder / "force_bump_joint_angles.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(time_points, nominal_velocities[:, 0], "--", linewidth=2, label="Nominal q1_dot")
    plt.plot(time_points, nominal_velocities[:, 1], "--", linewidth=2, label="Nominal q2_dot")
    plt.plot(time_points, perturbed_velocities[:, 0], linewidth=2, label="Perturbed q1_dot")
    plt.plot(time_points, perturbed_velocities[:, 1], linewidth=2, label="Perturbed q2_dot")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint velocity [rad/s]")
    plt.title("Joint Velocities Under Joint 2 Torque Bump")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_folder / "force_bump_joint_velocities.png", dpi=300)
    plt.show()

    print("\nForce-bump perturbation test:")
    print("Movement: Target 1 -> Target 9")
    print("Perturbation applied to Joint 2 torque")
    print("Amplitude = 0.15 Nm")
    print("Center time = 0.5 s")
    print("Width = 0.04 s")
    print(f"Maximum endpoint deviation from nominal = {max_deviation:.6e} m")
    print(f"Final endpoint deviation from nominal   = {final_deviation:.6e} m")



# Main script

if __name__ == "__main__":
    print("2-DOF arm reaching project started.")
    print(f"Number of targets: {len(target_points)}")
    print(target_points)

    plot_target_grid()
    test_inverse_kinematics()

    plot_example_endpoint_trajectory()
    plot_all_endpoint_trajectories()
    check_singularities_all_movements()

    plot_example_joint_kinematics()
    plot_example_joint_torques()

    plot_forward_dynamics_verification()
    plot_error_summary_all_movements()

    plot_force_bump_perturbation()

    print("\nFinished.")
    print("Check the 'figures' folder. All main project figures should be there.")
    print("If a plot looks weird, congratulations: you found the next debugging adventure.")
