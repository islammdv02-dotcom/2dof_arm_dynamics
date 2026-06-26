# 2-DOF Arm Reaching Simulation

This project simulates a planar 2-degree-of-freedom arm moving between targets on a 3 by 3 grid. The arm has two joints, a shoulder joint and an elbow joint, and the endpoint represents the hand.

The goal of the project is to generate smooth reaching movements, compute the required joint motion and torques, and then verify the result using forward dynamics.

## What the project does

The script creates a simple two-link arm model and simulates reaching movements between 9 target points.

For each movement, the endpoint follows a straight-line path with a smooth bell-shaped speed profile. The program then computes the required joint angles, joint velocities, joint accelerations, and joint torques.

The project also checks the forward dynamics by applying the computed torques back to the arm model and comparing the simulated endpoint path with the desired path.

Finally, a torque bump is added to the second joint to show how a disturbance affects the arm motion in an open-loop simulation.

## Main features

- Creates a 3 by 3 target grid
- Simulates all 72 directed reaching movements between targets
- Generates minimum-jerk endpoint trajectories
- Computes forward kinematics
- Computes inverse kinematics
- Computes joint angles, velocities, and accelerations
- Computes inverse dynamics torques
- Runs forward dynamics verification
- Calculates endpoint reconstruction errors
- Applies a torque bump to Joint 2
- Saves all generated figures into the figures folder

## Assumptions

The arm is modeled as a planar two-link arm.

I used the suggested arm parameters:

- Upper arm length: 0.30 m
- Forearm length: 0.30 m
- Upper arm mass: 1.5 kg
- Forearm mass: 1.0 kg
- Movement duration: 1.0 s
- Time step: 0.001 s

Gravity is neglected in this simulation. This means the arm is assumed to move in a horizontal plane. This simplifies the model and focuses the project on the relationship between kinematics, inverse dynamics, forward dynamics, and perturbations.

## Target grid

The target grid is centered around x = 0.35 m and y = 0.0 m.

The x coordinates are:

0.27, 0.35, 0.43

The y coordinates are:

-0.08, 0.00, 0.08

This gives 9 reachable target points. The program simulates movement from every target to every other target, excluding movements from a target to itself. This gives 72 total movements.

## Requirements

This project uses Python with the following packages:

numpy
matplotlib

If these packages are not installed, install them with:

py -m pip install numpy matplotlib

or, depending on the Python setup:

python -m pip install numpy matplotlib

## How to run

Open the project folder in VS Code.

Make sure the main script is named:

main.py

Then run the file from VS Code using:

Run Python File

or run it from the terminal with:

python main.py

On Windows, this may also work:

py main.py

The script will automatically create a folder called figures if it does not already exist.

All plots will be saved inside the figures folder.

## Output figures

The script generates figures such as:

target_grid.png
all_72_endpoint_trajectories.png
example_endpoint_trajectory.png
example_speed_profile.png
example_joint_angles.png
example_joint_velocities.png
example_joint_accelerations.png
example_joint_torques.png
forward_dynamics_verification.png
forward_dynamics_error_time.png
all_movements_rms_error.png
all_movements_max_error.png
force_bump_torque_profile.png
force_bump_endpoint_path.png
force_bump_endpoint_deviation.png
force_bump_joint_angles.png
force_bump_joint_velocities.png

These figures are used for the final report.

## Forward dynamics check

After computing the inverse dynamics torques, the same torques are used in the forward dynamics simulation.

The reconstructed endpoint path is then compared with the desired endpoint path. The code reports the RMS endpoint error and the maximum endpoint error.

Small numerical errors are expected because the simulation uses numerical differentiation and numerical integration.

## Force bump test

The perturbation test adds a temporary torque bump to Joint 2 during the movement.

The bump is applied at the middle of the movement. The simulation is open-loop, so there is no controller correcting the motion after the disturbance.

This shows how a disturbance at the elbow joint affects the endpoint path and the joint velocities.

## Notes

The main purpose of this project is to connect the full simulation pipeline:

endpoint trajectory
inverse kinematics
joint motion
inverse dynamics torques
forward dynamics verification
perturbation test

The code is written as a single script so it can be run easily without changing paths or editing hidden parameters.
