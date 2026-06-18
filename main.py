import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# Project setup
# =========================

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# =========================
# Arm parameters
# =========================

l1 = 0.30  # upper arm length [m]
l2 = 0.30  # forearm length [m]

m1 = 1.5   # upper arm mass [kg]
m2 = 1.0   # forearm mass [kg]


# =========================
# Movement parameters
# =========================

T = 1.0       # movement duration [s]
dt = 0.001    # timestep [s]
t = np.arange(0, T + dt, dt)


# =========================
# Target grid
# =========================

x_vals = np.array([0.27, 0.35, 0.43])
y_vals = np.array([-0.08, 0.00, 0.08])

targets = np.array([[x, y] for y in y_vals for x in x_vals])


# =========================
# Forward kinematics
# =========================

def forward_kinematics(q1, q2, l1=0.30, l2=0.30):
    """
    Given shoulder angle q1 and elbow angle q2,
    compute hand position x, y.
    """
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

    return x, y


# =========================
# Inverse kinematics
# =========================

def inverse_kinematics(x, y, l1=0.30, l2=0.30, elbow="down"):
    """
    Given hand position x, y,
    compute shoulder angle q1 and elbow angle q2.
    """
    r2 = x**2 + y**2

    cos_q2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)

    if elbow == "down":
        q2 = np.arccos(cos_q2)
    else:
        q2 = -np.arccos(cos_q2)

    q1 = np.arctan2(y, x) - np.arctan2(
        l2 * np.sin(q2),
        l1 + l2 * np.cos(q2)
    )

    return q1, q2


# =========================
# Plot target grid
# =========================

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


# =========================
# Test inverse kinematics
# =========================

def test_inverse_kinematics():
    print("Inverse kinematics test:")

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


# =========================
# Main script
# =========================

if __name__ == "__main__":
    print("2-DOF arm reaching project started.")
    print(f"Number of targets: {len(targets)}")
    print(targets)

    plot_target_grid()
    test_inverse_kinematics()