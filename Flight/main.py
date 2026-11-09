import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# -- Data configuration -- 
DATA_PATH = Path(__file__).parent / "data"
OUTPUT_PATH = Path(__file__).parent / "output"

HEADERS = {
    "time": "timestamp_ms",
    "acc": ["accel_x_ms2", "accel_y_ms2", "accel_z_ms2"],
    "gyro": ["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"],
    "mag": ["mag_x", "mag_y", "mag_z"],
}

METADATA = {
    "S_freq": 50,  # Sample frequency in Hz
    "Read_last": 60,  # Window of interest seconds
}


def load_data(file_path: Path, read_last: int = METADATA["Read_last"]) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    pandas_df = pd.read_csv(file_path, encoding='utf-8', comment='#')
    return pandas_df.tail(read_last * METADATA["S_freq"])


def extract_axis(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(data[HEADERS["acc"]], dtype=float),
        np.array(data[HEADERS["gyro"]], dtype=float),
        np.array(data[HEADERS["mag"]], dtype=float),
    )


def create_run_calibrated(acc_body: np.ndarray, gyro_dps: np.ndarray) -> np.ndarray:
    """
    Computes 3D trajectory using static bias calibration, 
    gravity cancellation, and velocity damping.
    """
    dt = 1.0 / METADATA["S_freq"]
    num_samples = len(acc_body)

    # 1. Calibrate static sensor biases using initial pad resting window (first 0.5 sec / 25 samples)
    pad_samples = min(25, num_samples)
    gyro_bias = np.mean(gyro_dps[:pad_samples], axis=0)
    acc_bias_pad = np.mean(acc_body[:pad_samples], axis=0)

    # Correct gyro offset
    gyro_corrected_rad = np.radians(gyro_dps - gyro_bias)

    # Determine static gravity magnitude from initial rest (+Y axis is UP)
    g_magnitude = np.linalg.norm(acc_bias_pad)
    if g_magnitude < 1.0:
        g_magnitude = 9.81  # Fallback if raw data is pre-centered
    gravity_world = np.array([0.0, g_magnitude, 0.0])

    # 2. Track orientation via Gyroscope Integration
    orientations = [R.identity()]
    for i in range(1, num_samples):
        delta_rotvec = gyro_corrected_rad[i] * dt
        delta_R = R.from_rotvec(delta_rotvec)
        orientations.append(orientations[-1] * delta_R)

    rotations = R.from_quat([r.as_quat() for r in orientations])

    # 3. Rotate body accelerations to global frame and subtract gravity
    acc_world = rotations.apply(acc_body)
    acc_kinematic = acc_world - gravity_world

    # 4. Filtered double integration (with noise threshold and velocity damping)
    velocity = np.zeros((num_samples, 3))
    position = np.zeros((num_samples, 3))
    
    # Velocity decay per frame prevents runaway quadratic drift over 60 seconds
    velocity_damping = 0.95  
    noise_threshold = 0.4    # Ignore residual noise below 0.4 m/s²

    for i in range(1, num_samples):
        k_acc = acc_kinematic[i]
        
        # Zero-velocity update / thresholding when motion is negligible
        if np.linalg.norm(k_acc) < noise_threshold:
            velocity[i] = velocity[i - 1] * 0.5  # Rapidly decay velocity to 0 at rest
        else:
            velocity[i] = (velocity[i - 1] + k_acc * dt) * velocity_damping

        position[i] = position[i - 1] + velocity[i] * dt

    return position


def plot_run_3d(run_data: np.ndarray):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(run_data[:, 0], run_data[:, 1], run_data[:, 2], label='Filtered Trajectory', color='blue', linewidth=2)
    ax.scatter(run_data[0, 0], run_data[0, 1], run_data[0, 2], color='green', s=60, label='Launch Point')
    ax.scatter(run_data[-1, 0], run_data[-1, 1], run_data[-1, 2], color='red', s=60, label='Landing Point')

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position - Altitude (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Calibrated Rocket Flight Trajectory')
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_data(DATA_PATH / "flight_001.csv")
    acc, gyro, mag = extract_axis(data)
    run = create_run_calibrated(acc, gyro)
    plot_run_3d(run)