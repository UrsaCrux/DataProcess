import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

# --- MOTOR & BENCH PARAMETERS ---
START_INDEX = 843
END_INDEX = 955
G = 9.802         # Gravity [m/s^2]

def read_data(file_path: str, start_index: int = 0, end_index: int = -1) -> np.ndarray:
    data = pd.read_csv(file_path, delimiter=',', header=1, names=['N muestra', 'Newtons', 'Tiempo'], comment="#")
    raw = data["Newtons"].to_numpy()
    return raw[start_index:end_index]

def mass_decaiment_func(initial_mass: float, final_mass: float, initial_index: int, final_index: int, dt: float, mode: str = "linear") -> np.ndarray:
    if mode == "linear":
        return np.linspace(initial_mass, final_mass, final_index - initial_index)
    elif mode == "exponential":
        return np.exp(np.linspace(np.log(initial_mass), np.log(final_mass), final_index - initial_index))
    elif mode == "None":
        return np.full(final_index - initial_index, initial_mass)
    elif mode == "full":
        return np.full(final_index - initial_index, final_mass)
    elif mode == "logarithmic":
        return np.logspace(np.log10(initial_mass), np.log10(final_mass), final_index - initial_index)
    else:
        raise ValueError(f"Unknown mass decay mode: {mode}")

def get_convertions(initial_mass: float, final_mass: float, raw_no_rocket: float, raw_rocket: float, sps: int) -> tuple:
    rocket_mass_raw = raw_rocket - raw_no_rocket
    raw_to_kg = initial_mass / rocket_mass_raw
    fuel_mass = initial_mass - final_mass
    dt = 1 / sps
    return (raw_to_kg, fuel_mass, dt)

def calculate_information(data: np.ndarray, initial_mass: float, final_mass: float, raw_no_rocket: float, raw_rocket: float, sps: int, mass_decaiment_mode: str = "linear") -> dict:
    raw_to_kg, fuel_mass, dt = get_convertions(initial_mass, final_mass, raw_no_rocket, raw_rocket, sps)
    mass = mass_decaiment_func(initial_mass, final_mass, START_INDEX, END_INDEX, dt, mode=mass_decaiment_mode)
    
    # 1. Restar la tara (baseline)
    data_net = data - raw_no_rocket
    
    # 2. Convertir a Newtons multiplicando por G
    measured_force = data_net * raw_to_kg * G
    
    # 3. Calcular empuje compensando el peso de la masa decreciente
    thrust = measured_force - mass * G

    return {
        "newtons": measured_force,
        "thrust": thrust,
        "mass": mass,
        "initial_mass": initial_mass,
        "fuel_mass": fuel_mass,
        "dt": dt,
        "burn_time": (END_INDEX - START_INDEX) * dt,
        "max_thrust": np.max(thrust),
        "avg_thrust": np.mean(thrust),
        "total_impulse": cumulative_trapezoid(thrust, dx=dt)[-1],
        "to_max_burn_time": np.argmax(thrust) * dt,
    }

def print_information(info: dict) -> None:
    print(f"Fuel mass: {info['fuel_mass']:.3f} kg")
    print(f"Burn time: {info['burn_time']:.2f} s")
    print(f"Max thrust: {info['max_thrust']:.2f} N")
    print(f"Average thrust: {info['avg_thrust']:.2f} N")
    print(f"Time to max thrust: {info['to_max_burn_time']:.2f} s")
    print(f"Total impulse: {info['total_impulse']:.2f} Ns")

def save_results(info: dict, motor_name: str = "Motor_2", output_dir: str = "output", motor_diameter_mm: float = 29.0, motor_length_mm: float = 100.0) -> None:
    """
    Saves metrics report, graphs, CSV time series, and OpenRocket .eng file.
    """
    os.makedirs(output_dir, exist_ok=True)
    time = np.arange(len(info['thrust'])) * info['dt']
    # 1. OpenRocket .eng Export (Standard RASP Format)
    eng_path = os.path.join(output_dir, f"{motor_name}.eng")
    thrust_export = np.maximum(0.0, info['thrust'].copy())
    thrust_export[-1] = 0.0  # Zero out final thrust point for OpenRocket bounds

    lines = [
        f"; Exported thrust curve from main.py analysis for {motor_name}",
        f"; Burn time: {info['burn_time']:.2f} s | Total impulse: {info['total_impulse']:.2f} Ns",
        f"{motor_name} {motor_diameter_mm:.1f} {motor_length_mm:.1f} 0 {info['fuel_mass']:.4f} {info['initial_mass']:.4f} Custom",
    ]
    for t, f in zip(time, thrust_export):
        lines.append(f"{t:.4f} {f:.4f}")
    lines.append(";")

    with open(eng_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 2. Complete CSV Export for Excel/MATLAB/Python
    csv_path = os.path.join(output_dir, f"{motor_name}_data.csv")
    df_export = pd.DataFrame({
        'time_s': time,
        'measured_force_N': info['newtons'],
        'thrust_N': info['thrust'],
        'mass_kg': info['mass']
    })
    df_export.to_csv(csv_path, index=False)

    # 3. High-Resolution Graph Export (PNG)
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.plot(time, info['newtons'], label='Measured Force (N)', color='blue')
    axs.plot(time, info['thrust'], label='Calculated Thrust (N)', color='orange')
    axs.set_title('Thrust vs Time')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Force (N)')
    axs.legend()
    axs.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{motor_name}_plot.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    # 4. Summary Text File
    report_path = os.path.join(output_dir, f"{motor_name}_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== MOTOR ANALYSIS REPORT: {motor_name} ===\n")
        f.write(f"Fuel mass: {info['fuel_mass']:.3f} kg\n")
        f.write(f"Burn time: {info['burn_time']:.2f} s\n")
        f.write(f"Max thrust: {info['max_thrust']:.2f} N\n")
        f.write(f"Average thrust: {info['avg_thrust']:.2f} N\n")
        f.write(f"Time to max thrust: {info['to_max_burn_time']:.2f} s\n")
        f.write(f"Total impulse: {info['total_impulse']:.2f} Ns\n")

    print(f"\n[+] Results successfully saved to directory '{output_dir}/':")
    print(f"  - OpenRocket Engine File: {eng_path}")
    print(f"  - Complete CSV Data:      {csv_path}")
    print(f"  - Graph Figure (300 DPI): {plot_path}")
    print(f"  - Text Summary:          {report_path}")

# --- EXECUTION ---
data = read_data('Motor 2.csv', start_index=START_INDEX, end_index=END_INDEX)
info = calculate_information(data, initial_mass=0.145, final_mass=0.101, raw_no_rocket=1254, raw_rocket=1280, sps=20, mass_decaiment_mode="logarithmic")

print_information(info)
save_results(info, motor_name="Motor_2", output_dir="output")