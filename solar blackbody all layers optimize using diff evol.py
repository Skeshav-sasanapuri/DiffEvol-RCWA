import pandas as pd
from scipy.constants import pi, c
import numpy as np
from rcwa_functions.cdwexpand import cdwexpand
from rcwa_functions.rotmap import rotmat
from rcwa_functions.rcwa1d import rcwa1d
from rcwa_functions.angles import get_angles
from scipy.optimize import differential_evolution


####################################################

def run_simulation(slant_angle, mu_cutoff, data):
    # Setup
    x = np.array([1, 0, 0])  # Unit vectors
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])

    R_s = 6.957e8  # Solar radius
    r = 1.496e11  # Orbital radius
    h = 6.6260705e-34  # Planck's constant
    K_B = 1.3806485e-23  # Boltzmann's constant
    t = 5770.2

    period_x = c / (mu_cutoff * 1e12)  # Grating period along x axis
    lambda_B = period_x * np.sin(np.deg2rad(slant_angle))  # Bragg grating period
    period_z = lambda_B / (np.cos(np.deg2rad(slant_angle)))  # Grating period along z axis
    lambda_b = 2 * n_eff * lambda_B * np.cos(np.deg2rad(slant_angle))  # Bragg wavelength

    KXY = np.array([2 * pi / period_x, 0])  # Grating vectors along x and y
    KZs = np.array([2 * pi / period_z])  # Grating vector along z

    n1 = None  # No different r- or t-side indexes
    n2 = None
    cdw = cdwexpand(nO, nE)  # CDWs Fourier expansion

    layers = [cdw]  # Set layer for RCWA

    # Run RCWA & Check polarization
    m = 2  # Include 3 harmonics
    tol = 1e-12  # Use default 1e-12 tolerance

    data_reflectance = []
    data_transmittance = []
    integrated_result = 0
    delta_mu = 920000000000.0

    # Iterate over rows in Excel and run simulations
    for row in data.itertuples(index=False):
        wavelength = row.Wavelength
        er_x = row.Et_x
        er_y = row.Et_y
        er_z = row.Et_z
        theta_in = row.Angle
        original_mode = row.previous_mode
        prev_irradiance = row.Irradiance
        mu = c / wavelength

        # # Convert strings to complex numbers
        # er_x = complex(er_x.strip('()'))
        # er_y = complex(er_y.strip('()'))
        # er_z = complex(er_z.strip('()'))

        # Create the numpy array
        if abs(er_x) < 1e-6:
            er_x = 0
        if abs(er_y) < 1e-6:
            er_y = 0
        if abs(er_z) < 1e-6:
            er_z = 0

        if not (er_x == 0 and er_y == 0 and er_z == 0):
            E0 = np.array([[er_x], [er_y], [er_z]])
            E0 = E0 / np.linalg.norm(E0)

            R_bragg = rotmat(y, np.deg2rad(theta_in))  # Rotation matrix
            direc = np.dot(R_bragg, z)  # Rotate incident light

            [kr, kt, Er, Et, R, T] = rcwa1d(m, wavelength, direc, E0, KXY, KZs, layers, thicks, n0, n1, n2, tol)

            kt_angles = get_angles(kt)
            kr_angles = get_angles(kr)

            mask_not_effervescent = (kr_angles > -90) & (kr_angles < 90)
            valid_R = R[0, 0, :][mask_not_effervescent.squeeze()]
            valid_R *= prev_irradiance

            solar_irradiance_at_mu = ((R_s ** 2) / (r ** 2)) * ((2 * np.pi * h * mu ** 3) / (c ** 2)) / \
                                     (np.exp((h * mu) / (K_B * t)) - 1)  # [W/m²/Hz]

            I_sum = np.sum(valid_R * np.sin(np.deg2rad(kr_angles[mask_not_effervescent]))) * solar_irradiance_at_mu

            integrated_result += I_sum * delta_mu

            for mode in range(-m, m):  # Modes: -2, -1, 0, 1, 2
                reflectance_eff = R[0, 0, mode + m]
                kr_components = kr[:, mode + m]
                Er_components = Er[0, :, mode + m]
                kr_angle = kr_angles[0, mode + m]

                transmittance_eff = T[0, 0, mode + m]
                kt_components = kt[:, mode + m]
                Et_components = Et[0, :, mode + m]
                kt_angle = kt_angles[0, mode + m]

                if -90 < kr_angle < 90:
                    data_reflectance.append([wavelength, mu*1e-12, theta_in, mode, original_mode, reflectance_eff,
                                             prev_irradiance*reflectance_eff, lambda_b, lambda_B, period_x,
                                             period_z, *Er_components, *kr_components, kr_angle, solar_irradiance_at_mu])

                if -90 < kt_angle < 90:
                    data_transmittance.append([wavelength, mu*1e-12, theta_in, mode, original_mode, transmittance_eff,
                                               prev_irradiance*transmittance_eff, lambda_b, lambda_B, period_x,
                                               period_z, *Et_components, *kt_components, kt_angle, solar_irradiance_at_mu])

    # Save data to Excel
    header_r = ["Wavelength", "Frequency", "theta_in", "Mode", "previous_mode", "Reflectance_Efficiency", "Irradiance", "lambda_b",
                "lambda_B", "period_x", "period_z", "Er_x", "Er_y", "Er_z", "Kr_x", "Kr_y", "Kr_z", "Angle",
                "Solar_irradiance_mu"]
    df_reflectance = pd.DataFrame(data_reflectance, columns=header_r)

    header_t = ["Wavelength", "Frequency", "theta_in", "Mode", "previous_mode", "Transmittance_Efficiency", "Irradiance", "lambda_b",
                "lambda_B", "period_x", "period_z", "Et_x", "Et_y", "Et_z", "Kt_x", "Kt_y", "Kt_z", "Angle",
                "Solar_irradiance_mu"]
    df_transmittance = pd.DataFrame(data_transmittance, columns=header_t)

    return {'Transmittance': df_transmittance, 'momentum_effeciency': integrated_result}

def run_first_simulation(slant_angle, mu_cutoff):
    # Setup
    x = np.array([1, 0, 0])  # Unit vectors
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])

    R_s = 6.957e8  # Solar radius
    r = 1.496e11  # Orbital radius
    h = 6.6260705e-34  # Planck's constant
    K_B = 1.3806485e-23  # Boltzmann's constant
    t = 5770.2

    delta_mu = ((mu_max_rcwa - mu_min_rcwa) / resolution)  # delta mu

    ER = np.array([[1], [-1j], [0]])  # E in
    ER = ER / np.linalg.norm(ER)

    period_x = c / (mu_cutoff * 1e12)  # Grating period along x axis
    lambda_B = period_x * np.sin(np.deg2rad(slant_angle))  # Bragg grating period
    period_z = lambda_B / (np.cos(np.deg2rad(slant_angle)))  # Grating period along z axis
    lambda_b = 2 * n_eff * lambda_B * np.cos(np.deg2rad(slant_angle))  # Bragg wavelength

    KXY = np.array([2 * pi / period_x, 0])  # Grating vectors along x and y
    KZs = np.array([2 * pi / period_z])  # Grating vector along z

    n1 = None  # No different r- or t-side indexes
    n2 = None
    cdw = cdwexpand(nO, nE)  # CDWs Fourier expansion

    layers = [cdw]  # Set layer for RCWA

    mus_range = np.linspace(mu_min_rcwa, mu_max_rcwa, resolution)  # input mu range for RCWA

    R_bragg = rotmat(y, theta_in)  # Rotation matrix
    direc = np.dot(R_bragg, z)  # Rotate incident light
    E0 = np.dot(R_bragg, ER)  # E in rotated by theta in along y-axis


    # Run RCWA & Check polarization
    m = 2  # Include 3 harmonics
    tol = 1e-12  # Use default 1e-12 tolerance

    data_reflectance = []
    data_transmittance = []
    integrated_result = 0

    for mu in mus_range:
        # m = math.floor((2 * period_x * 1e6)/((c/mu) * 1e6))
        [kr, kt, Er, Et, R, T] = rcwa1d(m, c / mu, direc, E0, KXY, KZs, layers, thicks, n0, n1, n2, tol)

        kt_angles = get_angles(kt)
        kr_angles = get_angles(kr)

        mask_not_effervescent = (kr_angles > -90) & (kr_angles < 90)
        valid_R = R[0, 0, :][mask_not_effervescent.squeeze()]

        solar_irradiance_at_mu = ((R_s ** 2) / (r ** 2)) * ((2 * np.pi * h * mu ** 3) / (c ** 2)) / \
                                 (np.exp((h * mu) / (K_B * t)) - 1)  # [W/m²/Hz]

        I_sum = np.sum(valid_R * np.sin(np.deg2rad(kr_angles[mask_not_effervescent]))) * solar_irradiance_at_mu

        integrated_result += I_sum * delta_mu

        wavelength = c / mu

        for mode in range(-m, m+1):  # Modes: -2, -1, 0, 1, 2
            reflectance_eff = R[0, 0, mode+m]
            kr_components = kr[:, mode+m]
            Er_components = Er[0, :, mode+m]
            kr_angle = kr_angles[0, mode+m]
            if -90 < kr_angle < 90:
                data_reflectance.append([wavelength, mu*1e-12, np.rad2deg(theta_in), mode, reflectance_eff, reflectance_eff,
                                         lambda_b, lambda_B, period_x, period_z, *Er_components, *kr_components,
                                         kr_angle, solar_irradiance_at_mu])

            transmittance_eff = T[0, 0, mode+m]
            kt_components = kt[:, mode+m]
            Et_components = Et[0, :, mode+m]
            kt_angle = kt_angles[0, mode+m]

            if -90 < kt_angle < 90:
                data_transmittance.append([wavelength, mu*1e-12, np.rad2deg(theta_in), mode, transmittance_eff, transmittance_eff,
                                           lambda_b, lambda_B, period_x, period_z, *Et_components, *kt_components,
                                           kt_angle, solar_irradiance_at_mu])

    # Save data to Excel
    header_r = ["Wavelength", "Frequency", "theta_in", "previous_mode", "Reflectance_Efficiency", "Irradiance", "lambda_b",
                "lambda_B", "period_x", "period_z", "Er_x", "Er_y", "Er_z", "Kr_x", "Kr_y", "Kr_z", "Angle",
                "Solar_irradiance_mu"]
    df_reflectance = pd.DataFrame(data_reflectance, columns=header_r)

    header_t = ["Wavelength", "Frequency", "theta_in", "previous_mode", "Transmittance_Efficiency", "Irradiance", "lambda_b",
                "lambda_B", "period_x", "period_z", "Et_x", "Et_y", "Et_z", "Kt_x", "Kt_y", "Kt_z", "Angle",
                "Solar_irradiance_mu"]
    df_transmittance = pd.DataFrame(data_transmittance, columns=header_t)

    return {'Transmittance': df_transmittance, 'momentum_effeciency': integrated_result}


####################################################
def progress_callback(xk, convergence):
    progress['iter'] += 1
    percent_complete = 100 * progress['iter'] / max_iter
    # with open("progress.log", "a") as f:
    #     f.write(f"Progress: {percent_complete:.1f}%\n")

def evaluate_total_effeciency(params):
    phi1, mu_c1, phi2, mu_c2, phi3, mu_c3 = params

    layer1 = run_first_simulation(slant_angle=phi1, mu_cutoff=mu_c1)
    layer2 = run_simulation(slant_angle=phi2, mu_cutoff=mu_c2, data=layer1['Transmittance'])
    layer3 = run_simulation(slant_angle=phi3, mu_cutoff=mu_c3, data=layer2['Transmittance'])

    total_eff = layer1['momentum_effeciency'] + layer2['momentum_effeciency'] + layer3['momentum_effeciency']
    print(f"Params: φ1={phi1:.2f}, ν1={mu_c1:.1f}, φ2={phi2:.2f}, ν2={mu_c2:.1f}, φ3={phi3:.2f}, "
          f"ν3={mu_c3:.1f} --> Efficiency: {total_eff:.4f}")
    with open("results.log", "a", encoding='utf-8') as f:
        f.write(f"Params: φ1={phi1:.2f}, ν1={mu_c1:.1f}, φ2={phi2:.2f}, ν2={mu_c2:.1f}, φ3={phi3:.2f}, "
                f"ν3={mu_c3:.1f} --> Efficiency: {total_eff:.4f}\n")
        f.flush()

    return -total_eff  # Minimize negative efficiency to maximize it

# Frequency range
mu_minimum_sbsi = 100 * 1e12  # Frequency lower limit for solar spectral irradiance [mu: Hz]
mu_maximum_sbsi = 1000 * 1e12  # Frequency upper limit for solar spectral irradiance [mu: Hz]

mu_max_rcwa = 640 * 1e12  # Frequency lower limit for RCWA [mu: Hz]
mu_min_rcwa = 180 * 1e12  # Frequency upper limit for RCWA[mu: Hz]

# slant angle phi
layer1_slant_angle = (10, 40)  # optimize for slant angle in this range for layer 1
layer2_slant_angle = (10, 40)  # optimize for slant angle in this range for layer 2
layer3_slant_angle = (10, 40)  # optimize for slant angle in this range for layer 3

# values of  cutoff mu for each layer in THz
# layer1_cutoff_mus = (400, 500)
# layer2_cutoff_mus = (250, 350)
# layer3_cutoff_mus = (170, 240)
layer1_cutoff_mus = (180, 600)
layer2_cutoff_mus = (180, 600)
layer3_cutoff_mus = (180, 600)

# Refractive indexes
nO = 1.7  # ordinary refractive index
nE = 1.2  # extraordinary refractive index
n_eff = np.sqrt(((nE ** 2) + (2 * (nO ** 2))) / 3)  # n effective
n0 = n_eff  # external refractive index

# Thickness
thicks = np.array([10e-6])  # Thickness of layer [m]

# incident angle
theta_in = np.deg2rad(0)  # incident angle [degrees]

# Resolution
resolution = 200  # number of steps between lambda_min and lambda_max

# Setup differential evolution

# To track progress
progress = {'iter': 0}
max_iter = 50  # default is 100 too. Can increase.

bounds = [
    layer1_slant_angle,
    layer1_cutoff_mus,
    layer2_slant_angle,
    layer2_cutoff_mus,
    layer3_slant_angle,
    layer3_cutoff_mus
]

if __name__ == "__main__":
    ###########################################

    print("\nFREQUENCY RANGE - ")
    print(f"For solar blackbody spectral irradiance curve: {mu_minimum_sbsi*1e-12}THz - {mu_maximum_sbsi*1e-12}THz | "
          f"{(c/mu_maximum_sbsi)*1e6}um - {(c/mu_minimum_sbsi)*1e6}um")
    print(f"For RCWA simulation for each layer: {mu_min_rcwa*1e-12}THz - {mu_max_rcwa*1e-12}THz | "
          f"{(c/mu_max_rcwa)*1e6}um - {(c/mu_min_rcwa)*1e6}um")

    ###########################################

    print(f"\nSlant angle for layer 1 optimized in range: {layer1_slant_angle}deg")
    print(f"\nSlant angle for layer 2 optimized in range: {layer2_slant_angle}deg")
    print(f"\nSlant angle for layer 3 optimized in range: {layer3_slant_angle}deg")

    ###########################################

    print(f"\nCut-off frequency for layer 1 optimized in range: {layer1_cutoff_mus}THz")
    print(f"\nCut-off frequency for layer 2 optimized in range: {layer2_cutoff_mus}THz")
    print(f"\nCut-off frequency for layer 3 optimized in range: {layer3_cutoff_mus}THz")

    ##########################################

    print("\nREFRACTIVE INDEX - Gap medium refractive index")
    print(f"Ordinary refractive index: {nO}")
    print(f"Extraordinary refractive index: {nE}")
    print(f"n effective: {n_eff}")
    print(f"External refractive index: {n0}")

    ###########################################

    print(f"Thickness for each layer: {thicks*1e6}um")

    ###########################################

    print(f"\nIncident angle: {np.rad2deg(theta_in)}deg")

    ###########################################

    print(f"\nResolution: {resolution}um")

    ###########################################

    result = differential_evolution(
        evaluate_total_effeciency,
        bounds,
        popsize=20,
        maxiter=max_iter,
        # callback=progress_callback,
        disp=True,
        workers=4
    )

    print("Best parameters:", result.x)
    print("Best efficiency:", -result.fun)

    print("Simulations completed.")
