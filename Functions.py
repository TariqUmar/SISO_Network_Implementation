import numpy as np
import random

# Function to generate channel coefficients

'''
#-------------------- description of variables ----------------------
Nt:         number of BS transmit antennas 
K:          number of single-antenna users 
nIRSrow:    number of rows of IRS elements 
nIRScol:    number of columns of IRS elements 
locU:       array of users' locations 
Lambda:     carrier wavelength   
kappa:      Rician factor 
xt:         x-coordiate of the center of tx ULA 
yt:         y-coordiate of the center of tx ULA 
zt:         z-coordiate of the center of tx ULA 
xs:         x-coordiate of the center of IRS UPA
ys:         y-coordiate of the center of IRS UPA 
zs:         z-coordiate of the center of IRS UPA
locT:       array of coordinates of the tx antennas 
locS:       array of coordinates of IRS elements 
dTU:        array of distance between tx antennas and user's antenna
dSU:        array of distance between IRS elements and user's antenna 
dTS:        array of distance between tx antennas and IRS elements 
alphaDir:   pathloss exponent for direct links 
alphaIRS:   pathloss exponent for IRS-related links 
betaTU:     pathloss for BS-users' links 
betaTS:     pathloss for BS-IRS links 
betaSU:     pathloss for IRS-user links 
hTU_LoS:    LoS conponent for BS-user links 
hTU_NLoS:   NLoS component for BS-user links 
hTU:        stack of tx-users' channel vectors 
hTS_LoS:    LoS conponent for BS-IRS links 
hTS_NLoS:   NLoS component for BS-IRS links 
hTS:        BS-IRS channel matrix 
hSU_LoS:    LoS conponent for IRS-user links 
hSU_NLoS:   NLoS component for IRS-user links 
hSU:        stack of IRS-users' channel vectors 
Gt:         transmit-antenna gain 
Gr:         receive-antenna gain
#--------------------------------------------------------------------            
'''

def generate_station_positions_2D(base_station_position):
    xt, yt = base_station_position
    return np.array([[xt, yt]])


def generate_station_positions_3D(base_station_position):
    xt, yt, zt = base_station_position
    return np.array([[xt, yt, zt]])


def generate_user_positions_2D(num_users, grid_radius):
    user_positions = []
    for _ in range(num_users):
        theta = 2 * np.pi * random.random()  # Azimuth angle
        radius = grid_radius * np.sqrt(random.random())  # Use sqrt for 2D
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        user_positions.append((x, y))
    return np.array(user_positions)


def generate_user_positions_3D(num_users, grid_radius):
    user_positions = []
    for _ in range(num_users):
        theta = 2 * np.pi * random.random()  # Azimuth angle
        phi = np.pi * random.random()       # Elevation angle
        radius = grid_radius * np.sqrt(random.random())  # Use cbrt for 3D
        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = 0
        user_positions.append((x, y, z))
    return np.array(user_positions)


def generate_transmit_antenna_coordinates_2D(Nt, xt, yt, halfLambda, quarterLambda):
    locTcenter = np.array([xt, yt], dtype=float)
    locT = np.tile(locTcenter, (Nt, 1))
    if Nt % 2 == 0:
        locT[0, 1] = yt - 0.5 * (Nt - 2) * halfLambda - quarterLambda
    else:
        locT[0, 1] = yt - 0.5 * (Nt - 1) * halfLambda
    locT[:, 1] = [locT[0, 1] + nt * halfLambda for nt in range(Nt)]
    return locT


def generate_transmit_antenna_coordinates_3D(Nt, xt, yt, zt, halfLambda, quarterLambda):
    locTcenter = np.array([xt, yt, zt], dtype=float)
    locT = np.tile(locTcenter, (Nt, 1))
    if Nt % 2 == 0:
        locT[0, 1] = yt - 0.5 * (Nt - 2) * halfLambda - quarterLambda
    else:
        locT[0, 1] = yt - 0.5 * (Nt - 1) * halfLambda
    locT[:, 1] = [locT[0, 1] + nt * halfLambda for nt in range(Nt)]
    return locT


def generate_IRS_2D(IRS_position):
    xs, ys = IRS_position
    return np.array([[xs, ys]])


def generate_IRS_3D(IRS_position):
    xs, ys, zs = IRS_position
    return np.array([[xs, ys, zs]])


def generate_irs_coordinates_2D(xs, ys, nIRSrow, nIRScol, halfLambda, quarterLambda):
    locS = np.zeros((nIRSrow, nIRScol, 2))
    for nRow in range(nIRSrow):
        for nCol in range(nIRScol):
            locS[nRow, nCol, 0] = xs - 0.5 * (nIRScol - 1) * halfLambda + nCol * halfLambda
            locS[nRow, nCol, 1] = ys - 0.5 * (nIRSrow - 1) * halfLambda + nRow * halfLambda
    return locS.reshape(nIRSrow * nIRScol, 2)


def generate_irs_coordinates_3D(xs, ys, zs, nIRSrow, nIRScol, halfLambda, quarterLambda):
    locS = np.zeros((nIRSrow, nIRScol, 3))
    for nRow in range(nIRSrow):
        for nCol in range(nIRScol):
            locS[nRow, nCol, 0] = xs - 0.5 * (nIRScol - 1) * halfLambda + nCol * halfLambda
            locS[nRow, nCol, 1] = ys - 0.5 * (nIRSrow - 1) * halfLambda + nRow * halfLambda
            locS[nRow, nCol, 2] = zs
    return locS.reshape(nIRSrow * nIRScol, 3)


def calc_distance_3D (var1, var2):
    x1,y1,z1 = var1
    x2,y2,z2 = var2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2)**2)

def calculate_distances_2D(locU, locT, locS):
    dTU = np.array([np.linalg.norm(locU[k, :-1] - locT[:, :-1], axis=1) for k in range(locU.shape[0])])
    dSU = np.array([np.linalg.norm(locU[k, :-1] - locS[:, :-1], axis=1) for k in range(locU.shape[0])])
    dTS = np.transpose(np.array([np.linalg.norm(locT[nt, :-1] - locS[:, :-1], axis=1) for nt in range(locT.shape[0])]))
    return dTU, dSU, dTS


def calculate_distances_3D(locU, locT, locS):
    dTU = np.array([np.linalg.norm(locU[k, :] - locT, axis=1) for k in range(locU.shape[0])])
    dSU = np.array([np.linalg.norm(locU[k, :] - locS, axis=1) for k in range(locU.shape[0])])
    dTS = np.transpose(np.array([np.linalg.norm(locT[nt, :] - locS, axis=1) for nt in range(locT.shape[0])]))
    return dTU, dSU, dTS


def compute_distances(user_positions, base_stations):
    distances = np.sqrt(np.sum(np.square(user_positions - base_stations), axis=1))
    return distances


# Function to compute outage probability at each iteration
def compute_outage_probability(num_users, rate, rate_threshold):
    outage = 0
    for j in range(num_users):
        if rate[j] < rate_threshold:
            outage += 1
    return outage / num_users


# Function to compute average outage probability
def compute_average_outage_probability(outage_probabilities):
    num_simulations = len(outage_probabilities)
    outage_prob_sum = np.sum(outage_probabilities)
    return outage_prob_sum / num_simulations

# Function to compute outage probability at each iteration
def compute_energy_efficiency(rate, power):
    return rate / power


# Function to compute average outage probability
def compute_average_energy_efficiency(ee):
    num_simulations = len(ee)
    outage_prob_sum = np.sum(ee)
    return outage_prob_sum / num_simulations


def compute_rate(SNR):
    SNR_watts = (10**(SNR/10))
    return np.log2(1 + SNR_watts)


def calc_link_budget(rayleigh_channel, distance, path_loss_exponent, transmit_power):
        link_inter = (((np.abs(rayleigh_channel)) / np.sqrt((distance) ** path_loss_exponent)) ** 2) * (transmit_power)
        link_budget = 10 * np.log10(link_inter) + 30 #need to add actual noise power
        return link_budget


def compute_noise(noise_floor, bandwidth):
    k = 1.38 * 10 ** (-23)
    T = 290
    NOISE_POWER = k*T*bandwidth*noise_floor
    return NOISE_POWER


def compute_path_loss(distances, path_loss_exponent):
    return 1 / np.sqrt(distances ** path_loss_exponent)


def generate_rayleigh_fading_channel(Nt, std_mean, std_dev):
    X = np.random.normal(std_mean, std_dev, Nt) 
    Y = np.random.normal(std_mean, std_dev, Nt) 
    rayleigh_channel = (X + 1j*Y)
    return rayleigh_channel


def generate_nakagami_samples(m, omega, size):
    magnitude_samples = np.sqrt(omega) * np.sqrt(np.random.gamma(m, 1, size)) / np.sqrt(np.random.gamma(m - 0.5, 1, size))
    phase_samples = np.random.uniform(0, 2 * np.pi, size=size)
    complex_samples = magnitude_samples * np.exp(1j * phase_samples)
    return complex_samples


def compute_SNR(link_budget, noise_floor):
    SNR = link_budget - noise_floor
    return SNR


def db2pow(x):
    # returns the dB value 
    return 10**(0.1*x)

def wrapTo2Pi(theta):
    return np.mod(theta,2*np.pi)

def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

#function for converting watts to dBm
def pow2dBm(watt):
    dBm = 10* np.log10(watt) + 30
    return dBm
    
#function for converting dBm to watts
def dBm2pow(dBm):
    watt = (10**(dBm/10))/1000
    return watt

def db2pow(dB):
    watt = (10**(dB/10))
    return watt

def pow2db(watt):
    db = 10 * np.log10(watt)
    return db


def generate_quantized_theta_set(B):
    K = 2**B
    delta_theta = 2 * np.pi / K
    quantized_theta_set = np.arange(0, K) * delta_theta - np.pi
    return quantized_theta_set

def compute_results_array_continuous(K, Ns, Nt, h_dk, h_rk, h_rk_transpose, G):
    # Initialize an empty list to store theta_n values for each i
    theta_n_values_complex = []

    for i in range(K):
        theta_n_i = []
        for j in range(Ns):
            theta_n = np.angle(h_dk[0][i]) - np.angle(h_rk[j][i]) - np.angle(G[j][0])
            theta_n = (theta_n + np.pi) % (2 * np.pi) - np.pi
            theta_n_i.append(theta_n)
        theta_n_values_complex.append(1 * np.exp(1j * np.array(theta_n_i)))

    theta_n_values_complex = np.array(theta_n_values_complex)

    # Initialize an empty list to store diagonal matrices
    diagonal_matrices = []

    for row in theta_n_values_complex:
        diagonal_matrix = np.diag(row[:Ns])
        diagonal_matrices.append(diagonal_matrix)

    # Convert diagonal_matrices to a NumPy array
    diagonal_matrices = np.array(diagonal_matrices)

    # Initialize an empty list to store the results for each column
    results_list = []

    for row_index in range(diagonal_matrices.shape[0]):
        single_row_diag = diagonal_matrices[row_index, :, :]
        single_row = h_rk_transpose[row_index,:]
    
        result_inter = np.dot(single_row, single_row_diag)

        result = np.dot(result_inter, G)
        results_list.append(result)

    # Convert the list of results into a numpy array
    results_array = np.array(results_list)
    results_array = results_array.reshape(Nt, K)

    return results_array

def results_array_discrete(K, Ns, Nt, h_dk, h_rk, h_rk_transpose, G, B):
    # Create a set of quantized theta values
    quantized_theta_set = ((2 * np.pi * np.arange(0, 2**B, 1) / (2**B)) - np.pi)
    quantized_theta_n_values_complex = []

    for i in range(K):
        quantized_theta_n_i = []

        for j in range(Ns):
            theta_n = - np.angle(h_rk[j][i]) - np.angle(G[j][0])
            theta_n = (theta_n + np.pi) % (2 * np.pi) - np.pi
            nearest_quantized_theta = quantized_theta_set[np.argmin(np.abs(theta_n - quantized_theta_set))]
            quantized_theta_n_i.append(nearest_quantized_theta)

        quantized_theta_n_values_complex.append(1 * np.exp(1j * np.array(quantized_theta_n_i)))

    theta_n_values_complex = np.array(quantized_theta_n_values_complex)

    # Initialize an empty list to store diagonal matrices
    diagonal_matrices = []

    # Transform each row into a diagonal matrix
    for row in theta_n_values_complex:
        diagonal_matrix = np.diag(row[:Ns])
        diagonal_matrices.append(diagonal_matrix)

    # Convert diagonal_matrices to a NumPy array
    diagonal_matrices = np.array(diagonal_matrices)

    # Initialize an empty list to store the results for each column
    results_list = []

    # Loop over each row/user in the diagonal_matrices
    for row_index in range(diagonal_matrices.shape[0]):
        single_row_diag = diagonal_matrices[row_index, :, :]
        single_row = h_rk_transpose[row_index,:]
        result_inter = np.dot(single_row, single_row_diag)
        result = np.dot(result_inter, G)
        results_list.append(result)

    # Convert the list of results into a numpy array
    results_array = np.array(results_list)
    results_array = results_array.reshape(Nt, K)

    return results_array

def results_array_practical_discrete(K, Ns, Nt, h_dk, h_rk, h_rk_transpose, G, B, beta_min, k, phi):
    # Create a set of quantized theta values
    quantized_theta_set = ((2 * np.pi * np.arange(0, 2**B, 1) / (2**B)) - np.pi)

    # Initialize an empty list to store quantized theta_n values for each i
    quantized_theta_n_values_complex = []

    for i in range(K):
        beta_n = []
        quantized_theta_n_i = []

        for j in range(Ns):
            theta_n = - np.angle(h_rk[j][i]) - np.angle(G[j][0])

            # Adjust theta_n to lie within the range (-π, π)
            theta_n = (theta_n + np.pi) % (2 * np.pi) - np.pi

            # Find the nearest quantized theta value
            nearest_quantized_theta_new = quantized_theta_set[np.argmin(np.abs(theta_n - quantized_theta_set))]
            quantized_theta_n_i.append(nearest_quantized_theta_new)

            beta_theta_n = ((1 - beta_min) * ((np.sin(nearest_quantized_theta_new - phi) + 1) / 2) ** k + beta_min)
            beta_n.append(beta_theta_n)

        quantized_theta_n_values_complex.append(np.array(beta_n) * np.exp(1j * np.array(quantized_theta_n_i)))

    theta_n_values_complex = np.array(quantized_theta_n_values_complex)

    # Initialize an empty list to store diagonal matrices
    diagonal_matrices = []

    # Transform each row into a diagonal matrix
    for row in theta_n_values_complex:
        diagonal_matrix = np.diag(row[:Ns])
        diagonal_matrices.append(diagonal_matrix)

    # Convert diagonal_matrices to a NumPy array
    diagonal_matrices = np.array(diagonal_matrices)

    # Initialize an empty list to store the results for each column
    results_list = []

    # Loop over each row/user in the diagonal_matrices
    for row_index in range(diagonal_matrices.shape[0]):
        # Get the corresponding diagonal matrix for the current row/user
        single_row_diag = diagonal_matrices[row_index, :, :]

        # Extract the single column from f_m_transpose using indexing and transpose
        single_row = h_rk_transpose[row_index,:]

        # Perform the dot product between f_m_transpose (5, 10) and the current diagonal matrix (10, 10)
        result_inter = np.dot(single_row, single_row_diag)

        # Perform the final matrix multiplication of the result_inter (5, 10) and g (10, 1)
        result = np.dot(result_inter, G)
        results_list.append(result)

    # Convert the list of results into a numpy array
    results_array = np.array(results_list)
    results_array = results_array.reshape(Nt, K)

    return results_array

def results_array_sharing_ideal(K, Ns, Nt, h_rk, h_rk_transpose, G):

    # Initialize an empty list to store theta_n values for each i
    theta_n_values_complex = []
    inc = int(Ns / K)

    for i in range(K):
        theta_n_i = []

        for j in range(i * inc, (i + 1) * inc):
            theta_n = -np.angle(h_rk[j][i]) - np.angle(G[j][0])

            # Adjust theta_n to lie within the range (-π, π)
            theta_n = (theta_n + np.pi) % (2 * np.pi) - np.pi

            # Find the nearest quantized theta value
            # nearest_quantized_theta_new = quantized_theta_set[np.argmin(np.abs(theta_n - quantized_theta_set))]
            # theta_n_i.append(nearest_quantized_theta_new)
            theta_n_i.append(theta_n)


        theta_n_values_complex.append(1 * np.exp(1j * np.array(theta_n_i)))

    theta_n_values_complex = np.array(theta_n_values_complex)

    # Initialize an empty list to store diagonal matrices
    diagonal_matrices = []

    # Transform each row into a diagonal matrix
    for row in theta_n_values_complex:
        diagonal_matrix = np.diag(row[:Ns])
        diagonal_matrices.append(diagonal_matrix)

    # Convert diagonal_matrices to a NumPy array
    diagonal_matrices = np.array(diagonal_matrices)

    results_list = []

    for row_index in range(diagonal_matrices.shape[0]):
        # Get the corresponding diagonal matrix for the current row/user is 1x1
        single_row_diag = diagonal_matrices[row_index]

        # Calculate the starting and ending indices for slicing based on row_index
        start_index = row_index * inc
        end_index = start_index + inc

        # Extract the single column from h_rk_transpose using slicing and transpose
        single_row = h_rk_transpose[row_index, start_index:end_index]

        # Reshape the single_row to (1, inc)
        single_row = single_row.reshape(1, inc)

        # Perform the dot product between f_m_transpose (1, inc) and the current diagonal matrix (inc, inc)
        result_inter = np.dot(single_row, single_row_diag)

        # Perform the final matrix multiplication of result_inter (1, inc) and a subset of G (inc, 1)
        subset_G = G[start_index:end_index]
        result = np.dot(result_inter, subset_G)
        results_list.append(result)

    # Convert the list of results into a numpy array
    results_array = np.array(results_list)
    results_array = results_array.reshape(Nt, K)

    return results_array

def results_array_sharing_practical(K, Ns, Nt, h_rk, h_rk_transpose, G, B, beta_min, phi, k):
    # Create a set of quantized theta values
    quantized_theta_set = ((2 * np.pi * np.arange(0, 2**B, 1) / (2**B)) - np.pi)

    # Initialize an empty list to store theta_n values for each i
    theta_n_values_complex = []
    inc = int(Ns / K)

    for i in range(K):
        theta_n_i = []
        beta_n = []

        for j in range(inc * i, inc * (i + 1)):
            theta_n = -np.angle(h_rk[j][i]) - np.angle(G[j][0])

            # Adjust theta_n to lie within the range (-π, π)
            theta_n = (theta_n + np.pi) % (2 * np.pi) - np.pi

            # Find the nearest quantized theta value
            nearest_quantized_theta_new = quantized_theta_set[np.argmin(np.abs(theta_n - quantized_theta_set))]
            theta_n_i.append(nearest_quantized_theta_new)

            beta_theta_n = ((1 - beta_min) * ((np.sin(nearest_quantized_theta_new - phi) + 1) / 2) ** k + beta_min)
            beta_n.append(beta_theta_n)

        theta_n_values_complex.append(np.array(beta_n) * np.exp(1j * np.array(theta_n_i)))

    theta_n_values_complex = np.array(theta_n_values_complex)

    # Initialize an empty list to store diagonal matrices
    diagonal_matrices = []

    # Transform each row into a diagonal matrix
    for row in theta_n_values_complex:
        diagonal_matrix = np.diag(row[:Ns])
        diagonal_matrices.append(diagonal_matrix)

    # Convert diagonal_matrices to a NumPy array
    diagonal_matrices = np.array(diagonal_matrices)

    results_list = []

    for row_index in range(diagonal_matrices.shape[0]):
        # Get the corresponding diagonal matrix for the current row/user is 1x1
        single_row_diag = diagonal_matrices[row_index]

        # Calculate the starting and ending indices for slicing based on row_index
        start_index = row_index * inc
        end_index = start_index + inc

        # Extract the single column from h_rk_transpose using slicing and transpose
        single_row = h_rk_transpose[row_index, start_index:end_index]
        
        # Reshape the single_row to (1, inc)
        single_row = single_row.reshape(1, inc)
        
        # Perform the dot product between f_m_transpose (1, inc) and the current diagonal matrix (inc, inc)
        result_inter = np.dot(single_row, single_row_diag)

        # Perform the final matrix multiplication of result_inter (1, inc) and a subset of G (inc, 1)
        subset_G = G[start_index:end_index]
        result = np.dot(result_inter, subset_G)
        results_list.append(result)

    # Convert the list of results into a numpy array
    results_array = np.array(results_list)
    results_array = results_array.reshape(Nt, K)

    return results_array

def compute_power_at_base_station(wn, Pt, PB_dBW):
    # Convert PB from dBW to dBm
    PB_dBm = PB_dBW + 30
    PB_watts = (10**(PB_dBm/10))/1000

    # Calculate P1
    P1 = wn * Pt + PB_watts

    return P1

def compute_power_consumption_at_ris(B, Ns):
    # Define power consumption levels for different quantization bits
    if B == 1:
        power_per_element = 5
    elif B == 2:
        power_per_element = 10
    elif B == 3:
        power_per_element = 15
    else:
        power_per_element = 20  # Default power consumption

    # Calculate total power consumption for all Ns elements
    power_consumption = power_per_element 
    power_consumption = (10**(power_consumption/10))/1000
    total_power_consumption = power_consumption * Ns
    return total_power_consumption
