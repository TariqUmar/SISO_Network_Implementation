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
    rayleigh_channel = X + 1j*Y
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