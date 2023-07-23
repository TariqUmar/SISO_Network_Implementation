import numpy as np
import random

'''
    Functions here are:

     1) generate_station_positions
     2) generate_user_positions
     3) compute_distances
     4) compute_outage_probability
     5) compute_average_outage_probability
     6) compute_rate
     7) calc_link_budget
     8) compute_noise
     9) compute_path_loss
    10) generate_rayleigh_fading_channel
    11) compute_SNR

'''

def generate_station_positions(base_station_position):
    return np.array([base_station_position])


def generate_user_positions(num_users, grid_radius):
    user_positions = []
    for _ in range(num_users):
        angle = 2 * np.pi * random.random()
        radius = grid_radius * np.sqrt(random.random())
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        user_positions.append((x, y))
    return np.array(user_positions)


def compute_distances(user_positions):
    distances = np.sqrt(np.sum(np.square(user_positions), axis=1))
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
    SNR_watts = (10**(SNR/10))/1000
    return np.log2(1 + SNR_watts)


def calc_link_budget(rayleigh_channel, distance, path_loss_exponent, transmit_power):
        link_inter = (((np.abs(rayleigh_channel)) / np.sqrt((distance) ** path_loss_exponent)) ** 2)
        link_budget = 10 * np.log10(link_inter) + 30 + transmit_power #need to add actual noise power
        return link_budget


def compute_noise(noise_floor, bandwidth):
    k = 1.38 * 10 ** (-23)
    T = 290
    NOISE_POWER = k*T*bandwidth*noise_floor
    return NOISE_POWER


def compute_path_loss(distances, path_loss_exponent):
    return 1 / np.sqrt(distances ** path_loss_exponent)


def generate_rayleigh_fading_channel(num_users, std_mean, std_dev):
    X = np.random.normal(std_mean, std_dev, num_users) 
    Y = np.random.normal(std_mean, std_dev, num_users) 
    rayleigh_channel = np.abs(X + 1j*Y)
    return rayleigh_channel


def compute_SNR(link_budget, noise_floor):
    SNR = link_budget - noise_floor
    return SNR