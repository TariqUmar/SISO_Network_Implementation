import numpy as np
import torch
import random

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
    '''
    Input: Position as (x,y)
    Returns base station positions in the form of an array [(x,y)]
    '''
    xt, yt = base_station_position
    return np.array([[xt, yt]])


def generate_station_positions_3D(base_station_position: int):
    '''
    Input: Position as (x,y,z)
    Returns base station positions in the form of an array [(x,y,z)]
    '''
    xt, yt, zt = base_station_position
    return np.array([[xt, yt, zt]])


def generate_user_positions_2D(num_pos:int , r_range: int):
    '''
    Generates random user positions in (x,y)
    Input: num_pos = Number of positions to be generated, 
           r_range = Radius of Grid
    Output: positions = An array of positions as [(x,y)] 
    '''
    positions = np.array([])
    user_theta = np.random.uniform(0, 2*np.pi,num_pos)
    user_r = np.random.uniform(0,  r_range,num_pos)
    user_x = user_r * np.cos(user_theta)
    user_y = user_r * np.sin(user_theta)
    positions = [(x, y) for x, y in zip(user_x, user_y)]
    return positions


def generate_user_positions_3D(num_pos: int, r_range: int):
    user_positions = []
    for _ in range(num_pos):
        theta = 2 * np.pi * random.random()  # Azimuth angle
        phi = np.pi * random.random()       # Elevation angle
        radius = r_range * np.sqrt(random.random())  # Use cbrt for 3D
        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = 0
        user_positions.append((x, y, z))
    return np.array(user_positions)

def generate_transmit_antenna_coordinates_2D(Nt: int, xt, yt, halfLambda, quarterLambda):

    '''
        Generates coordinates of all the transmit antennas, located half a wavelength parat on the same transmitter.
        Input : Nt = Number of antennas, xt = x coordinate, yt = y coordinate, half lambda = half wavelength
    '''
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

class ExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.not_done[index]).to(self.device)
        )
