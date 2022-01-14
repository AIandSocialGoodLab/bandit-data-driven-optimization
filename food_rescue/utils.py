import torch
import datetime
from math import *
import numpy as np

def getTimeIntervalSigned(t1, t2):
    if isnan(t1) or isnan(t2):
        return np.nan
    duration = getTime(t2) - getTime(t1)
    (quo, rem) = divmod(duration.total_seconds(), 60)
    return (quo+1) if rem>0 else quo


def getNeighborGrid(grid):
    if grid == lon_brick * lat_brick:
        return []
    r, c = divmod(grid, lon_brick)
    neighbors = []
    if c > 0:
        neighbors.append(r*lon_brick+(c-1))
    if c < lon_brick - 1:
        neighbors.append(r*lon_brick+(c+1))
    if r > 0:
        neighbors.append((r-1)*lon_brick+c)
        if c > 0:
            neighbors.append((r-1)*lon_brick+(c-1))
        if c < lon_brick - 1:
            neighbors.append((r-1)*lon_brick+(c+1))
    if r < lat_brick - 1:
        neighbors.append((r+1)*lon_brick+c)
        if c > 0:
            neighbors.append((r+1)*lon_brick+(c-1))
        if c < lon_brick - 1:
            neighbors.append((r+1)*lon_brick+(c+1))
    return neighbors


def getDate(t):
    t = getTime(t)
    date = datetime.datetime(t.year, t.month, t.day, 0, 0, 0)
    return date

def calc_straight_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    if np.any(np.array([lat1, lon1, lat2, lon2]) == 0):
        return np.nan
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R*c
    distance = distance / 1.6
    return distance

calc_straight_dist_vec = np.vectorize(calc_straight_dist)

# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                                          lr=params['adam_lr'],
                                                          weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer
