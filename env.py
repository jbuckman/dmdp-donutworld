# environment is an agent on a circular track.
# reward is given by clockwise velocity.
# observations are given by a pixel array.
# actions are a movement in any direction, i.e. a vector in [-1, 1].
# if you are in the track, i.e. between 4 and 5 units from the center, you move at a rate of 1
# if you are far from the track, your movement is slowed proportional to your distance from the track
# if you are more than 1 away from the track, your movement is slowed to 0 i.e. you are stuck.
# thus, the location of the car is bounded by a circle of size 6

import numpy as np
import itertools

BASE_GRID = np.ones([32, 32])*0.25
xx, yy = np.meshgrid(np.arange(32), np.arange(32))
# outside 6 is no-go
BASE_GRID[(xx-16) ** 2 + (yy-16) ** 2 > (16 * 6./6) ** 2] = 0.
# inside 3 is no-go
BASE_GRID[(xx-16) ** 2 + (yy-16) ** 2 < (16 * 3./6) ** 2] = 0.
# between 4 and 5 is track
BASE_GRID[((xx-16) ** 2 + (yy-16) ** 2 > (16 * 4./6) ** 2) * ((xx-16) ** 2 + (yy-16) ** 2 < (16 * 5./6) ** 2)] = 0.5

# all possible actions
ACTIONS = np.zeros([16,2])
ACTIONS[:,0] = np.cos(np.array(range(16))*2*np.pi/16.)
ACTIONS[:,1] = np.sin(np.array(range(16))*2*np.pi/16.)

def get_initial_latent_states(batch_size):
    angles = np.random.uniform(0,2.*np.pi, [batch_size])
    dists = np.random.uniform(3.,6., [batch_size])
    points = np.stack([dists*np.cos(angles), dists*np.sin(angles)], axis=1)
    return points

def sample_random_actions(batch_size):
    # angles = np.random.uniform(0,2.*np.pi, [batch_size])
    # actions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    actions = np.random.randint(0, 16, size=[batch_size])
    return actions

def all_valid_obs():
    all_obs = []
    locs = []
    for i,j in itertools.product(range(32), range(32)):
        if BASE_GRID[i,j] == 0: continue
        new_grid = np.array(BASE_GRID)
        new_grid[i,j] = 1.
        all_obs.append(new_grid)
        locs.append((i,j))
    return np.stack(all_obs), locs

def convert_to_obs(latents):
    base = np.stack([BASE_GRID]*latents.shape[0])
    swap_lats = np.roll(latents, 1, 1)
    swap_lats[:, 0] *= -1
    idx = np.floor(swap_lats/6. * 16 + 16).astype(np.int)
    for i in range(latents.shape[0]):
        base[i, idx[i,0], idx[i,1]] = 1.0
    return base

def convert_to_latents(obs):
    locs = np.argmax(obs.reshape([-1,32**2]), axis=1)
    x = (np.floor(locs/32)/32. - 0.5 + 1/64.) * 12.
    y = ((locs%32)/32. - 0.5 + 1/64.) * 12.
    latents = np.stack([x,y], axis=1)
    swap_lats = np.roll(latents, 1, 1)
    swap_lats[:, 1] *= -1
    return swap_lats

def apply_actions(lats, actions):
    dist = np.linalg.norm(lats, axis=1)
    speed = (5 < dist).astype(np.float) * np.maximum(6 - dist, 0) + \
            ((3 <= dist) * (dist <= 5)).astype(np.float) * np.ones([lats.shape[0]]) + \
            (dist < 3).astype(np.float) * np.maximum(dist - 2, 0)
    actions = ACTIONS[actions]
    real_actions = actions / np.expand_dims(np.linalg.norm(actions, axis=1),1) * np.expand_dims(speed,1)
    new_lats = lats + real_actions
    return new_lats

def get_reward(states, actions, next_states):
    start_angle = np.angle(states[:,0] + 1j*states[:,1])
    end_angle = np.angle(next_states[:,0] + 1j*next_states[:,1])
    angle_change = ((end_angle - start_angle) % (2*np.pi))
    reward = -(angle_change * (angle_change < np.pi) + (angle_change - 2*np.pi) * (angle_change >= np.pi))
    # reward += .1*np.abs(np.sqrt(np.sum(np.square(next_states),-1)) - 4.5)
    return reward

def sample_env_tuple(batch_size):
    states = get_initial_latent_states(batch_size)
    actions = sample_random_actions(batch_size)
    next_states = apply_actions(states, actions)
    rewards = get_reward(states, actions, next_states)

    obs = convert_to_obs(states)
    next_obs = convert_to_obs(next_states)
    next_obs2 = convert_to_obs(next_states)

    return obs, actions, rewards, next_obs, next_obs2

def get_return_from_action_sequence(state, actions_sequence, gamma):
    returns = 0.
    for i, actions in enumerate(actions_sequence):
        next_state = apply_actions(state, actions)
        returns += gamma**i * get_reward(state, actions, next_state)
        state = next_state
    return returns

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    lats = get_initial_latent_states(12)
    actions = sample_random_actions(12)
    print ACTIONS[actions][0]
    obs = convert_to_obs(lats)
    reclats = convert_to_latents(obs)
    plt.imshow(obs[0], cmap='gray')
    plt.show()

    new_lats = apply_actions(lats, actions)
    new_obs = convert_to_obs(new_lats)
    new_reclats = convert_to_latents(new_obs)
    print get_reward(lats, actions, new_lats)[0]
    plt.imshow(new_obs[0], cmap='gray')
    plt.show()

