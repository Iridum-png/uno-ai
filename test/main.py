import gym
import numpy as np


class Productivity(gym.Env):
    def __init__(self, heart_pole_proclivity=0.5, max_steps=1000):
        self.actions = actions
        self.observations = observations
        self.action_space = heartpole_action_space
        self.observation_space = make_heartpole_obs_space()
        self.heart_attack_proclivity = heart_attack_proclivity
        self.log = ''
        self.max_steps = max_steps

    def observation(self):
        return np.array([self.state[o] for o in self.observations])

    def reset(self):
        self.state = {
            'alertness': 0,
            'hypertension': 0,
            'intoxication': 0,
            'time_since_slept': 0,
            'time_elapsed': 0,
            'work_done': 0,
        }

        self.steps_left = self.max_steps

        wakeup(self.state)
        return self.observation

    def step(self, action):
        if self.state['time_elapsed'] == 0:
            old_score = 0
        else:
            old_score = self.state['work_done'] / self.state['time_elapsed']

        self.actions[action](self.state)
        self.log += f'Chosen action: {self.actions[action].__name__}\n'

        work(self.state)

        new_score = self.state['work_done'] / self.state['time_elapsed']

        reward = new_score - old_score

        if heart_attack_occured(self.state, self.heart_attack_proclivity):
            self.log += 'HEART ATTACK\n'
            reward -= 100
            self.state['hypertension'] = 0

        self.log += str(self.state) + '\n'
        done = self.steps_left <= 0

        return self.observation, reward, done, {}

    def close(self):
        pass

    def render(self, mode=None):
        print(self.log)
        self.log = ''


def make_heartpole_obs_space():
    lower_obs_bound = {
        'alertness': - np.inf,
        'hypertension': 0,
        'intoxication': 0,
        'time_since_slept': 0,
        'time_elapsed': 0,
        'work_done': - np.inf
    }
    higher_obs_bound = {
        'alertness': np.inf,
        'hypertension': np.inf,
        'intoxication': np.inf,
        'time_since_slept': np.inf,
        'time_elapsed': np.inf,
        'work_done': np.inf
    }

    low = np.array([lower_obs_bound[o] for o in observations])
    high = np.array([higher_obs_bound[o] for o in observations])
    shape = (len(observations),)
    return gym.spaces.Box(low, high, shape)


def do_nothing(state):
    pass


def sleep(state):
    """Have 16 half-hours of healthy sleep"""
    for hh in range(16):
        half_hour_passed(state)
    wakeup(state)


def heart_attack_occured(state, heart_attack_proclivity=0.5):
    return np.random.uniform(0, 1) < heart_attack_risk(state['hypertension'], heart_attack_proclivity)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def heart_attack_risk(hypertension, heart_attack_proclivity=0.5):
    return heart_attack_proclivity * sigmoid(hypertension - 6)


actions = [do_nothing, drink_coffee, drink_beer, sleep]
heartpole_action_space = gym.spaces.Discrete(len(actions))
observations = ['alertness', 'hypertension', 'intoxication',
                'time_since_slept', 'time_elapsed', 'work_done']
