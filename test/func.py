import gym
import matplotlib.pyplot as plt
import numpy as np


class HeartPole(gym.Env):
    def __init__(self, heart_attack_proclivity=0.5, max_steps=1000):
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


def do_nothing(state):
    pass


def drink_coffee(state):
    state['alertness'] += np.random.uniform(0, 1)
    state['hypertension'] += np.random.uniform(0, 0.3)


def drink_beer(state):
    state['alertness'] -= np.random.uniform(0, 0.5)
    state['hypertension'] += np.random.uniform(0, 0.3)
    state['intoxication'] += np.random.uniform(0.01, 0.03)


def sleep(state):
    """Have 16 half-hours of healthy sleep"""
    for _ in range(16):
        half_hour_passed(state)
    wakeup(state)


def wakeup(state):
    state['alertness'] = np.random.uniform(0.7, 1.3)
    state['time_since_slept'] = 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def heart_attack_risk(hypertension, heart_attack_proclivity=0.5):
    return heart_attack_proclivity * sigmoid(hypertension - 6)


def heart_attack_occured(state, heart_attack_proclivity=0.5):
    return np.random.uniform(0, 1) < heart_attack_risk(state['hypertension'], heart_attack_proclivity)


def alertness_decay(time_since_slept):
    return sigmoid((time_since_slept - 40) / 10)


def half_hour_passed(state):
    state['alertness'] -= alertness_decay(state['time_since_slept'])
    state['hypertension'] = decay_rate * state['hypertension']
    state['intoxication'] = decay_rate * state['intoxication']
    state['time_since_slept'] += 1
    state['time_elapsed'] += 1


def crippling_anxiety(alertness):
    return sigmoid(alertness - 3)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def ballmer_function(intoxication):
    return sigmoid((0.05 - intoxication) * 50) + 2 * gaussian(intoxication, 0.135, 0.005)


def productivity(state):
    p = 1
    p *= state['alertness']
    p *= 1 - crippling_anxiety(state['alertness'])
    p *= ballmer_function(state['intoxication'])
    return p


def work(state):
    state['work_done'] += productivity(state)
    half_hour_passed(state)


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


state = {
    'alertness': 0,
    'hypertension': 0,
    'intoxication': 0,
    'time_since_slept': 10,
    'time_elapsed': 0,
    'work_done': 0
}

plt.figure(figsize=(12, 8))
xspace = np.linspace(0, 10, 100)
plt.xlabel('hypertension')
plt.ylabel('heart attack risk')
plt.plot(xspace, heart_attack_risk(xspace))

decay_rate = 0.97
half_life = decay_rate ** 24
half_life

observations = ['alertness', 'hypertension', 'intoxication',
                'time_since_slept', 'time_elapsed', 'work_done']
actions = [do_nothing, drink_coffee, drink_beer, sleep]
heartpole_action_space = gym.spaces.Discrete(len(actions))

heartpole = HeartPole()

observation = heartpole.reset()
cups = 0

while 'HEART ATTACK' not in heartpole.log:
    observation, reward, done, _ = heartpole.step(1)
    cups += 1
    
print(cups)