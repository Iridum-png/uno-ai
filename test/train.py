import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

gym.register('HeartPole-v0', entry_point='heartpole:HeartPole')
heartpole = gym.make('HeartPole-v0')

print(1)

model = DQN("MlpPolicy", heartpole, verbose=1,
            policy_kwargs={'net_arch': [16, 16]})
print(2)
model.learn(total_timesteps=500, log_interval=10)
print(3)
# mean_reward, std_reward = evaluate_policy(model, heartpole, n_eval_episodes=5)
# print(f'Reward per episode {mean_reward}±{std_reward}')
