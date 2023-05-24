import os
import gym
from stable_baselines3 import PPO

class BipedalWalkerV3:
    def __init__(self, env_name='BipedalWalker-v3'):
        self.env_name = env_name
        self.env = gym.make(env_name)

        if self.env is None:
            raise ValueError("Invalid environment. Please check if the environment name is correct.")
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def change_environment(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.env, key, value)

    def save_model(self, save_path, **kwargs):
        modified_save_path = self._modify_path(save_path, **kwargs)
        self.model.save(modified_save_path)

    def load_model(self, load_path):
        self.model = PPO.load(load_path)

    def train_model(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def retrain_model(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    def render_model(self, num_episodes=1):
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = self.env.step(action)

                try:
                    self.env.render()
                except Exception as e:
                    print(f"An error occurred during rendering: {e}")
                    break

        self.env.close()

    def _modify_path(self, path, **kwargs):
        modified_path = path

        for key, value in kwargs.items():
            modified_path = modified_path.replace(f'{{{key}}}', str(value))

        # create the 'models' folder if it doesn't exist
        models_folder = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_folder, exist_ok=True)

        modified_path = os.path.join(models_folder, modified_path)

        return modified_path

    def evaluate_model(self, num_episodes=10):
        rewards = []
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)

        average_reward = sum(rewards) / num_episodes
        print(f"Average reward over {num_episodes} episodes: {average_reward}")
