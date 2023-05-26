# Default environment for the BipedalWalkerV3 environment, testing timesteps
from BipedalWalkerV3 import BipedalWalkerV3

environment_name = "default_env"
num_episodes = 100

# Random actions - this strategy serves as a baseline to compare to the others

obj = BipedalWalkerV3()
obj.evaluate_model(num_episodes)
obj.render_model(num_episodes)

# Basic Policy Optimization - this strategy uses the PPO algorithm to train the agent using default hyperparameter.
total_timesteps = 10000

obj = BipedalWalkerV3()
obj.train_model(total_timesteps)
obj.evaluate_model(num_episodes)
obj.save_model(f'bipedal_walker_model_{environment_name}_{total_timesteps}_steps_default_params.zip')
obj.render_model(num_episodes)


# with 25000 steps
env = BipedalWalkerV3()
env.train_model(total_timesteps=25000)
env.evaluate_model(num_episodes)
env.save_model(f'bipedal_walker_model_{environment_name}_{25000}_steps_default_params.zip')
env.render_model(num_episodes)

# with 50000 steps
env = BipedalWalkerV3()
env.train_model(total_timesteps=50000)
env.evaluate_model(num_episodes)
env.save_model(f'bipedal_walker_model_{environment_name}_{50000}_steps_default_params.zip')
env.render_model(num_episodes)

# with 75000 steps
env = BipedalWalkerV3()
env.train_model(total_timesteps=75000)
env.evaluate_model(num_episodes)
env.save_model(f'bipedal_walker_model_{environment_name}_{75000}_steps_default_params.zip')
env.render_model(num_episodes)

# with 100000 steps
env = BipedalWalkerV3()
env.train_model(total_timesteps=100000)
env.evaluate_model(num_episodes)
env.save_model(f'bipedal_walker_model_{environment_name}_{100000}_steps_default_params.zip')
env.render_model(num_episodes)

# with 250000 steps
env = BipedalWalkerV3()
env.train_model(total_timesteps=250000)
env.evaluate_model(num_episodes)
env.save_model(f'bipedal_walker_model_{environment_name}_{250000}_steps_default_params.zip')
env.render_model(num_episodes)

# with 500000 steps
env = BipedalWalkerV3()
env.train_model(total_timesteps=500000)
env.evaluate_model(num_episodes)
env.save_model(f'bipedal_walker_model_{environment_name}_{500000}_steps_default_params.zip')
env.render_model(num_episodes)
