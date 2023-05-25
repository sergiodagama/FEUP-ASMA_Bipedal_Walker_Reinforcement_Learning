# Environment changes tests

from BipedalWalkerV3 import BipedalWalkerV3

num_episodes = 100
total_timesteps = 100000

# Changing the wind force (wind pushing from the left)
environment_name = "wind_5"

env = BipedalWalkerV3()
env.change_environment(wind = 5)  # modify the wind force as desired
env.train_model(total_timesteps)
env.evaluate_model(num_episodes)
env.save_model(f'bipedal_walker_model_{environment_name}_{total_timesteps}_steps_default_params.zip')
env.render_model(num_episodes)

# Changing the wind force (wind pushing from the right)
environment_name = "wind_-5"

env = BipedalWalkerV3()
env.change_environment(wind = -5)  # modify the wind force as desired
env.train_model(total_timesteps)
env.evaluate_model(num_episodes)
env.save_model(f'bipedal_walker_model_{environment_name}_{total_timesteps}_steps_default_params.zip')
env.render_model(num_episodes)

# Changing the gravity
environment_name = "gravity_-5"

env = BipedalWalkerV3()
env.change_environment(gravity = -10)
env.train_model(total_timesteps)
env.evaluate_model(num_episodes)
env.save_model(f'bipedal_walker_model_{environment_name}_{total_timesteps}_steps_default_params.zip')
env.render_model(num_episodes)

# Changing the sensor noise
environment_name = "sensor_noise_1.0"

env = BipedalWalkerV3()
env.change_environment(sensor_noise = 1.0)
env.train_model(total_timesteps)
env.evaluate_model(num_episodes)
env.save_model(f'bipedal_walker_model_{environment_name}_{total_timesteps}_steps_default_params.zip')
env.render_model(num_episodes)
