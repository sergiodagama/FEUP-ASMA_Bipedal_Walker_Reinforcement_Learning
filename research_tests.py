from BipedalWalkerV3 import BipedalWalkerV3

obj = BipedalWalkerV3()

# Change the environment parameters
gravity=-10
target_velocity=5
learning_rate=0.01 # determines the step size at which the model updates its estimates based on the observed rewards, higher - more relevant updates
discount_factor=0.99 # determines the importance of future rewards compared to immediate rewards, higher - more focus on future rewards
epsilon=0.1 # balance between exploration and exploitation, higher - more exploration
max_steps_per_episode=1000

obj.change_environment(gravity=gravity, target_velocity=target_velocity, learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon, max_steps_per_episode=max_steps_per_episode)

# Train the model
obj.train_model(total_timesteps=10000)

# Save the model with parameter values in the path name
filename=f"bipedal_walker_model_gravity_{gravity}_velocity_{target_velocity}_learning_rate_{learning_rate}_discount_factor{discount_factor}_epsilon_{epsilon}_max_steps_per_episode_{max_steps_per_episode}.zip"
obj.save_model(filename, gravity=gravity, target_velocity=target_velocity, learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon, max_steps_per_episode=max_steps_per_episode)

# Load the saved model
obj.load_model(filename)

# Render the model in the environment
obj.render_model(num_episodes=1)

# Evaluate the model's performance
obj.evaluate_model(num_episodes=10)
