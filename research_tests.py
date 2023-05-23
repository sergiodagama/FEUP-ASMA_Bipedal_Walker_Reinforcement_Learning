from BipedalWalkerV3 import BipedalWalkerV3

obj = BipedalWalkerV3()

# Change only the gravity and target velocity parameters
obj.change_environment(gravity=-10, target_velocity=5)

# Train the model
obj.train_model(total_timesteps=10000)

# Save the model with parameter values in the path name
obj.save_model('bipedal_walker_model_gravity_{gravity}_velocity_{target_velocity}.zip', gravity=-10, target_velocity=5)

# Load the saved model
obj.load_model('.\\models\\bipedal_walker_model_gravity_-10_velocity_5.zip')

# Render the model in the environment
obj.render_model(num_episodes=1)

# Evaluate the model's performance
# obj.evaluate_model(num_episodes=10)
