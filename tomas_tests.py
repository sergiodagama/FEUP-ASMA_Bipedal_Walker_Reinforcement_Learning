from BipedalWalkerV3 import BipedalWalkerV3
    
obj = BipedalWalkerV3()

# Change the environment parameters
gravity=-10
target_velocity=5
learning_rate=0.1 # determines the step size at which the model updates its estimates based on the observed rewards, higher - more relevant updates [0.000001,0.1]
discount_factor=0.99 # determines the importance of future rewards compared to immediate rewards, higher - more focus on future rewards [0,1]
epsilon=0.99 # balance between exploration and exploitation, higher - more exploration [0,1]
# max_steps_per_episode=100000

# obj.change_environment(gravity=gravity, target_velocity=target_velocity, learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon)

# Train the model
# obj.train_model(total_timesteps=100000)

# Save the model with parameter values in the path name
filename=f"a2c"
# obj.save_model(filename, gravity=gravity, target_velocity=target_velocity, learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon)

# Load the saved model
obj.load_model(filename)

# Render the model in the environment
obj.render_model(num_episodes=10)

# Retrain the model for better results
# obj.retrain_model(total_timesteps=10000)

# Render the model again
# obj.render_model(num_episodes=1)

# Evaluate the model's performance
# obj.evaluate_model(num_episodes=10)
