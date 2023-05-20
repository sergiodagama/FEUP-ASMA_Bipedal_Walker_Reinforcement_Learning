from stable_baselines3 import A2C

env = "LunarLander-v2"
suffix = "simple"
TIMESTEPS = 1000
model = A2C("MlpPolicy", env, verbose=1)
model.learn(TIMESTEPS)


# TODO
""" functions for:
filenames
saving stuff
loading stuff
retraining from
automatically varying environments (if you want a lot of environments which is what we went for)
consider putting command line arguments on stuff
use models"""