import minari

dataset = minari.load_dataset("D4RL/antmaze/medium-diverse-v1", download=True)
env = dataset.recover_environment()   
print(env.spec.id)                   

import gymnasium as gym

render_env = gym.make(env.spec.id, render_mode="human")  
obs, info = render_env.reset()

for _ in range(500):
    action = render_env.action_space.sample()
    obs, reward, terminated, truncated, info = render_env.step(action)
    if terminated or truncated:
        obs, info = render_env.reset()

render_env.close()
