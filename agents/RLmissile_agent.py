"""Environment usage for a machine."""

import gymnasium as gym

from PIL import Image
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from gym_missile_command import MissileCommandEnv


from config import CONFIG
# from game.utility import  action_reset, sim_src_action # sim_enemies_action, sim_friends_action,
import __init__


if __name__ == "__main__":

    render = True
    # Create the environment
    config = {}
    # env = gym.make("missile-command-v0",env_config={})
    env = MissileCommandEnv("")
    # Reset it
    observation = env.reset()

    recorded_frames = []

    action = env.action_space.sample()
    # action_reset(action)

    # While the episode is not finished
    done = False
    step = 0

    while step < 1000 and not done:
        action = env.action_space.sample()

        # One step forward
        observation, reward, done, truncated, infos = env.step(action)
        # [obs], [reward], [terminated], [truncated], and [infos]

        if render:
            # Render (or not) the environment
            frame = env.render(mode="rgb_array")  # "processed_observation"/"rgb_array"
            if frame is not None:
                frame = np.array(Image.fromarray(frame.astype('uint8')))  #.rotate(180)
                recorded_frames.append(frame)

        step += 1

    clip = ImageSequenceClip(recorded_frames, fps=10)
    clip.write_videofile('capt.mp4')
    env.close()
