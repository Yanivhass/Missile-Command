from pettingzoo.atari import pong_v3
import pettingzoo
if __name__ == "__main__":
    env = pong_v3.env(render_mode="human")
    env.reset()

    timesteps = 200000
    for i in range(timesteps):
        live_agents = set(env.agents[:])
        for agent in env.agent_iter(env.num_agents*2):
            obs, reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                live_agents.remove(agent)
                action = None
            else:
                action = env.action_space(agent).sample()
            env.step(action)
        env.render()
    env.close()