how to run:
1)start a new python project with a virutal environment
2)pip install: 
pytorch version compatible with your machine, recommended to use the instructions in the official website
rllib, recommended to use the instructions in the official website
other requirements can be downloaded from pip wherever they are required, but most are included in the above downloads.
3)run random_agent to test everything is working, run rllib_agent to start training, or rllib_agent_record to test a checkpoint and save the video.

gym_missile_command folder - environment files
missile_command_env is the main environment we used, with the other variants still WIP

sprites - graphics for env, change the files to whatever you like

agents folder - scripts for running algorithms
random_agent - random agent to test the environment
DQN_agent - basic script of DQN implementation for testing
rllib_agent - train Rllib algorithms on the environment

We currently support PPO & IMPALA


![Screenshot from the game](https://github.com/Yanivhass/Missile-Command/blob/main/Screenshot.png)



