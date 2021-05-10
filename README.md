## OpenAI Gym Taxi v2

Guillermo del Valle Reboul

## Project details

The goal of the agent is to collect a client at point A and then move to point B while avoiding obstacles.

Unity brain name: BananaBrain
Vector Observation space type: discrete
Vector Action space type: discrete
Vector Action space size (per agent): 6 (up, down, right, left, pick client, drop client)
Vector Action descriptions: , , ,


state vector = discrete values, grid
actions = 4 discrete actions (forward, backward, turn left, turn right)
The environment is considered solved when agents reaches average score of 9.7.

## Algorithm Used: Q Learning (RL)

A Q Learning agent was used for this project. The policy in use is epsilon greedy.
Hyperparameters:
* Num episodes = 20000
* GAMMA (discount factor) = 0.77
* Alpha = 0.25

</br>

## Instructions

Download OpenAI Gym Taxi v2 and execute main.py.
