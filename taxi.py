import gym
import numpy as np

game = 'Taxi-v2'
env = gym.make(game)

Q = np.zeros([env.observation_space.n, env.action_space.n])

env.render()

# Set learning parameters.
lr = 0.85  # Learning rate
gamma = 0.99
num_episodes = 2000  # AKA number of times to play.
# Create lists to contain total rewards and steps per episode.
j_list = []
r_list = []

for i in range(num_episodes):
    # Reset environment and get first new observation.
    s = env.reset()
    r_tot = 0  # Total reward so far
    d = False  # Is the game 'done'?
    j = 0  # We have taken 'j' steps.
    # The Q-Table learning algorithm.
    while j < 99:
        j += 1
        # Choose an action by greedily (with noise) picking from Q table.
        # Note that the noise decreases with num. games played.
        a = np.argmax(Q[s] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # Get new state, reward, 'done' from environment.
        s1, r, d, _ = env.step(a)
        # Update Q-Table with new knowledge according to Bellman.
        Q[s, a] += lr * (r + gamma * np.max(Q[s1]) - Q[s, a])
        r_tot += r
        s = s1
        if d is True:
            break
    j_list.append(j)  # Total number of steps
    r_list.append(r_tot)  # Total reward

print('Score over time: %s' % (np.average(r_list)))
print('Score over last half: %s' % (np.average(r_list[1000:])))

# Reset environment and show a simulation.
# Essentially same code as before, but with no learning
s = env.reset()
env.render()
d = False
while d is False:
    # time.sleep(1)  # Import time.time if you want to slow it down.
    a = np.argmax(Q[s])
    s1, r, d, _ = env.step(a)
    s = s1
    env.render()
