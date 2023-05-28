import numpy as np
import matplotlib.pyplot as plt

class CliffWalkingEnvironment:
    def __init__(self, grid):
        self.grid = grid
        self.grid_height = len(grid)
        self.grid_width = len(grid[0])
        self.action_space_size = 4

        self.state_space = [(i, j) for i in range(self.grid_height) for j in range(self.grid_width)]
        self.start_state = self._get_state('S')
        self.goal_state = self._get_state('G')

        self.current_state = self.start_state
        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']  # 0, 1, 2, 3

    def _get_state(self, label):
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.grid[i][j] == label:
                    return (i, j)

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        i, j = self.current_state
        if action == 0:  # UP
            next_state = max(i-1, 0), j
        elif action == 1:  # RIGHT
            next_state = i, min(j+1, self.grid_width-1)
        elif action == 2:  # DOWN
            next_state = min(i+1, self.grid_height-1), j
        elif action == 3:  # LEFT
            next_state = i, max(j-1, 0)

        self.current_state = next_state
        reward = -1  # default reward

        if next_state == self.goal_state:
            done = True
            reward = 10  # reward for reaching the goal
        elif self.grid[next_state[0]][next_state[1]] == -100:
            done = True
            reward = -100
            self.current_state = self.start_state
        else:
            done = False

        return next_state, reward, done
    

def epsilon_greedy_policy(Q, state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice(4)  # 4 actions
    else:
        return np.argmax(Q[state])
    
def q_learning(env, num_episodes, alpha, gamma, epsilon):

    Q = {state: [0 for _ in range(env.action_space_size)] for state in env.state_space}
    rewards = []  # for storing cumulative rewards per episode

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])
            state = next_state

        rewards.append(episode_reward)

    return Q, rewards

def sarsa(env, num_episodes, alpha, gamma, epsilon):
    Q = {state: [0 for _ in range(env.action_space_size)] for state in env.state_space}
    rewards = []  # for storing cumulative rewards per episode

    for episode in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon)
        done = False
        episode_reward = 0

        while not done:
            next_state, reward, done = env.step(action)
            episode_reward += reward
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action

        rewards.append(episode_reward)

    return Q, rewards

def print_optimal_policy(Q, actions):
    optimal_policy = {}
    for state in Q:
        optimal_action_index = np.argmax(Q[state])
        optimal_policy[state] = actions[optimal_action_index]
    
    print('Optimal policy:')
    for state in optimal_policy:
        print(f'State: {state}, Optimal action: {optimal_policy[state]}')

def running_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    grid = [[' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
            [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
            [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
            ['S',-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 'G']]

    env = CliffWalkingEnvironment(grid)

    num_episodes = 500
    alpha = 0.5
    gamma = 0.95
    epsilon = 0.1

    Q_q_learning, rewards_q_learning = q_learning(env, num_episodes, alpha, gamma, epsilon)
    Q_sarsa, rewards_sarsa = sarsa(env, num_episodes, alpha, gamma, epsilon)

    print('Q-Learning Optimal Policy:')
    print_optimal_policy(Q_q_learning, env.actions)
    
    print('SARSA Optimal Policy:')
    print_optimal_policy(Q_sarsa, env.actions)

    plt.plot(rewards_q_learning, label='Q-Learning', color='red')
    plt.plot(rewards_sarsa, label='SARSA', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()


    window_size = 50  
    smoothed_cumulative_rewards_q_learning = running_average(rewards_q_learning, window_size)
    smoothed_cumulative_rewards_sarsa = running_average(rewards_sarsa, window_size)

    plt.plot(range(len(smoothed_cumulative_rewards_q_learning)), smoothed_cumulative_rewards_q_learning, color='red', label='Q-Learning')
    plt.plot(range(len(smoothed_cumulative_rewards_sarsa)), smoothed_cumulative_rewards_sarsa, color='blue', label='SARSA')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    plt.title('Smoothed Cumulative Rewards of Q-Learning and SARSA over Episodes')
    plt.show()

    

if __name__ == "__main__":
    main()