import gym
import numpy as np

# Create environment
env = gym.make("MountainCar-v0", render_mode="human")

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 1000  # Keep it small for debugging, increase later
SHOW_EVERY = 20  # Render every few episodes

# Discretization settings
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
print("Discrete Window Size:", discrete_win_size)

# Initialize Q-table
q_table = np.random.uniform(low=0, high=1, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print("Initial Q-table shape:", q_table.shape)

# Discretize a continuous state
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_win_size
    return tuple(discrete_state.astype(np.int32))

# Training loop
for episode in range(EPISODES):
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)

    done = False
    step = 0

    print(f"\n--- Episode: {episode} ---")
    print("Initial state (continuous):", state)
    print("Initial state (discrete):", discrete_state)

    while not done:
        # Choose action
        action = np.argmax(q_table[discrete_state])

        # Take action
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        new_discrete_state = get_discrete_state(new_state)

        # Update Q-value
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q


        elif new_state[0] >= env.goal_position:
            # If we reach the goal, set Q value to 0 (best outcome)
            q_table[discrete_state + (action,)] = 0
            print("Goal reached! Set Q-value to 0.")

        # Move to next state
        discrete_state = new_discrete_state
        step += 1

        # Optional rendering
        if episode % SHOW_EVERY == 0:
            pass  # env.render() already happens with render_mode="human"

env.close()
