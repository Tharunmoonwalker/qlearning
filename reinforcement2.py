import gym 
import numpy as np

Learning_rate=0.1
Discount=0.95
EPISODES=25000
SHOW_EVERY=2000

Discrete_OS_size=[20]*2
discrete_os_win_size=None

q_table=np.random.uniform(low=-2, high=0, size=(Discrete_OS_size+[3]))


def get_discrete_state(state):
    discrete_state=(state-env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))



for episode in range(EPISODES):
    
    if episode != 0 and episode % SHOW_EVERY == 0:
        print(episode)
        env = gym.make("MountainCar-v0", render_mode="human")
        render = True
    else:
        env = gym.make("MountainCar-v0")
        render = False

    if discrete_os_win_size is None:
        discrete_os_win_size = (env.observation_space.high-env.observation_space.low)/Discrete_OS_size

    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done=False
    while not done:
        action=np.argmax(q_table[discrete_state])
        new_state, reward, terminated, truncated, _=env.step(action)
        done=terminated or truncated
        new_discrete_state=get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q=np.max(q_table[new_discrete_state])
            current_q=q_table[discrete_state+(action, )]
            new_q=(1-Learning_rate)*current_q+Learning_rate*(reward+Discount*max_future_q)
            q_table[discrete_state+(action, )]=new_q

        elif new_state[0]>=env.goal_position:
            print(f"we made it on {episode}th episode ")
            q_table[discrete_state+(action, )]=0

        discrete_state=new_discrete_state
    env.close()