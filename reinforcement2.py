import gym 
import numpy as np
import matplotlib.pyplot as plt

Learning_rate=0.2
Discount=0.95
EPISODES=2000
SHOW_EVERY=500

epsilon=1

START_EPSILON_DECAYING=1
END_EPSILON_DECAYING=EPISODES//2
epsilon_decay_value=epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)


Discrete_OS_size=[20]*2
discrete_os_win_size=None

q_table=np.random.uniform(low=-2, high=0, size=(Discrete_OS_size+[3]))
ep_rewards=[]
aggr_ep_rewards={'ep': [],'avg':[],'min':[],'max':[]}

def get_discrete_state(state):
    discrete_state=(state-env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))



for episode in range(EPISODES):
    episode_reward=0
    
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
        episode_reward+=reward
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

    if END_EPSILON_DECAYING>=episode>=START_EPSILON_DECAYING:
        epsilon-=epsilon_decay_value

    ep_rewards.append(episode_reward)

    
    if not episode % SHOW_EVERY:
        average=sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average)
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))

        print(f"episode:{episode},average{average},max{max(ep_rewards[-SHOW_EVERY:])},min{min(ep_rewards[-SHOW_EVERY:])})")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')

plt.legend(loc=4)
plt.show()
