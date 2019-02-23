import torch
import gym

import matplotlib
import matplotlib.pyplot as plt

from itertools import count

from env import *
from agent import *
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCHSIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.005
TARGET_UPDATE = 10
REPLAY_MEMORY_CAPACITY = 3000


def plot_duration(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)


def optimize_model(batch_size, memory, policy_net, target_net, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)

    # I used torch.cat instead of torch.tensor because state is already tensor.
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.view(-1, 1))

    # Optimize the model
    optimizer = optim.RMSprop(policy_net.parameters())
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def main():
    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    env = gym.make('CartPole-v0').unwrapped
    env.reset()
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape

    policy_net = CnnQLearning(screen_height, screen_width).to(device)
    fixed_target_net = CnnQLearning(screen_height, screen_width).to(device)
    fixed_target_net.load_state_dict(policy_net.state_dict())
    fixed_target_net.eval()

    memory = ReplayMemory(REPLAY_MEMORY_CAPACITY)

    steps_done = 0
    episode_durations = []
    num_episodes = 600

    for i_episode in range(num_episodes):
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(policy_net, state, steps_done, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_END)
            _, reward, done, _ = env.step(action.item())  # TODO env step github code 보기
            reward = torch.tensor([reward], device=device) # TODO [] 왜 감싸는지 확인하기

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)

            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state
            steps_done += 1

            optimize_model(batch_size=BATCHSIZE, memory=memory, policy_net=policy_net, target_net=fixed_target_net, gamma=GAMMA)
            if done:
                episode_durations.append(t + 1)
                plot_duration(episode_durations)
                print(i_episode)
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            fixed_target_net.load_state_dict(policy_net.state_dict())

        if i_episode % 100 == 0:
            torch.save(policy_net.state_dict(), './model{}.pt'.format(i_episode))

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
