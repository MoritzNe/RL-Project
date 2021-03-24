# Imports
import pygame, sys
from pygame.locals import *
import random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
#import dqn as plot_qvalues

# Initializing pygame environment
pygame.init()

# Setting up FPS
FPS = 180
FramePerSec = pygame.time.Clock()

# Device definition
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# Creating colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Screen and speed definition
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
SPEED = 3
font_small = pygame.font.SysFont("Verdana", 20)
DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
DISPLAYSURF.fill(WHITE)
pygame.display.set_caption("Game")


class DQN(nn.Module):
    """
    Class of the neural net used by the agent.
    Same for both policy and target.
    Layers can be uncommented in order to change the architecture.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        # self.fc3 = nn.Linear(in_features=4, out_features=4)
        # self.fc4 = nn.Linear(in_features=2, out_features=2)
        self.out = nn.Linear(in_features=32, out_features=2)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, input):
        input = (F.rrelu(self.fc1(input)))
        input = (F.rrelu(self.fc2(input)))
        # input = F.rrelu(self.fc3(input))
        # input = F.leaky_relu(self.fc4(input))
        output = self.out(input)
        return output


class Player(pygame.sprite.Sprite):
    """
    Class of the player (the spaceship).
    Has methods for deciding where to move and for actually moving
    """

    def __init__(self, strategy):
        super().__init__()
        self.image = pygame.image.load("download.png")
        self.surf = pygame.Surface((50, 50))
        self.rect = self.surf.get_rect(center=(200, 200))
        self.actions = [0, 1]
        self.strategy = strategy
        self.num_actions = len(self.actions)
        self.current_step = 0

    def move(self):
        """
        Move the player if being controlled from keyboard
        For debugging purposes
        """
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[K_LEFT]:
            if self.rect.left > 0:
                self.rect.move_ip(-3, 0)
        if pressed_keys[K_RIGHT]:
            if self.rect.right < SCREEN_WIDTH:
                self.rect.move_ip(3, 0)

    def select_action(self, state, policy):
        """
        Decide where to move given current state and policy network.
        Every second round is exploiting learned strategies.
        When in exploration mode, gets exploration rate from eps-greedy strategy.
        """
        if counter % 2 == 0:  # if in exploration mode
            # Getting exploration rate which decays over time.
            rate = self.strategy.get_exploration_rate(self.current_step)
            self.current_step += 1
            if rate > random.random():  # if decided to take a random action (explore)
                return random.randrange(self.num_actions), rate
            else:  # if decided to exploit
                self.strategy.cnt += 1
                with torch.no_grad():
                    network_output = policy(state)
                    return network_output.argmax(dim=0).item(), rate
        else:  # if in exploitation mode
            if 0.3 > random.random():  # if decided to take a random action
                return random.randrange(self.num_actions), 0.3
            else:  # if decided to exploit
                with torch.no_grad():
                    network_output = policy(state)
                    return network_output.argmax(dim=0).item(), 0.3


class Enemy(pygame.sprite.Sprite):
    """
    Class of the enemy. Can move.
    'num' allows to select the axis.
    num=1 is top -> bottom enemy, num=2 is left -> right.
    """

    def __init__(self, num):
        super().__init__()
        self.image = pygame.image.load("enemy.png")
        self.surf = pygame.Surface((50, 50))
        self.num = num
        if self.num == 1:
            self.rect = self.surf.get_rect(
                center=(random.randint(25, SCREEN_WIDTH - 25), 0))
        elif self.num == 2:
            self.rect = self.surf.get_rect(
                center=(0, random.randint(25, SCREEN_HEIGHT - 25)))

    def move(self):
        if self.num == 1:  # if vertical enemy
            # Move the enemy
            self.rect.move_ip(0, SPEED)
            if (self.rect.top > SCREEN_HEIGHT):
                # if reached end of screen
                self.rect.top = 0
                self.rect.center = (random.randint(25, SCREEN_WIDTH - 25), 0)
                self.alien_reward = 1
            else:
                self.alien_reward = 0
        elif self.num == 2:  # if horizontal enemy
            self.rect.move_ip(SPEED, 0)
            if (self.rect.right > SCREEN_WIDTH):
                # if reached end of screen
                self.rect.left = 0
                self.rect.center = (0, random.randint(25, SCREEN_HEIGHT - 25))
                self.alien_reward = 1
            else:
                self.alien_reward = 0


class Item(pygame.sprite.Sprite):
    """
    Class of the item.
    Includes initialisation of the instance.
    """

    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("item.png")
        self.surf = pygame.Surface((30, 30))
        self.rect = self.surf.get_rect(center=(np.random.randint(8, SCREEN_WIDTH - 8),
                                               np.random.randint(165, 235)))


class ReplayMemory():
    """
    Class of the replay memory which is used to store experiences for training.
    Can save an experience, sample and reply if it's
    able to sample the batch of required size.
    """

    def __init__(self, memory_size, replaceable_memory):
        self.memory_size = memory_size
        self.memory = []
        self.push_count = 0
        self.first_overflow = 1
        self.replaceable_memory = replaceable_memory

    def push(self, experience):
        """
        Saves received experience to self.memory if not full yet.
        If full, overwrites experiences from the beginning.
        Doesn't overwrite first (1 - self.replaceable_memory)*100% part of the samples.
        """

        if len(self.memory) < self.memory_size:
            self.memory.append(experience)
        else:
            if self.first_overflow:
                self.memory_size = int(np.round((1 - self.replaceable_memory) *
                                                self.memory_size))
                self.first_overflow = 0
                print('overflowed')
            self.memory[self.push_count % self.memory_size] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size


class ExploreExploit():
    """
    Exploration-Exploitation epsilon-greedy strategy.
    Can provide exploration rate given current_step variable.
    """

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.cnt = 0

    def get_exploration_rate(self, current_step):
        expl_rate = self.end + (self.start - self.end) * np.exp(-1. * counter * self.decay)
        return expl_rate


def qvals_get_current(policy_net, prev_states, prev_actions):
    """
    Calculates current q-values given policy network and previous states.
    prev_actions is used as index.
    """

    curr = policy_net(prev_states)
    return curr.gather(dim=-1, index=prev_actions.unsqueeze(1))


def qvals_get_next(target_net, states):
    """
    Calculates next q-values given target network and current state.
    """

    return target_net(states).max(dim=1)[0].detach()


def extract_tensors(experiences):
    """
    Stacks experiences into tensors.
    """

    batch = Experience(*zip(*experiences))
    t1 = torch.stack(batch.prev_state)
    t2 = torch.cat(batch.prev_action)
    t3 = torch.cat(batch.reward)
    t4 = torch.stack(batch.state)
    return (t1, t2, t3, t4)


def weights_init_normal(policy_net):
    """
    Initializes all linear layers in given NN with
    weight values taken from a normal distribution
    and zero biases.
    """

    classname = policy_net.__class__.__name__
    if classname.find('Linear') != -1:
        y = policy_net.in_features
        policy_net.weight.data.normal_(0.0, 1 / np.sqrt(y))
        policy_net.bias.data.fill_(0)
    return policy_net


# Other variables used throughout the program
Experience = namedtuple(
    'Experience',
    ('prev_state', 'prev_action', 'reward', 'state'))
reward = 0
state = torch.Tensor([0., 0.])
action = 0
timer = 0
batch_size = 256
batch_size_important = 16
gamma = 0.999
criterion = nn.MSELoss()
lr = 0.001
epochs = 10
epochs_important = 5
eps_start = 1.0
eps_end = 0.3
eps_decay = 0.005
global counter
counter = 0
prev_reward = 0
rewards_for_avg = []
done = False
target_every = 5
rate = 0
replaceable_memory = 0.5
for_plotting = []
game_over = 0
p_i_collision = 0
pygame.key.set_repeat(700)

# Creating class instances
strategy = ExploreExploit(eps_start, eps_end, eps_decay)
mem = ReplayMemory(10000, replaceable_memory)
mem_important = ReplayMemory(5000, replaceable_memory)
P1 = Player(strategy)
I1 = Item()
all_sprites = pygame.sprite.Group()
all_sprites.add(P1)
all_sprites.add(I1)
items = pygame.sprite.Group()
items.add(I1)
policy_net = DQN().to(device)
# policy_net = weights_init_normal(policy_net)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net = target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)  # , weight_decay=0.001)
# optimizer = optim.SGD(params=policy_net.parameters(), lr=lr)


# Game loop
while True:

    current_reward = 0
    # Condition to reset the game round: game over or running out of time.
    if game_over or pygame.time.get_ticks() - timer > 3000:  # 20000:
        if counter % 2 != 0:  # if in exploitation mode
            for_plotting.append(reward)
        prev_reward = reward
        rewards_for_avg.append(prev_reward)

        # Block of output information.
        print('last reward: ' + str(np.round(prev_reward, 2)))
        print('generation: ' + str(counter) + '(' + str(counter // target_every) + ')')
        print('average reward: ' + str(np.round(np.mean(rewards_for_avg)
                                                if len(rewards_for_avg) > 0 else 0, 2)))
        if counter % 2 == 0:  # if exploration
            print('exploration rate: ' + str(np.round(rate, 2)))
        else:  # if exploitation
            print('exploration rate: 0.3')
        print('=====================')

        # Clearing game
        for entity in all_sprites:
            entity.kill()

        if counter % 2 == 0:  # train when dying in exploration mode
            policy_net = policy_net.train()

            # Training on regular memory
            if mem.can_sample(batch_size):
                for i in range(epochs):
                    batch = mem.sample(batch_size)
                    prev_states, prev_actions, rewards, states = extract_tensors(batch)
                    # Calculating current and target q-values.
                    curr_qvals = qvals_get_current(policy_net, prev_states, prev_actions)
                    next_qvals = qvals_get_next(target_net, states)
                    target_qvals = (next_qvals * gamma) + rewards
                    # Setting terminal states (which led to player dying)
                    for i in range(batch_size):
                        if rewards[i] == -1:
                            target_qvals[i] = rewards[i]
                    loss = criterion(curr_qvals, target_qvals.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Training on 'important' memory
            if mem_important.can_sample(batch_size_important):
                for i in range(epochs_important):
                    batch = mem_important.sample(batch_size_important)
                    prev_states, prev_actions, rewards, states = extract_tensors(batch)
                    # Calculating current and target q-values.
                    curr_qvals = qvals_get_current(policy_net, prev_states, prev_actions)
                    next_qvals = qvals_get_next(policy_net, states)
                    target_qvals = (next_qvals * gamma) + rewards
                    # Setting terminal states (which led to player dying)
                    for i in range(batch_size_important):
                        if rewards[i] == -1:
                            target_qvals[i] = rewards[i]
                    loss = criterion(curr_qvals, target_qvals.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print('policy trained')
            policy_net = policy_net.eval()

        timer = pygame.time.get_ticks()
        reward = 0
        print(f'{len(mem_important.memory)} important samples')
        game_over = 0

        # Updating target net every 'target_every' iteration.
        if counter % target_every == 0 and counter > 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net = target_net.eval()
            print('target loaded')

        # Reinitializing player and item
        I1 = Item()
        all_sprites.add(I1)
        items.add(I1)
        P1 = Player(strategy)
        all_sprites.add(P1)

        counter += 1

    prev_state = state
    prev_action = action

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    DISPLAYSURF.fill(WHITE)

    for entity in all_sprites:
        DISPLAYSURF.blit(entity.image, entity.rect)
        if entity == P1:
            entity.move()

    if pygame.sprite.spritecollideany(P1, items):
        # Player picks up the item
        p_i_collision = 1
        current_reward = 1  # np.random.uniform(0.5, 1.0)
        reward += current_reward
    else:
        # Player doesn't pick up the item
        reward -= 0.001
        current_reward = -0.001

    """if P1.rect.left <= 3 or P1.rect.right >= SCREEN_WIDTH - 3:
        # Player touches a wall
        reward -= 1
        current_reward = -1
        game_over = 1"""

    # Current state.
    state = (torch.Tensor([P1.rect.center[0], I1.rect.center[0]]) / 400 - 0.5)
    # Choose an action given state and policy network
    action, rate = P1.select_action(state, policy_net)

    if p_i_collision:
        for item in items:
            item.kill()
        I1 = Item()
        all_sprites.add(I1)
        items.add(I1)
        p_i_collision = 0

    if counter % 2 == 0:  # writing to regular memory only in exploration mode
        mem.push(Experience(prev_state, torch.LongTensor([prev_action]),
                            torch.Tensor([current_reward]), state))
    if current_reward != 0 and current_reward != -0.001:
        # Write -1/+1 experiences to 'important' memory in any mode
        mem_important.push(Experience(prev_state, torch.LongTensor([prev_action]),
                                      torch.Tensor([current_reward]), state))

    # Move player given the taken action
    if action == 0:
        if P1.rect.left > 0:
            P1.rect.move_ip(-5, 0)
    if action == 1:
        if P1.rect.right < SCREEN_WIDTH:
            P1.rect.move_ip(5, 0)

    # Block for printing information on the display
    displ_curr_reward = font_small.render(
        'current reward: ' + str(np.round(reward, 2)), True, BLACK)
    DISPLAYSURF.blit(displ_curr_reward, (10, 10))
    displ_last_reward = font_small.render(
        'last reward: ' + str(np.round(prev_reward, 2)), True, BLACK)
    DISPLAYSURF.blit(displ_last_reward, (10, 30))
    displ_generation = font_small.render(
        'generation: ' + str(counter) + '(' +
        str(counter // target_every) + ')', True, BLACK)
    DISPLAYSURF.blit(displ_generation, (10, 50))
    displ_avg_rewards = font_small.render(
        'average reward: ' + str(np.round(np.mean(rewards_for_avg)
                                          if len(rewards_for_avg) > 0
                                          else 0, 2)), True, BLACK)
    DISPLAYSURF.blit(displ_avg_rewards, (10, 70))
    if counter % 2 == 0:
        displ_avg_rewards = font_small.render(
            'exploration rate: ' + str(np.round(rate, 2)), True, BLACK)
    else:
        displ_avg_rewards = font_small.render(
            'exploration rate: 0.3', True, BLACK)
    DISPLAYSURF.blit(displ_avg_rewards, (10, 90))

    pygame.display.update()
    FramePerSec.tick(FPS)
    pressed_keys = pygame.key.get_pressed()

    # Save reward plot, qvalues plot and the model
    # when reaching 3000 iterations or arrow_up key is pressed.
    if pressed_keys[K_UP] or counter == 3000:
        print(policy_net.state_dict())
        torch.save(policy_net.state_dict(), 'policy_net')
        #plot_qvalues.main()
        print(P1.rect.centerx)
        print(I1.rect.centerx)
        f = plt.figure()
        plt.plot(for_plotting, label='reward')
        window_width = 10
        if len(for_plotting) > window_width:
            # Calculating moving average
            x = np.linspace(0, len(for_plotting), len(for_plotting))
            for_plotting_padded = np.pad(
                for_plotting, (window_width // 2,
                               window_width - 1 - window_width // 2),
                mode='edge')
            for_plotting_smoothed = np.convolve(
                for_plotting_padded, np.ones((window_width,)) / window_width,
                mode='valid')
            plt.plot(x, for_plotting_smoothed, color='red',
                     label='running avg.')
        plt.xlabel('iteration')
        plt.ylabel('reward')
        plt.title('agent\'s reward at the iteration i')
        f.savefig("rewards.pdf")
        if counter == 3000:
            break
