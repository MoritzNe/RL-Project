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

# Initializing
pygame.init()

# Setting up FPS
FPS = 180 #180
FramePerSec = pygame.time.Clock()

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# Creating colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Other Variables for use in the program
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
SPEED = 3

font_small = pygame.font.SysFont("Verdana", 20)
# Create a white screen
DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
DISPLAYSURF.fill(WHITE)
pygame.display.set_caption("Game")


class DQN(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        #self.fc3 = nn.Linear(in_features=4, out_features=4)
        #self.fc4 = nn.Linear(in_features=2, out_features=2)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, input):
        input = F.rrelu(self.fc1(input))
        input = F.rrelu(self.fc2(input))
        #input = F.rrelu(self.fc3(input))
        #input = F.leaky_relu(self.fc4(input))
        output = self.out(input)
        return output


class Player(pygame.sprite.Sprite):

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

        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[K_LEFT]:
            if self.rect.left > 0:
                self.rect.move_ip(-3, 0)
        if pressed_keys[K_RIGHT]:
            if self.rect.right < SCREEN_WIDTH:
                self.rect.move_ip(3, 0)

    def select_action(self, state, policy):
        if counter % 2 == 0:
            rate = self.strategy.get_exploration_rate(self.current_step)
            self.current_step += 1

            if rate > random.random():
                return random.randrange(self.num_actions), rate
            else:
                self.strategy.cnt += 1
                with torch.no_grad():
                    network_output = policy(state)
                    #print(network_output)
                    return network_output.argmax(dim=0).item(), rate
        else:
            if 0.3 > random.random():
                return random.randrange(self.num_actions), 0.3
            else:
                with torch.no_grad():
                    network_output = policy(state)
                    return network_output.argmax(dim=0).item(), 0.3


class Item(pygame.sprite.Sprite):

    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("item.png")
        self.surf = pygame.Surface((30, 30))
        self.rect = self.surf.get_rect(center=(np.random.randint(8, SCREEN_WIDTH - 8), np.random.randint(165,235)))


class ReplayMemory():

    def __init__(self, memory_size, replaceable_memory):
        self.memory_size = memory_size
        self.memory = []
        self.push_count = 0
        self.first_overflow = 1
        self.replaceable_memory = replaceable_memory

    def push(self, experience):
        if len(self.memory) < self.memory_size:
            self.memory.append(experience)
        else:
            if self.first_overflow:
                self.memory_size = int(np.round((1 - self.replaceable_memory) * self.memory_size))
                self.first_overflow = 0
                print('overflowed')
            self.memory[self.push_count % self.memory_size] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size


class ExploreExploit():

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.cnt = 0

    def get_exploration_rate(self, current_step):
        expl_rate = self.end + (self.start - self.end) * np.exp(-1. * counter * self.decay)
        return expl_rate


def qvals_get_current(policy_net, prev_states, prev_actions):
    #print('-------')
    #print('prev act', prev_actions)
    curr = policy_net(prev_states)
    #print('nn out', curr)
    #print('gathered', curr.gather(dim=-1, index=prev_actions.unsqueeze(1)))
    return curr.gather(dim=-1, index=prev_actions.unsqueeze(1))

def qvals_get_next(target_net, states):
    return target_net(states).max(dim=1)[0].detach()

def extract_tensors(experiences):

    batch = Experience(*zip(*experiences))
    t1 = torch.stack(batch.prev_state)
    t2 = torch.cat(batch.prev_action)
    t3 = torch.cat(batch.reward)
    t4 = torch.stack(batch.state)
    return (t1, t2, t3, t4)

def weights_init_normal(policy_net):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''
    classname = policy_net.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = policy_net.in_features
        # m.weight.data shoud be taken from a normal distribution
        policy_net.weight.data.normal_(0.0, 1 / np.sqrt(y))
        # m.bias.data should be 0
        policy_net.bias.data.fill_(0)
    return policy_net

Experience = namedtuple(
    'Experience',
    ('prev_state', 'prev_action', 'reward', 'state')
)

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
eps_end = 0.3#0.01
eps_decay = 0.005#0.02
global counter
counter = 0
prev_reward = 0
rewards_for_avg = []
done = False
target_every = 5
rate = 0
replaceable_memory = 0.5

for_plotting = []

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
#policy_net = weights_init_normal(policy_net)

target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net = target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
#optimizer = optim.SGD(params=policy_net.parameters(), lr=lr)
game_over = 0
p_i_collision = 0
pygame.key.set_repeat(700)

while True:

    current_reward = 0
    if game_over or pygame.time.get_ticks() - timer > 5000:#20000:
        if counter % 2 != 0:
            for_plotting.append(reward)

        prev_reward = reward
        rewards_for_avg.append(prev_reward)

        print('last reward: ' + str(np.round(prev_reward, 2)))
        print('generation: ' + str(counter) + '(' + str(counter // target_every) + ')')
        print('average reward: ' + str(np.round(np.mean(rewards_for_avg) if len(rewards_for_avg) > 0 else 0, 2)))
        if counter % 2 == 0:
            print('exploration rate: ' + str(np.round(rate, 2)))
        else:
            print('exploration rate: 0.3')
        print('=====================')

        P1.kill()
        I1.kill()

        if counter % 2 == 0:
            policy_net = policy_net.train()
            if mem.can_sample(batch_size):
                for i in range(epochs):
                    batch = mem.sample(batch_size)
                    #print(batch)
                    prev_states, prev_actions, rewards, states = extract_tensors(batch)
                    curr_qvals = qvals_get_current(policy_net, prev_states, prev_actions)
                    next_qvals = qvals_get_next(target_net, states)
                    target_qvals = (next_qvals * gamma) + rewards
                    for i in range(batch_size):
                        if rewards[i] == -1:
                            target_qvals[i] = rewards[i]
                    #for_plotting.append(curr_qvals.view(-1))
                    loss = criterion(curr_qvals, target_qvals.unsqueeze(1))
                    #print('loss', loss)
                    #print('cur', curr_qvals)
                    #print('trg', target_qvals.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if mem_important.can_sample(batch_size_important):
                for i in range(epochs_important):
                    batch = mem_important.sample(batch_size_important)
                    prev_states, prev_actions, rewards, states = extract_tensors(batch)
                    curr_qvals = qvals_get_current(policy_net, prev_states, prev_actions)
                    next_qvals = qvals_get_next(policy_net, states)
                    target_qvals = (next_qvals * gamma) + rewards
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

        if counter % target_every == 0 and counter > 0:
            target_net.load_state_dict(policy_net.state_dict())
            print('target loaded')

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
        p_i_collision = 1
        current_reward = 1 #np.random.uniform(0.5, 1.0)
        reward += current_reward
    else:
        reward -= 0.001
        current_reward = -0.001

    if P1.rect.left <= 3 or P1.rect.right >= SCREEN_WIDTH - 3:
        reward -= 1
        current_reward = -1
        game_over = 1

    state = (torch.Tensor([P1.rect.center[0], I1.rect.center[0]]) / 400 - 0.5)
    action, rate = P1.select_action(state, policy_net)

    if p_i_collision:
        for item in items:
            item.kill()
        I1 = Item()
        all_sprites.add(I1)
        items.add(I1)
        p_i_collision = 0

    if counter % 2 == 0:
        mem.push(Experience(prev_state, torch.LongTensor([prev_action]), torch.Tensor([current_reward]), state))
    if current_reward != 0 and current_reward != -0.001:
        mem_important.push(Experience(prev_state, torch.LongTensor([prev_action]), torch.Tensor([current_reward]), state))

    if action == 0:
        if P1.rect.left > 0:
            P1.rect.move_ip(-5, 0)
    if action == 1:
        if P1.rect.right < SCREEN_WIDTH:
            P1.rect.move_ip(5, 0)
    displ_curr_reward = font_small.render('current reward: ' + str(np.round(reward, 2)), True, BLACK)
    DISPLAYSURF.blit(displ_curr_reward, (10, 10))
    displ_last_reward = font_small.render('last reward: ' + str(np.round(prev_reward, 2)), True, BLACK)
    DISPLAYSURF.blit(displ_last_reward, (10, 30))
    displ_generation = font_small.render('generation: ' + str(counter) + '(' + str(counter // target_every) + ')', True, BLACK)
    DISPLAYSURF.blit(displ_generation, (10, 50))
    displ_avg_rewards = font_small.render('average reward: ' + str(np.round(np.mean(rewards_for_avg) if len(rewards_for_avg) > 0 else 0, 2)), True, BLACK)
    DISPLAYSURF.blit(displ_avg_rewards, (10, 70))
    if counter % 2 == 0:
        displ_avg_rewards = font_small.render('exploration rate: ' + str(np.round(rate, 2)), True, BLACK)
    else:
        displ_avg_rewards = font_small.render('exploration rate: 0.3', True, BLACK)
    DISPLAYSURF.blit(displ_avg_rewards, (10, 90))

    pygame.display.update()
    FramePerSec.tick(FPS)

    pressed_keys = pygame.key.get_pressed()

    if pressed_keys[K_UP] or counter == 3000:
        print(policy_net.state_dict())
        torch.save(policy_net.state_dict(), 'policy_net')
        print(P1.rect.centerx)
        print(I1.rect.centerx)
        f = plt.figure()
        plt.plot(for_plotting, label='reward')
        window_width = 10
        if len(for_plotting) > window_width:
            x = np.linspace(0, len(for_plotting), len(for_plotting))
            for_plotting_padded = np.pad(for_plotting, (window_width // 2, window_width - 1 - window_width // 2), mode='edge')
            for_plotting_smoothed = np.convolve(for_plotting_padded, np.ones((window_width,)) / window_width, mode='valid')
            plt.plot(x, for_plotting_smoothed, color='red', label='running avg.')
        plt.xlabel('iteration')
        plt.ylabel('reward')
        plt.title('agent\'s reward at the iteration i')
        #plt.legend()
        f.savefig("rewards.pdf")
        if counter == 3000:
            break

    #print(pygame.surfarray.array3d(DISPLAYSURF))
    #pygame.image.save(DISPLAYSURF, 'screen.png')
    #break