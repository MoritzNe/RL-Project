#Imports
import pygame, sys
from pygame.locals import *
import random, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
#Initializing 
pygame.init()
 
#Setting up FPS 
FPS = 5000
FramePerSec = pygame.time.Clock()
 



device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


#Creating colors
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
 
#Other Variables for use in the program
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
SPEED = 3
PSPEED = 5 

font_small = pygame.font.SysFont("Verdana", 15)
#Create a white screen 
DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
DISPLAYSURF.fill(WHITE)
pygame.display.set_caption("Game")
 
decision = 0
reward = 0
bg = pygame.image.load("background.png")
class Item(pygame.sprite.Sprite):
      def __init__(self):
        super().__init__() 
        self.image = pygame.image.load("item.png")
        self.surf = pygame.Surface((30, 30))
        self.rect = self.surf.get_rect(center = (random.randint(15,SCREEN_WIDTH-15), random.randint(15,SCREEN_HEIGHT-15)))
      def move(self):
          donothing=0

      def return_pos(self):
          return self.rect.center 

class Enemy(pygame.sprite.Sprite):
    
      def __init__(self):
        super().__init__() 
        self.image = pygame.image.load("enemy.png")
        self.surf = pygame.Surface((50, 50))
        self.rect = self.surf.get_rect(center = (random.randint(30, 570), 0))#SCREEN_HEIGHT)))
        self.reward = reward
      def move(self):
        self.rect.move_ip(0,SPEED)
        if (self.rect.top > 600):
            self.rect.top = 0
            self.rect.center = (random.randint(30, 570), 0)#random.randint(30, 570)
            
            self.reward += 1
        
      def return_pos(self):
          return self.rect.center
      
      def return_reward(self):
          return self.reward
      
class Enemy_2(pygame.sprite.Sprite):
    
      def __init__(self):
        super().__init__() 
        self.image = pygame.image.load("enemy.png")
        self.surf = pygame.Surface((50, 50))
        self.rect = self.surf.get_rect(center = (0, random.randint(25,SCREEN_HEIGHT-25)))#SCREEN_HEIGHT)))random.randint(25,SCREEN_HEIGHT-25)
        self.reward = reward
      def move(self):
        self.rect.move_ip(SPEED,0)
        if (self.rect.left > 600):
            self.rect.left = 0
            self.rect.center = (0, random.randint(25,SCREEN_HEIGHT-25))
            
            self.reward += 1
        
      def return_pos(self):
          return self.rect.center
      
      def return_reward(self):
          return self.reward


          
 
class Player(pygame.sprite.Sprite):
    
    def __init__(self, policy_network=None, target_network=None, limit=5000, gamma=0.95, lr=0.0001, memory=[]):
        super().__init__() 
        self.image = pygame.image.load("download.png")
        self.surf = pygame.Surface((50, 50))
        self.rect = self.surf.get_rect(center = (300, 300))
        self.brain = nn.Sequential(
                        nn.Linear(8, 20),
                        nn.ReLU(),
                        nn.Linear(20,50),
                        nn.ReLU(),
                        nn.Linear(50,50),
                        nn.ReLU(),
                        nn.Linear(50,20),
                        nn.ReLU(),
                        nn.Linear(20, 5),
                        #nn.Sigmoid()
                        ).to(device)
        self.target = nn.Sequential(
                        nn.Linear(8, 20),
                        nn.ReLU(),
                        nn.Linear(20,50),
                        nn.ReLU(),
                        nn.Linear(50,50),
                        nn.ReLU(),
                        nn.Linear(50,20),
                        nn.ReLU(),
                        nn.Linear(20, 5),
                        #nn.Sigmoid()
                        ).to(device)
        
        if policy_network!=None:
            self.brain.load_state_dict(policy_network)
        if target_network!=None:
            self.target.load_state_dict(target_network)
            
        self.limit = limit   
        self.memory = memory
        self.decision = 0
        self.gamma = gamma
        self.optim = optim.Adam(params=self.brain.parameters(),lr=lr)
        self.rew_sum = 0
        self.explore = True
        self.randomc=1
    
    def memory_cap(self):
        return len(self.memory)/self.limit
    def make_choise(self, state):
        if self.explore==True:
            
            pred = self.randomc
            #print("random")
        else: pred = torch.argmax(self.brain(state))
        #print(self.brain(state))
        self.decision = pred
        return pred
    
    def set_exp(self, set_):
        self.explore=set_
        self.randomc = random.randint(0, 4)
        
    def improve(self, batch_size):
        if self.can_provide(batch_size)==True:
            for j in range(5):
                sample = self.random_sample(batch_size)
                #print(sample)
                state_T, action_T, reward_T, next_state_T = [],[],[],[]
                for i in range(batch_size):
                    state, action, reward, next_state = sample[i][0],sample[i][1],sample[i][2],sample[i][3]
                    state_T.append(state)
                    action_T.append(action)
                    reward_T.append(reward)
                    next_state_T.append(next_state)
                
                state_T = torch.Tensor(state_T)
                action_T = torch.Tensor(action_T)
                reward_T = torch.Tensor(reward_T)
                next_state_T = torch.Tensor(next_state_T)
                #print(action_T.type(torch.int64))
                current_Q = self.give_current(state_T, action_T)
                next_Q = self.get_next(next_state_T)
            
                target_v = (next_Q * self.gamma) + reward_T
                loss = F.l1_loss(current_Q, target_v.unsqueeze(1))
                #print(loss)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                return loss.item()
            #print(current_Q)

            #print(len(state_T))
            #print(sample.shape)
            #state, action, reward, next_state = sample[0]
    def give_current(self, state, action):
        pred = self.brain(state).gather(dim=1, index=action.type(torch.int64).unsqueeze(-1))
        return pred
    
    def get_next(self, next_states):
        values = self.target(next_states).max(dim=1)[0].detach()
        
        
        return values
    def get_rs(self):
        return self.rew_sum
    def add_memory(self, experience):
        self.memory.append(experience)
        self.rew_sum += experience[2]
        if len(self.memory) > self.limit:
            self.memory.pop(0)
        #print(len(self.memory))

    def can_provide(self, batch_size):
        return len(self.memory) >= batch_size
    
    def random_sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
        
    def get_networks(self):
        return self.brain.state_dict(), self.target.state_dict()
    
    def get_memory(self):
        return self.memory
    #def move(self):
    #    pressed_keys = pygame.key.get_pressed()
     #   if self.rect.top > 0:
      #      if pressed_keys[K_UP]:
       #         self.rect.move_ip(0, -PSPEED)
   #     if self.rect.bottom < SCREEN_HEIGHT:
    #        if pressed_keys[K_DOWN]:
     #           self.rect.move_ip(0,PSPEED)
      #   
       # if self.rect.left > 0:
  #            if pressed_keys[K_LEFT]:
   #               self.rect.move_ip(-PSPEED, 0)
    #    if self.rect.right < SCREEN_WIDTH:        
     #         if pressed_keys[K_RIGHT]:
      #            self.rect.move_ip(PSPEED, 0)
                  
#    def move(self):
#        #pressed_keys = pygame.key.get_pressed()
#        if self.rect.top > 0:
#            if self.decision==0:
#                self.rect.move_ip(0, -PSPEED)
#        if self.rect.bottom < SCREEN_HEIGHT:
#            if self.decision==1:
#                self.rect.move_ip(0,PSPEED)
#         
#        if self.rect.left > 0:
#              if self.decision==2:
#                  self.rect.move_ip(-PSPEED, 0)
#        if self.rect.right < SCREEN_WIDTH:        
#              if self.decision==3:
#                  self.rect.move_ip(PSPEED, 0)
    def return_pos(self):
        return self.rect.center
    
    def move(self):
        #pressed_keys = pygame.key.get_pressed()

        if self.decision==0:
            self.rect.move_ip(0, -PSPEED)

        if self.decision==1:
            self.rect.move_ip(0,PSPEED)
         

        if self.decision==2:
            self.rect.move_ip(-PSPEED, 0)
   
        if self.decision==3:
            self.rect.move_ip(PSPEED, 0)
            
            
    def return_pos(self):
        return self.rect.center

def process_pos(P1):
    positions = list(P1.return_pos())
    positions[0]=(positions[0]/600)-0.5
    positions[1]=(positions[1]/600)-0.5
    return positions[0], positions[1]

def create_experience(P1, E1, E2, I1, action, reward, P1_, E1_, E2_, I1_):
    p1_pos, p1_pos1 = process_pos(P1)
    e1_pos, e1_pos1 = process_pos(E1)
    e2_pos, e2_pos1 = process_pos(E2)
    i1_pos, i1_pos1 = process_pos(I1)
    p1_pos_, p1_pos1_ = process_pos(P1_)
    e1_pos_, e1_pos1_ = process_pos(E1_)
    e2_pos_, e2_pos1_ = process_pos(E2_)
    i1_pos_, i1_pos1_ = process_pos(I1_)
    experience=[[p1_pos, p1_pos1, e1_pos, e1_pos1, e2_pos, e2_pos1, i1_pos, i1_pos1], 
                action, reward, 
                [p1_pos_, p1_pos1_, e1_pos_, e1_pos1_, e2_pos_, e2_pos1_, i1_pos_, i1_pos1_]]
    
    return experience

def give_state(P1, E1, E2, I1):
    p1_pos, p1_pos1 = process_pos(P1)
    e1_pos, e1_pos1 = process_pos(E1)
    e2_pos, e2_pos1 = process_pos(E2)
    i1_pos, i1_pos1 = process_pos(I1)
    return(torch.Tensor([p1_pos, p1_pos1, e1_pos, e1_pos1, e2_pos, e2_pos1, i1_pos, i1_pos1]))
    
#Setting up Sprites        
P1 = Player()
E1 = Enemy()
E2 = Enemy_2()
I1 = Item()
#Creating Sprites Groups
enemies = pygame.sprite.Group()
enemies.add(E1)
enemies.add(E2)
all_sprites = pygame.sprite.Group()
all_sprites.add(P1)
all_sprites.add(E1)
all_sprites.add(E2)
all_sprites.add(I1)

#Adding a new User event 
INC_SPEED = pygame.USEREVENT + 1
pygame.time.set_timer(INC_SPEED, 2000)


def restart(policy, epsilon, try_, target, memory):
    update = False
    if try_%25==0:
        P1 = Player(policy_network=policy, target_network=policy, memory=memory)
        update=True

    else: P1 = Player(policy_network=policy, target_network=target, memory=memory)
    E1 = Enemy()
    E2 = Enemy_2()
    I1 = Item()
    #Creating Sprites Groups
    enemies = pygame.sprite.Group()
    enemies.add(E1)
    enemies.add(E2)
    all_sprites = pygame.sprite.Group()
    all_sprites.add(P1)
    all_sprites.add(E1)
    all_sprites.add(E2)
    all_sprites.add(I1)
                      
    SPEED = 3
    PSPEED = 5
    epsilon -= 0.01
    if epsilon <= 0.3:
        epsilon = 0.3
    
    return P1, E1, E2, I1, enemies, all_sprites, epsilon, SPEED, PSPEED, update


input_tensor = torch.tensor([])


#Game Loop
counter = -1
timer = 0
epsilon = 0.3
rew_sum=0
mean_rew=0
try_ = 1
version = 1
memory_save=0.1
update=False
train_c = 0
while True:
    counter+=1
    #Cycles through all events occuring  
    for event in pygame.event.get():
        if event.type == INC_SPEED:
            if SPEED <= 20:
                SPEED += 1
           
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
 
    
    DISPLAYSURF.blit(bg, (0,0))
    
    #save states
    P1_save = P1
    E1_save = E1
    E2_save = E2
    I1_save = I1
    #old_state_grid = create_input_grid(P1, E1, E2, I1)
    #action
    if counter%2==0:
        P1.set_exp(epsilon>random.uniform(0, 1))
        
    action = int(P1.make_choise(give_state(P1, E1, E2, I1)))
    if epsilon <= 0.9:
        epsilon += 0.0001
    if epsilon >= 0.9:
        epsilon += 0.00004
    if epsilon >= 1:
        epsilon = 1
    #Moves and Re-draws all Sprites
    for entity in all_sprites:
        DISPLAYSURF.blit(entity.image, entity.rect)
        entity.move()
    P1_save_ = P1
    E1_save_ = E1
    E2_save_ = E2
    I1_save_ = I1 
    

    if P1.rect.left==0 or P1.rect.right==SCREEN_WIDTH or P1.rect.top==0 or P1.rect.bottom==SCREEN_HEIGHT:
        touch = True
    else: touch=False
        
    #To be run if collision occurs between Player and Enemy
    if pygame.sprite.spritecollideany(P1, enemies) or touch==True:
          DISPLAYSURF.fill(RED)
          pygame.display.update()
          policy, target = P1.get_networks()
          memory = P1.get_memory()
          #memsave = memory.copy()
          inherit_memory = []
          #memcount = 0
          for i in range(len(memory)):
              if memory[i][2]!=0.001 or random.uniform(0, 1)<= memory_save:
                  inherit_memory.append(memory[i])
                  #memsave.pop(i)
                  #memcount += 1
          #memc2 = 0
          #for i in range(int(len(memsave)/10)):
         #     rand = random.randint(0, len(memsave)-1)
          #    
           #   inherit_memory.append(memsave[rand])
          #    memc2 += 1
          #print(memcount)
          #print(len(inherit_memory))   
          #print(len(inherit_memory))
          #print(len(inherit_memory))
          
          rew_sum = P1.get_rs()
          mean_rew += rew_sum
          try_ +=1
          for entity in all_sprites:
                entity.kill() 
          reward = -1
          #timer = 0
          
              
          
          
          
          
          P1, E1, E2, I1, enemies, all_sprites, epsilon, SPEED, PSPEED, update = restart(policy, epsilon, try_, target, inherit_memory)
          if update==True:
              version += 1
    else: reward=0.001
    
    #check for collision with Item

    
    if pygame.sprite.collide_rect(P1, I1):
        #PSPEED += 1
        I1.kill()
        I1 = Item()
        all_sprites.add(I1)
        reward = 1    

    P1.add_memory(create_experience(P1_save, E1_save, E2_save, I1_save, action, reward, P1_save_, E1_save_, E2_save_, I1_save_))
    loss=0.0
    if counter%500==0:
        train_c += 1
        loss = P1.improve(128)
        

    timer += reward
    scores = font_small.render("total score: "+str(np.round(timer, 1)), True, WHITE)      
    DISPLAYSURF.blit(scores, (10,50)) 
    exp_ = font_small.render("1/epsilon: "+str(np.round(epsilon, 3)), True, WHITE) 
    DISPLAYSURF.blit(exp_, (10,110)) 
    rews = font_small.render("last score: "+str(np.round(rew_sum*100,0)), True, WHITE) 
    DISPLAYSURF.blit(rews, (10,70)) 

    mrews = font_small.render("mean score: "+str(np.round(mean_rew*100/(try_), 1)), True, WHITE) 
    DISPLAYSURF.blit(mrews, (10,90)) 
    
    it = font_small.render("game no.: "+str(try_), True, WHITE) 
    DISPLAYSURF.blit(it, (10,10)) 
    
    DISPLAYSURF.blit(font_small.render("target v.: "+str(version), True, WHITE) , (10,30)) 
    
    DISPLAYSURF.blit(font_small.render("memory: "+str(np.round(P1.memory_cap(), 2)), True, WHITE) , (10,130)) 
    DISPLAYSURF.blit(font_small.render("traincount: "+str(train_c), True, WHITE) , (10,150)) 
    pygame.display.update()

    if update==True:
        print("==========")
        print("TARGET UPDATE")
        print("mean score: "+str(np.round(mean_rew*100/(try_), 1)))
    update=False
    FramePerSec.tick(FPS)
    