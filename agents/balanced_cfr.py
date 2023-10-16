import numpy as np
from collections import defaultdict
from open_spiel.python import policy
import pyspiel
#from numba import njit

from agents.omd import OMDBase
from agents.utils import sample_from_weights
from agents.utils import compute_log_sum_from_logit
from agents.fixed_sampling_omd import OMDFixedSampling
from agents.ixomd import IXOMD

from open_spiel.python.algorithms.exploitability import nash_conv
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class BalancedCFR(OMDFixedSampling):
  """A class for the Balanced CFR algorithm from "Near-Optimal Learning of Extensive-Form Games
with Imperfect Information"
   """

  def __init__(
    self,
    game,
    budget,
    base_constant=1.0,
    lr_constant=1.0,
    ix_constant=1.0,
    name=None
  ):

    OMDBase.__init__(
      self,
      game,
      budget,
      base_constant=base_constant,
      lr_constant=lr_constant,
      ix_constant=ix_constant
      )

    self.name = 'BalancedCFR'
    if name:
      self.name = name

    #Balanced policy
    self.compute_balanced()

    self.current_observed_depth=0
    
    #Set rates
    self.learning_rates = self.base_learning_rate * np.ones(self.policy_shape[0])
    
    #Set policy
    self.current_policy.action_probability_array=self.uniform_policy
    self.current_logit=np.log(self.current_policy.action_probability_array,where=self.legal_actions_indicator)
    
  def compute_balanced(self):
      #H is only an upperbound of the max depth, in contrast with max_depth computed with the algorithm
      self.H=round(self.game.max_game_length())
      self.max_depth=0
      self.initial_keys=[]
      self.depth_from_key=np.zeros(self.policy_shape[0], dtype=int)
      self.current_player_from_key = np.zeros(self.policy_shape[0], dtype=int)
      self.total_actions_from_key = np.zeros((self.H,self.policy_shape[0]), dtype=int)
      self.total_actions_from_action = np.zeros((self.H,self.policy_shape[0],self.policy_shape[1]))
      self.legal_actions_from_key = [[] for i in range(self.policy_shape[0])]
      
      #The balanced policy is defined for each depth
      self.balanced_policy=np.zeros((self.H, self.policy_shape[0],self.policy_shape[1]))
      
      #The balanced plan is the same for all legal actions of the same state in Balanced OMD
      self.balanced_plan=np.zeros(self.policy_shape[0])
      self.key_children = [defaultdict(list) for i in range(self.policy_shape[0])] #key_children gives for each state_key a dictionnary that associates to each action the list of children 
      
      self.compute_information_tree_from_state(self.game.new_initial_state(),[[],[]],[0]*self.num_players)
      
      for initial_key in self.initial_keys:
        self.add_dummy_actions(initial_key)
        self.compute_balanced_policy_from_key(initial_key)
        self.compute_balanced_plan_from_key(initial_key,[1.0]*self.H)

  def compute_information_tree_from_state(self, state, trajectory, depth):
    if state.is_terminal():
        return
    if state.is_chance_node():
        for action, _ in state.chance_outcomes():
            self.compute_information_tree_from_state(state.child(action), trajectory, depth)
        return
    current_player = state.current_player()
    legal_actions = state.legal_actions(current_player)
    number_legal_actions=len(legal_actions)
    state_key = self.state_index(state)
    h=depth[current_player]
    
    if h>self.max_depth:
      self.max_depth=h
      
    if self.total_actions_from_key[h,state_key] == 0:
        self.current_player_from_key[state_key] = current_player
        self.legal_actions_from_key[state_key] = legal_actions
        self.depth_from_key[state_key]=h
        
        if len(trajectory[current_player]) == 0:
          self.initial_keys.append(state_key)
        else:
          self.key_children[trajectory[current_player][-1][0]][trajectory[current_player][-1][1]].append(state_key)
        
        self.total_actions_from_key[h,state_key]=number_legal_actions
        for action in legal_actions:
          self.total_actions_from_action[h,state_key, action]=1
        for parent_couple in trajectory[current_player]:
          self.total_actions_from_key[h,parent_couple[0]] += number_legal_actions
          self.total_actions_from_action[h,parent_couple[0],parent_couple[1]]+=number_legal_actions
          
    depth[current_player]=h+1
    for action in legal_actions:
      trajectory[current_player].append([state_key,action])
      self.compute_information_tree_from_state(state.child(action), trajectory, depth)
      trajectory[current_player].pop()
    depth[current_player]=h

  def compute_balanced_plan_from_key(self, state_key, current_plan):
    for action in self.legal_actions_from_key[state_key]:
        new_plan=[0]*self.H
        for h in range(self.depth_from_key[state_key],self.H):
            if self.total_actions_from_key[h,state_key]!=0:
                new_plan[h]=current_plan[h]*self.total_actions_from_action[h,state_key,action]/self.total_actions_from_key[h,state_key]
        self.balanced_plan[state_key]=new_plan[self.depth_from_key[state_key]]
        for state_key_child in self.key_children[state_key][action]:
            self.compute_balanced_plan_from_key(state_key_child,new_plan)
    
  #Recursivel adds one actions in the count for when the game stops early
  def add_dummy_actions(self, state_key):
    for action in self.legal_actions_from_key[state_key]:
      for h in range(self.H):
        if self.total_actions_from_action[h,state_key,action]==0:
          self.total_actions_from_action[h,state_key,action]+=1
          self.total_actions_from_key[h,state_key]+=1
      for child_key in self.key_children [state_key][action]:
        self.add_dummy_actions(child_key)
        
  def compute_balanced_policy_from_key(self, state_key):
    for action in self.legal_actions_from_key[state_key]:
      for h in range(self.H):
        self.balanced_policy[h,state_key,action]=self.total_actions_from_action[h,state_key,action]/self.total_actions_from_key[h,state_key]
      for child_key in self.key_children [state_key][action]:
        self.compute_balanced_policy_from_key(child_key)

  def sample_action_from_idx_from_sampling(self, state_idx, return_idx=False):
    '''Sample an action from the current policy at a state. According to the balanced policy up to the current observed depth, then according to the current policy
    '''
    if self.depth_from_key[state_idx] <= self.current_observed_depth:
      probs = self.balanced_policy[self.current_observed_depth,state_idx,:]
    else:
      probs = self.get_current_policy(state_idx)

    action_idx = sample_from_weights(list(range(probs.shape[0])), probs)
    action=action_idx
    if return_idx:
      return action, action_idx
    return action
  
  def update(self, trajectory):
    value=0
    for transition in reversed(trajectory):
        player, state_idx, action_idx, plan, loss = transition.values()
        
        current_player=player
        
        h=self.depth_from_key[state_idx]
        
        if h==self.current_observed_depth:
          policy = self.current_policy.action_probability_array[state_idx,:]
          
          adjusted_loss = (loss-value)/self.balanced_policy[h,state_idx,action_idx]  
          
          lr=self.learning_rates[state_idx]
        
          #Update lr:
    
          #Compute new policy 
          legal_actions=self.legal_actions_indicator[state_idx,:]
          self.current_logit[state_idx,action_idx]-=lr*adjusted_loss
          logz=compute_log_sum_from_logit(self.current_logit[state_idx,:],legal_actions)
          self.current_logit[state_idx,:]-=logz*legal_actions

          new_policy=np.exp(self.current_logit[state_idx,:],where=legal_actions)*legal_actions
    
          #Update new policy 
          self.set_current_policy(state_idx, new_policy)
        else:
          value-=loss
          
    #Change to the next depth after all players observed the current observed depth
    if current_player==self.num_players-1:
      self.current_observed_depth+=1
      if self.current_observed_depth==self.max_depth+1:
        self.current_observed_depth=0