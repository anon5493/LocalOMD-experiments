import numpy as np
import math
from collections import defaultdict
from numba import njit

from open_spiel.python import policy
import pyspiel

from open_spiel.python.algorithms.exploitability import nash_conv
from agents.omd import OMDBase
from agents.fixed_sampling_omd import OMDFixedSampling
from agents.utils import sample_from_weights
from agents.utils import compute_log_sum_from_logit
from tqdm import tqdm


class BalancedISFree(OMDFixedSampling):
  
  """A class for the LocalOMD algorithm from "Local and adaptive mirror descents in extensive-form
games"
   """


  def __init__(
    self,
    game,
    budget,
    adaptation,
    base_constant=1.0,
    lr_constant=1.0,
    ix_constant=1.0,
    name=None
  ):

    OMDFixedSampling.__init__(
      self,
      game,
      budget,
      base_constant=base_constant,
      lr_constant=lr_constant,
      ix_constant=ix_constant,
      )
    
    
    self.name = 'BalancedISFree'
    if name:
      self.name = name
    
    self.adaptation=adaptation
    
    #Balanced policy
    self.compute_balanced()
    
    #Set rates
    if adaptation:
      self.learning_rates = self.base_learning_rate * np.ones(self.policy_shape[0])
      self.cumulative_squared_loss=np.ones(self.policy_shape[0])
    else:
      self.learning_rates = self.base_learning_rate * (self.total_actions_from_key.astype(float) ** (-1))
    
    #Set policy
    self.current_policy.action_probability_array=self.balanced_policy.copy()
    self.sampling_policy=self.balanced_policy.copy()
    
    self.current_logit=np.log(self.current_policy.action_probability_array,where=self.legal_actions_indicator)
    self.initial_logit=self.current_logit.copy()


  def compute_balanced(self):
      self.initial_keys=[]
      self.depth_from_key=np.zeros(self.policy_shape[0], dtype=int)
      self.current_player_from_key = np.zeros(self.policy_shape[0], dtype=int)
      self.total_actions_from_key = np.zeros(self.policy_shape[0], dtype=int)
      self.total_actions_from_action = np.zeros(self.policy_shape)
      self.legal_actions_from_key = [[] for i in range(self.policy_shape[0])]
      self.balanced_policy=np.zeros(self.policy_shape)
      
      self.key_children = [defaultdict(list) for i in range(self.policy_shape[0])] #key_children gives for each state_key a dictionnary that associates to each action the list of children 
      
      self.compute_information_tree_from_state(self.game.new_initial_state(),[[],[]],[0,0])
      for initial_key in self.initial_keys:
        self.compute_balanced_policy_from_key(initial_key)

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
    if self.total_actions_from_key[state_key] == 0:
        self.current_player_from_key[state_key] = current_player
        self.legal_actions_from_key[state_key] = legal_actions
        self.depth_from_key[state_key]=h
        
        if len(trajectory[current_player]) == 0:
          self.initial_keys.append(state_key)
        else:
          self.key_children[trajectory[current_player][-1][0]][trajectory[current_player][-1][1]].append(state_key)
        
        self.total_actions_from_key[state_key]=number_legal_actions
        for action in legal_actions:
          self.total_actions_from_action[state_key, action]=1
        for parent_couple in trajectory[current_player]:
          self.total_actions_from_key[parent_couple[0]] += number_legal_actions
          self.total_actions_from_action[parent_couple[0],parent_couple[1]]+=number_legal_actions
          
    depth[current_player]=h+1
    for action in legal_actions:
      trajectory[current_player].append([state_key,action])
      self.compute_information_tree_from_state(state.child(action), trajectory, depth)
      trajectory[current_player].pop()
    depth[current_player]=h
    
  def compute_balanced_policy_from_key(self, state_key):
    for action in self.legal_actions_from_key[state_key]:
      self.balanced_policy[state_key,action]=self.total_actions_from_action[state_key,action]/self.total_actions_from_key[state_key]
      for state_key_child in self.key_children[state_key][action]:
        self.compute_balanced_policy_from_key(state_key_child)
        
        
  #Change the update function to introduce the adaptation of the learning rate
  def update(self, trajectory):

    #Initialize values
    value =  0

    for transition in reversed(trajectory):
        player, state_idx, action_idx, plan, loss = transition.values()
      
        policy = self.current_policy.action_probability_array[state_idx,:]
        
        adjusted_loss = (loss - value)/self.sampling_policy[state_idx,action_idx]  
      
        #Update lr:
        if self.adaptation:
          old_lr=self.learning_rates[state_idx]
          self.cumulative_squared_loss[state_idx]+= adjusted_loss ** 2
          lr=self.base_learning_rate/math.sqrt(self.cumulative_squared_loss[state_idx])
          self.learning_rates[state_idx]=lr
          alpha=lr/old_lr
        
        else:
          lr=self.learning_rates[state_idx]
          alpha=1
  
        #Compute new policy 
        legal_actions=self.legal_actions_indicator[state_idx,:]
        self.current_logit[state_idx,:]=alpha*self.current_logit[state_idx,:]+(1-alpha)*self.initial_logit[state_idx,:]
        self.current_logit[state_idx,action_idx]-=lr*adjusted_loss
        logz=compute_log_sum_from_logit(self.current_logit[state_idx,:],legal_actions)
        self.current_logit[state_idx,:]-=logz*legal_actions
        value = logz/lr
        new_policy=np.exp(self.current_logit[state_idx,:],where=legal_actions)*legal_actions
  
        #Update new policy 
        self.set_current_policy(state_idx, new_policy)