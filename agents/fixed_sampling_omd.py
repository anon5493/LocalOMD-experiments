import numpy as np
import math
from numba import njit

from open_spiel.python import policy
import pyspiel


from open_spiel.python.algorithms.exploitability import nash_conv
from agents.utils import sample_from_weights
from agents.utils import compute_log_sum_from_logit
from agents.omd import OMDBase
from tqdm import tqdm

class OMDFixedSampling(OMDBase):
  """A class for OMD family with fixed sampling,

  -base leaning rate is 
      lr_base = H**lr_pow_H* A**lr_pow_A * X**lr_pow_X * T**lr_pow_T
  -base implicit exploration is 
      ix_base = H**ix_pow_H* A**ix_pow_A * X**ix_pow_X * T**ix_pow_T
   """

  def __init__(
    self,
    game,
    budget,
    base_constant=1.0,
    lr_constant=1.0,
    ix_constant=1.0
  ):
    
    OMDBase.__init__(
      self,
      game,
      budget,
      base_constant=base_constant,
      lr_constant=lr_constant,
      ix_constant=ix_constant,
      )
      
  def sample_action_from_idx_from_sampling(self, state_idx, return_idx=False):
    '''Sample an action from the current policy at a state.
    '''
    probs = self.sampling_policy[state_idx,:]
    action_idx = sample_from_weights(list(range(probs.shape[0])), probs)
    action=action_idx
    if return_idx:
      return action, action_idx
    return action

  def sample_trajectory(self, step):
    plans =  np.ones(self.num_players)
    cum_plans = np.ones(self.num_players)*(step+1.0)
    trajectory = []
    state = self.game.new_initial_state()
    self.current_learning_player=step%self.num_players
    while not state.is_terminal():
      if state.is_chance_node():
        #Chance state
        outcomes_with_probs = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes_with_probs)
        action = sample_from_weights(action_list, prob_list)
        state.apply_action(action)
      else:
        #Current state
        current_player = state.current_player() 
        state_idx = self.state_index(state)
        
        #Sample action
        if self.current_learning_player==current_player:
          action, action_idx = self.sample_action_from_idx_from_sampling(state_idx, return_idx=True)
          #action, action_idx = self.sample_action_from_idx(state_idx, return_idx=True)
          #Update cumulative plans
          policy = self.get_current_policy(state_idx)
          self.cumulative_plan[state_idx,:] += (cum_plans[current_player]-self.cumulative_plan[state_idx,:].sum())* policy
          cum_plans[current_player] = self.cumulative_plan[state_idx, action_idx]
          #Update plans
          plans[current_player] *= policy[action_idx]
          #Record transition
          transition = {
            'player': current_player,
            'state_idx': state_idx,
            'action_idx': action_idx,
            'plan': plans[current_player],
            'loss': 0.0
          }
          trajectory += [transition]
        else:
          action, action_idx = self.sample_action_from_idx(state_idx, return_idx=True)
        #Apply action
        state.apply_action(action)

    #Compute loss
    losses = self.reward_to_loss(np.asarray(state.returns()))
    trajectory[-1]['loss'] = losses[trajectory[-1]['player']] 

    return trajectory
    
