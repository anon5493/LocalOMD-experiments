import os
from datetime import datetime


import yaml
import argparse
import gc

import pyspiel
import numpy as np

from agents.ixomd import IXOMD

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        help='Path to configuration file.')
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    #Environement
    game_name = config['game_name']
    game = pyspiel.load_game(game_name)

    #Global params
    budget = config['training_kwargs']['budget']

    #Agent params
    agent_kwargs = config['agent_kwargs']
    agent_kwargs['game'] = game
    lr = np.sqrt(budget)
    agent_kwargs['learning_rate'] = lr
    agent_kwargs['implicit_exploration'] = lr
    agent_kwargs['cf_prior'] = (lr**2)
    

    agent = IXOMD(**agent_kwargs)

    #Logs
    writer_path = None
    if 'writer_path' in config:
        now = datetime.now().strftime("%d-%m__%H:%M")
        writer_path = config['writer_path']
        writer_path = os.path.join(writer_path, game_name, agent.name, now)

    #Training
    training_kwargs = config['training_kwargs']
    training_kwargs['writer_path'] = writer_path

    try:
        agent.fit(**training_kwargs)
    except KeyboardInterrupt:
        gc.collect()
        pass
