from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from mlPlayer import MlPlayer

#TODO:config the config as our wish
config = setup_config(max_round=3, initial_stack=10000, small_blind_amount=10)

config.register_player(name="f1", algorithm=MlPlayer())
config.register_player(name="FT2", algorithm=RandomPlayer())


game_result = start_poker(config, verbose=1)
