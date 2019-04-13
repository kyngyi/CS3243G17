from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import random as rand
import pprint
import random
from keras.models import load_model
import numpy as np

class MlPlayer(BasePokerPlayer):

  static_weight = 0

  def __init__(self):
    self.r_value = []
    self.score = []
    self.model = load_model('mymodel.h5')

  def declare_action(self, valid_actions, hole_card, round_state):
    # valid_actions format => [raise_action_pp = pprint.PrettyPrinter(indent=2)
    # pp = pprint.PrettyPrinter(indent=2)
    # print("------------ROUND_STATE(RANDOM)--------")
    # pp.pprint(round_state)
    # print("------------HOLE_CARD----------")
    # pp.pprint(hole_card)
    # print("------------VALID_ACTIONS----------")
    # pp.pprint(valid_actions)
    # print("-------------------------------")
    r = estimate_hole_card_win_rate(nb_simulation=500, nb_player=2, hole_card=gen_cards(hole_card), community_card=gen_cards(round_state['community_card']))
    input = np.array([r])
    input.shape = (1,1)

    # 0 for fold, 1 for call, 2 for raise
    best_move = self.model.predict_classes(input)[0]

    # if player decides to raise and unable to raise, then call
    if ((len(valid_actions) != 3) and (best_move == 2)):
        best_move = 1


    call_action_info = valid_actions[best_move]

    # # based on weight, randomly have a chance to bluff (invert the estimated win rate)
    # if(self.static_weight > random.uniform(0, 1.0)):
    #   r = 1-r
    #
    # if r <= 0.5:
    #   call_action_info = valid_actions[1]
    # elif r<= 0.9 and len(valid_actions ) == 3:
    #   call_action_info = valid_actions[2]
    # else:
    #   call_action_info = valid_actions[0]

    action = call_action_info["action"]
    return action # action returned here is sent to the poker engine

  def pass_r_value(self, hole_card, round_state):
    r = estimate_hole_card_win_rate(nb_simulation=500, nb_player=2, hole_card=gen_cards(hole_card),
                                    community_card=gen_cards(round_state['community_card']))
    return r

  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass

def setup_ai():
  return RandomPlayer()
