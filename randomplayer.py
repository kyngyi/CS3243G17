from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import random as rand
import pprint

class RandomPlayer(BasePokerPlayer):

  def declare_action(self, valid_actions, hole_card, round_state):
     #valid_actions format => [raise_action_pp = pprint.PrettyPrinter(indent=2)
    #pp = pprint.PrettyPrinter(indent=2)
    #print("------------ROUND_STATE(RANDOM)--------")
    #pp.pprint(round_state)
    #print("------------HOLE_CARD----------")
    #pp.pprint(hole_card)
    #print("------------VALID_ACTIONS----------")
    #pp.pprint(valid_actions)
    r = rand.random()
    if r <= 0.5:
      call_action_info = valid_actions[1]
    elif r<= 0.9 and len(valid_actions ) == 3:
      call_action_info = valid_actions[2]
    else:
      call_action_info = valid_actions[0]
    action = call_action_info["action"]
    return action  # action returned here is sent to the poker engine


  def pass_r_value(self, hole_card, round_state):
    r = estimate_hole_card_win_rate(nb_simulation=1000, nb_player=2, hole_card=gen_cards(hole_card),
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
    # pp = pprint.PrettyPrinter(indent=2)
    # print("POST ROUND RESULT")
    # print("--------Winners---------")
    # pp.pprint(winners)
    # print("--------HAND INFO---------")
    # pp.pprint(hand_info)
    # print("--------ROUND STATE---------")
    # pp.pprint(round_state)
    # print("-------------------------------")
    pass

def setup_ai():
  return RandomPlayer()
