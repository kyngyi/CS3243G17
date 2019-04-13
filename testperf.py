
import sys
sys.path.insert(0, './pypokerengine/api/')
import game
setup_config = game.setup_config
start_poker = game.start_poker
import time
from argparse import ArgumentParser
import random
import pprint


""" =========== *Remember to import your agent!!! =========== """
from randomplayer import RandomPlayer
from myplayer import MyPlayer
# from smartwarrior import SmartWarrior
""" ========================================================= """

""" Example---To run testperf.py with random warrior AI against itself. 

$ python testperf.py -n1 "Random Warrior 1" -a1 RandomPlayer -n2 "Random Warrior 2" -a2 RandomPlayer
"""



def testperf(agent_name1, agent1, agent_name2, agent2):

	# init a random generation of 10 entries
	pair1 = (random.uniform(0, 1.0), 0)
	pair2 = (random.uniform(0, 1.0), 0)
	generation = [pair1, pair2]
	for i in range(0, 8):
		pair1 = (random.uniform(0, 1.0), 0)
		generation.append(pair1)

	fit_weight = run_generation(generation, agent_name1, agent1, agent_name2, agent2)

	# we go through 10 generations
	for itr in range(0, 1):
		print("testing generation %d" % itr)
		pair1 = (fit_weight, 0)
		generation = [pair1, pair1]
		# mutate the rest of the generation
		mutation_rate = 0.5
		for i in range(1, 8):
			# we assume bluffing is better so a mutation would be to not bluff
			if (random.uniform(0, 1.0) < 0.4):
				pair2 = (max(fit_weight - random.uniform(0, 0.1), 0), 0)
				generation.append(pair2)
			else:
				pair2 = (min(fit_weight + random.uniform(0, 0.05),1.0),0)
				generation.append(pair2)
		fit_weight = run_generation(generation, agent_name1, agent1, agent_name2, agent2)

	print("Final best weight is %.3f" % fit_weight)



def run_generation(generation, agent_name1, agent1, agent_name2, agent2):

	for i in range(0, generation.__len__()):
		curr_weight, _ = generation[i]
		print("Testing %d in this generation" % i)
		# Init to play 500 games of 1000 rounds
		num_game = 1
		max_round = 1
		initial_stack = 10000
		smallblind_amount = 20

		# Init pot of players
		agent1_pot = 0
		agent2_pot = 0

		# Setting configuration
		config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=smallblind_amount)

		# Register players
		config.register_player(name=agent_name1, algorithm=MyPlayer(curr_weight))
		config.register_player(name=agent_name2, algorithm=MyPlayer(curr_weight))
		# config.register_player(name=agent_name1, algorithm=agent1())
		# config.register_player(name=agent_name2, algorithm=agent2())

		# Start playing num_game games

		for game in range(1, num_game + 1):
			print("Game number: ", game)
			game_result = start_poker(config, verbose=0)
			# pp = pprint.PrettyPrinter(indent=2)
			# pp.pprint(game_result)
			agent1_pot = agent1_pot + game_result['players'][0]['stack']
			agent2_pot = agent2_pot + game_result['players'][1]['stack']

		print("\n After playing {} games of {} rounds, the results are: ".format(num_game, max_round))
		# print("\n Agent 1's final pot: ", agent1_pot)
		print("\n " + agent_name1 + "'s final pot: ", agent1_pot)
		print("\n " + agent_name2 + "'s final pot: ", agent2_pot)

		# print("\n ", game_result)
		# print("\n Random player's final stack: ", game_result['players'][0]['stack'])
		# print("\n " + agent_name + "'s final stack: ", game_result['players'][1]['stack'])

		if (agent1_pot < agent2_pot):
			print("\n You lost")
		elif (agent1_pot > agent2_pot):
			print("\n You won!")
			print(generation[i])
		# print("\n Random Player has won!")
		else:
			print("\n It's a draw!")

		# update fitness function
		x, y = generation[i]
		generation[i] = (x, agent1_pot - agent2_pot)
		print("Updated generation")
		print(generation[i])

	# end of current generation, find most fit in the generation
	fit_index = -1
	fit_weight = -1
	max_fitness = -999999
	for i in range(0, generation.__len__()):
		_, y = generation[i]
		if (y > max_fitness):
			max_fitness = y
			fit_index = i
			fit_weight = x
	print("The most fit in this generation has weight %.3f and fitness %d" % (fit_weight, max_fitness))
	return fit_weight


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-n1', '--agent_name1', help="Name of agent 1", default="Your agent", type=str)
    parser.add_argument('-a1', '--agent1', help="Agent 1", default=RandomPlayer())    
    parser.add_argument('-n2', '--agent_name2', help="Name of agent 2", default="Your agent", type=str)
    parser.add_argument('-a2', '--agent2', help="Agent 2", default=RandomPlayer())    
    args = parser.parse_args()
    return args.agent_name1, args.agent1, args.agent_name2, args.agent2

if __name__ == '__main__':
	name1, agent1, name2, agent2 = parse_arguments()
	start = time.time()
	testperf(name1, agent1, name2, agent2)
	end = time.time()

	print("\n Time taken to play: %.4f seconds" %(end-start))