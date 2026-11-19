# baseline.py  (General)
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# =============================================================================
# BASELINE  -  reference team, no tricks
# =============================================================================
# What it is: the course template (OffensiveReflexAgent + DefensiveReflexAgent),
# cleaned up. It only sees opponents within 5 cells, as the rules intend.
# Use it as the reference every new agent should beat.
#
# How it decides: every turn it scores each legal action with
#   value = features x weights   (a linear "reflex" evaluation of the next state)
# and takes the best one (ties broken at random). It only looks ONE move ahead.
#
# Attacker (OffensiveReflexAgent)
#   - Features: food left on the enemy side (fewer is better) and distance to
#     the nearest food. It ignores ghosts, so it often gets eaten.
#   - With 2 or fewer food left it walks back towards its start to score.
#
# Defender (DefensiveReflexAgent)
#   - Stays a ghost on our side and chases invaders it can SEE (<= 5 cells).
#   - Penalties for stopping and for reversing direction.
#
# Safety: each move has a 0.8 s budget (MOVE_TIME_BUDGET); if choosing a move
# raises an error it plays a random legal move instead of losing the game.
# Successor states are cached per move, so features and weights share them.
#
# Typical result: mostly ties against baseline_team (it is the same strategy).
# =============================================================================

import random
import time
import traceback

import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

# Contest time limits: each move should take under 1 s (slower moves get a
# warning, the third warning or a single move over 3 s forfeits the game) and
# register_initial_state has 15 s. Search code should stop before this budget,
# which leaves a margin for the framework's own work on the same clock.
MOVE_TIME_BUDGET = 0.8


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    Return the two agents that form the team, built with the given agent
    indices. `is_red` tells whether this is the red or the blue team.

    `first` and `second` name the agent classes to use. They can be
    overridden with --red_opts / --blue_opts when running capture.py, but
    the contest creates the team with the defaults, so the defaults must
    be the configuration we want to compete with.
    """
    return [AGENT_CLASSES[first](first_index), AGENT_CLASSES[second](second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    Base class for reflex agents: each turn, score every legal action with
    a linear combination of features and weights, and take the best one.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.turn_start = None
        self._successors = {}

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def get_action(self, game_state):
        """
        Start the move clock and clear the successor cache before choosing.
        An exception here would forfeit the game, so if choosing an action
        fails, print the error (it shows in the contest logs) and play a
        legal move instead.
        """
        self.turn_start = time.perf_counter()
        self._successors = {}
        try:
            return super().get_action(game_state)
        except Exception:
            traceback.print_exc()
            return self.fallback_action(game_state)
        finally:
            self._successors = {}

    def time_left(self):
        """
        Seconds left of MOVE_TIME_BUDGET for the current move.
        """
        return MOVE_TIME_BUDGET - (time.perf_counter() - self.turn_start)

    def fallback_action(self, game_state):
        """
        A safe legal action when the normal choice fails: any move but STOP.
        """
        actions = game_state.get_legal_actions(self.index)
        moves = [a for a in actions if a != Directions.STOP]
        return random.choice(moves or actions)

    def choose_action(self, game_state):
        """
        Pick an action with the highest evaluation, breaking ties at random.
        With two or fewer food pellets left to eat, head back to the start
        position instead to bank what we are carrying.
        """
        actions = game_state.get_legal_actions(self.index)

        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            return self.action_towards_start(game_state, actions)

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def action_towards_start(self, game_state, actions):
        """
        Return the action whose successor position is closest (by maze
        distance) to the start position. Ties go to the first such action.
        """
        best_action = None
        best_dist = float('inf')
        for action in actions:
            successor = self.get_successor(game_state, action)
            dist = self.get_maze_distance(self.start, successor.get_agent_position(self.index))
            if dist < best_dist:
                best_action = action
                best_dist = dist
        return best_action

    def get_successor(self, game_state, action):
        """
        Return the state after taking `action`, making sure the agent ends
        on a grid point (a move can stop halfway between two cells).
        Successors are cached for the current move, so asking again for the
        same state and action (e.g. from features and weights) is free.
        """
        key = (id(game_state), action)
        cached = self._successors.get(key)
        if cached is not None and cached[0] is game_state:
            return cached[1]
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            successor = successor.generate_successor(self.index, action)
        self._successors[key] = (game_state, successor)
        return successor

    def evaluate(self, game_state, action):
        """
        Score an action as the dot product of its features and weights.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Return a Counter of features for the state reached by `action`.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Return the weight of each feature, as a Counter or a dict.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Reflex agent that goes after the opponent's food: it prefers actions
    that eat food, then actions that get closer to the nearest pellet.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()

        # Less food left on the opponent's side means we ate some.
        features['successor_score'] = -len(food_list)

        if food_list:
            my_pos = successor.get_agent_state(self.index).get_position()
            features['distance_to_food'] = min(self.get_maze_distance(my_pos, food) for food in food_list)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Reflex agent that guards our side: it stays a ghost on our half and
    chases any visible invading Pacman, avoiding stopping or turning back.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # 1 while we stay on our own side as a ghost, 0 once we cross over.
        features['on_defense'] = 0 if my_state.is_pacman else 1

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if invaders:
            features['invader_distance'] = min(self.get_maze_distance(my_pos, a.get_position()) for a in invaders)

        if action == Directions.STOP:
            features['stop'] = 1
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


# Agent classes that create_team can build, by name.
AGENT_CLASSES = {cls.__name__: cls for cls in (ReflexCaptureAgent, OffensiveReflexAgent, DefensiveReflexAgent)}
