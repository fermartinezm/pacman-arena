# full_view.py  (General)
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# =============================================================================
# FULL VIEW  -  baseline improved with the full world view
# =============================================================================
# The trick: the engine (game.py) gives every agent the FULL game state and
# trusts CaptureAgent.observation_function to hide distant opponents. This
# team overrides observation_function to return the state unfiltered, so it
# knows the exact position of every opponent at every turn.
#
# How it decides: same one-move reflex evaluation as the baseline
# (value = features x weights), with extra features that use that information.
#
# Attacker (OffensiveReflexAgent)
#   - Eats food (fewer food left, closer to the next one).
#   - 'danger': penalty when an active (not scared) ghost is within 5 cells.
#   - 'eaten': big penalty for moves that send it back to the start.
#   - Goes home (distance to our border) when carrying 5 food, or carrying
#     anything with a ghost close.
#
# Defender (DefensiveReflexAgent)
#   - Chases every invader (all of them are visible now).
#   - Scared: keeps a 2-cell margin from invaders.
#   - No invaders: moves towards the closest enemy while staying on our side.
#     Side effect: it MIRRORS the enemy across the border (face-off), which
#     blocks the game. hunter.py fixes this.
#
# Typical result: beats the baseline about half the time, many ties.
# =============================================================================

import random

import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    Returns the two agents of the team. `first` / `second` can be overridden
    from the command line with --redOpts / --blueOpts (e.g. first=DefensiveReflexAgent).
    The tournament calls this without extra arguments, so the defaults are what plays.
    """
    agents = {cls.__name__: cls for cls in (OffensiveReflexAgent, DefensiveReflexAgent)}
    return [agents[first](first_index), agents[second](second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    Base class for reflex agents that pick the action with the highest
    linear evaluation features * weights. Subclasses define the features.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.home_border = []

    def observation_function(self, game_state):
        """
        Full world view. The engine (game.py, run loop) hands every agent a full
        copy of the state and relies on CaptureAgent.observation_function to hide
        distant opponents via make_observation(). Returning the state unfiltered
        keeps the exact position of every opponent at every turn.
        """
        return game_state

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # Cells on our side of the middle line: the points where carried food is scored.
        walls = game_state.get_walls()
        border_x = walls.width // 2 - 1 if self.red else walls.width // 2
        self.home_border = [(border_x, y) for y in range(walls.height) if not walls[border_x][y]]

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)

        # With 2 or fewer food left the game is decided by returning home.
        if len(self.get_food(game_state).as_list()) <= 2:
            return min(actions, key=lambda a: self.distance_home(
                self.get_successor(game_state, a).get_agent_position(self.index)))

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        return random.choice([a for a, v in zip(actions, values) if v == max_value])

    def get_successor(self, game_state, action):
        """Successor state, advancing a second step if the move left us on a half grid position."""
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def evaluate(self, game_state, action):
        return self.get_features(game_state, action) * self.get_weights(game_state, action)

    def distance_home(self, pos):
        return min(self.get_maze_distance(pos, b) for b in self.home_border)

    def enemies(self, game_state):
        """All opponent states; thanks to observation_function every position is known."""
        return [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

    def get_features(self, game_state, action):
        raise NotImplementedError

    def get_weights(self, game_state, action):
        raise NotImplementedError


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Eats food while avoiding the defending ghosts (whose positions are always
    known) and goes home to score when it carries enough or a ghost gets close.
    """

    CARRY_LIMIT = 5      # food carried before heading home
    DANGER_RADIUS = 5    # maze distance at which a ghost starts to matter

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food(successor).as_list()

        features['successor_score'] = -len(food_list)
        if food_list:
            features['distance_to_food'] = min(self.get_maze_distance(my_pos, f) for f in food_list)

        # Active (non-scared) enemy ghosts on their own side.
        ghosts = [e for e in self.enemies(successor) if not e.is_pacman and e.scared_timer <= 1]
        ghost_dist = min((self.get_maze_distance(my_pos, g.get_position()) for g in ghosts), default=None)
        if ghost_dist is not None and my_state.is_pacman and ghost_dist < self.DANGER_RADIUS:
            features['danger'] = self.DANGER_RADIUS - ghost_dist

        # Being sent back to the start means we were eaten.
        if my_pos == self.start and self.get_maze_distance(game_state.get_agent_position(self.index), my_pos) > 1:
            features['eaten'] = 1

        carrying = game_state.get_agent_state(self.index).num_carrying
        if carrying >= self.CARRY_LIMIT or (carrying > 0 and features['danger'] > 0):
            features['go_home'] = 1
            features['distance_home'] = self.distance_home(my_pos)
        # evaluate() always calls get_weights right after get_features for the same action.
        self.going_home = bool(features['go_home'])

        if action == Directions.STOP:
            features['stop'] = 1
        return features

    def get_weights(self, game_state, action):
        weights = {'successor_score': 100, 'distance_to_food': -1, 'danger': -60,
                   'eaten': -1000, 'stop': -50}
        if self.going_home:
            weights.update({'distance_to_food': 0, 'successor_score': 0, 'distance_home': -5})
        return weights


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Stays on its own side, chases every invader (all of them are visible) and,
    when there are none, shadows the closest enemy from our side of the border.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        features['on_defense'] = 0 if my_state.is_pacman else 1

        enemies = self.enemies(successor)
        invaders = [e for e in enemies if e.is_pacman]
        features['num_invaders'] = len(invaders)

        if invaders:
            dist = min(self.get_maze_distance(my_pos, e.get_position()) for e in invaders)
            # While scared, touching an invader gets us eaten: keep a 2-cell margin.
            features['invader_distance'] = abs(dist - 2) if my_state.scared_timer > 0 else dist
        else:
            features['enemy_distance'] = min(self.get_maze_distance(my_pos, e.get_position()) for e in enemies)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10,
                'enemy_distance': -1, 'stop': -100, 'reverse': -2}
