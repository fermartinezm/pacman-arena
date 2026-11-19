# my_team.py — IW(1) Offensive Agent + Reflex Defensive Agent

import random
import time
import contest.util as util
from collections import deque

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveIWAgent', second='DefensiveReflexAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# IW(1)  #
##########

def get_atoms(game_state, agent_index):
    """
    Extract boolean atoms from a game state.
    Each atom is a (name, value) tuple. A state is "novel" if it produces
    an atom that has never been seen before in this search.
    """
    my_state = game_state.get_agent_state(agent_index)
    my_pos = nearest_point(my_state.get_position())
    atoms = set()

    # Atom 1: agent is at position (x, y)
    atoms.add(('at', my_pos))

    # Atom 2: for each food position, is food still there?
    food = game_state.get_blue_food() if game_state.is_on_red_team(agent_index) else game_state.get_red_food()
    for pos in food.as_list():
        atoms.add(('food', pos))

    # Atom 3: how many food dots the agent is carrying
    atoms.add(('carrying', my_state.num_carrying))

    return atoms


def iw1_search(game_state, agent_index, goal_test, max_depth=20, time_limit=0.8):
    """
    IW(1) search: BFS that prunes states which don't produce any new atom.

    Returns: list of actions to reach goal, or None if no path found.
    """
    start_time = time.time()

    # seen_atoms: set of all individual atoms seen so far across ALL explored states
    seen_atoms = get_atoms(game_state, agent_index)

    # BFS queue: each entry is (state, list_of_actions_to_reach_it)
    queue = deque()
    queue.append((game_state, []))

    best_partial = None  # best state found so far (closest to goal)
    best_partial_actions = []

    while queue:
        # Check time limit
        if time.time() - start_time > time_limit:
            break

        current_state, actions = queue.popleft()

        # Depth limit
        if len(actions) >= max_depth:
            continue

        # Try each legal action
        for action in current_state.get_legal_actions(agent_index):
            if action == Directions.STOP:
                continue

            # Generate successor state
            successor = current_state.generate_successor(agent_index, action)

            new_actions = actions + [action]

            # Goal test
            if goal_test(successor):
                return new_actions

            # Novelty test: does this state produce at least one NEW atom?
            new_atoms = get_atoms(successor, agent_index)
            novel = False
            for atom in new_atoms:
                if atom not in seen_atoms:
                    novel = True
                    seen_atoms.add(atom)

            # only add to queue if the state is novel
            if novel:
                queue.append((successor, new_actions))

    # No path to goal found — return best partial plan or None
    return best_partial_actions if best_partial_actions else None


##########################
# Offensive Agent (IW1)  #
##########################

class OffensiveIWAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.boundary = []

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.boundary = self._get_boundary_positions(game_state)
        self.position_history = []
        self.last_carrying = 0
        self.is_currently_stuck = False

    def _get_boundary_positions(self, game_state):
        walls = game_state.get_walls()
        mid_x = walls.width // 2
        x = mid_x - 1 if self.red else mid_x
        return [(x, y) for y in range(walls.height) if not walls[x][y]]

    def _get_home_distance(self, pos):
        return min([self.get_maze_distance(pos, b) for b in self.boundary])

    def _get_nearest_ghost_dist(self, game_state):
        """Returns distance to nearest dangerous ghost, or None."""
        my_pos = game_state.get_agent_state(self.index).get_position()
        dists = []
        for enemy in self.get_opponents(game_state):
            e = game_state.get_agent_state(enemy)
            if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0:
                dists.append(self.get_maze_distance(my_pos, e.get_position()))
        return min(dists) if dists else None

    def _update_stuck_status(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        carrying = game_state.get_agent_state(self.index).num_carrying

        self.position_history.append(my_pos)

        # Reset if we ate food
        if carrying != self.last_carrying:
            self.last_carrying = carrying
            self.position_history = []
            self.is_currently_stuck = False
            return

        # Check last 8 positions — are we oscillating?
        if len(self.position_history) >= 3:
            recent = self.position_history[-8:]
            unique_positions = set(recent)
            if len(unique_positions) <= 3:
                self.is_currently_stuck = True
                self.position_history = []  # reset so we try the new target for a while
                return

        self.is_currently_stuck = False

    def _pick_subgoal(self, game_state):
        """
        Decide what the agent should be doing right now.
        Returns: ('eat', food_pos) or ('home', None)
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food(game_state).as_list()

        # If carrying enough food or threatened while carrying, go home
        ghost_dist = self._get_nearest_ghost_dist(game_state)
        threatened = ghost_dist is not None and ghost_dist <= 5

        if my_state.num_carrying >= 5:
            return ('home', None)
        if threatened and my_state.num_carrying > 0:
            return ('home', None)
        if len(food_list) == 0:
            return ('home', None)

        if self.is_currently_stuck:
            target = max(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        else:
            target = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))

        return ('eat', target)

    def _make_goal_test(self, subgoal):
        """
        Returns a function that checks whether a game state satisfies the subgoal.
        """
        agent_index = self.index

        if subgoal[0] == 'eat':
            target_pos = subgoal[1]
            def goal_test(state):
                pos = nearest_point(state.get_agent_state(agent_index).get_position())
                # return True if the agent's position equals the target food position
                return pos == target_pos
            return goal_test

        else:  # 'home'
            boundary = self.boundary
            def goal_test(state):
                pos = nearest_point(state.get_agent_state(agent_index).get_position())
                return pos in boundary
            return goal_test

    def _reflex_fallback(self, game_state):
        """
        Simple reflex fallback: move toward nearest food or home, away from ghosts.
        Used when IW(1) fails or when immediately threatened.
        """
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()

        ghost_dist = self._get_nearest_ghost_dist(game_state)
        threatened = ghost_dist is not None and ghost_dist <= 5

        best_action = Directions.STOP
        best_score = float('-inf')

        current_direction = game_state.get_agent_state(self.index).configuration.direction
        reverse = Directions.REVERSE[current_direction]

        for action in actions:
            if action == Directions.STOP:
                continue
            successor = game_state.generate_successor(self.index, action)
            s_pos = successor.get_agent_state(self.index).get_position()
            score = 0

            if action == reverse:
                score -= 50

            if threatened:
                # Run away from ghost, toward home
                new_ghost_dist = self._get_nearest_ghost_dist(successor)
                if new_ghost_dist is not None:
                    score += new_ghost_dist * 10
                score -= self._get_home_distance(s_pos)
            else:
                # Move toward food
                food_list = self.get_food(successor).as_list()
                if food_list:
                    nearest = min(self.get_maze_distance(s_pos, f) for f in food_list)
                    score -= nearest

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def choose_action(self, game_state):
        """
        Main decision loop:
        1. Pick subgoal
        2. If immediately threatened → reflex escape
        3. Run IW(1) toward subgoal
        4. If IW(1) finds path → take first action
        5. If IW(1) fails → reflex fallback
        """

        self._update_stuck_status(game_state)

        ghost_dist = self._get_nearest_ghost_dist(game_state)
        if ghost_dist is not None and ghost_dist <= 3:
            # Emergency: ghost too close, no time to plan
            return self._reflex_fallback(game_state)

        subgoal = self._pick_subgoal(game_state)
        goal_test = self._make_goal_test(subgoal)

        plan = iw1_search(game_state, self.index, goal_test, max_depth=15, time_limit=0.8)

        if plan and len(plan) > 0:
            return plan[0]  # receding horizon: only execute first action
        else:
            return self._reflex_fallback(game_state)


############################
# Defensive Agent (reflex) #
############################

class DefensiveReflexAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.boundary = self._get_boundary_positions(game_state)

    def _get_boundary_positions(self, game_state):
            walls = game_state.get_walls()
            mid_x = walls.width // 2
            x = mid_x - 1 if self.red else mid_x
            return [(x, y) for y in range(walls.height) if not walls[x][y]]

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self._evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def _get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def _evaluate(self, game_state, action):
        features = self._get_features(game_state, action)
        weights = {'num_invaders': -1000, 'on_defense': 100,
                   'invader_distance': -10, 'stop': -100, 'reverse': -2, 'border_patrol':-10}
        return sum(features.get(k, 0) * v for k, v in weights.items())

    def _get_features(self, game_state, action):
        features = util.Counter()
        successor = self._get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        

        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        if len(invaders) == 0:
            target = self._get_patrol_target(game_state)
            features['border_patrol'] = self.get_maze_distance(my_pos, target)

        return features

    def _get_patrol_target(self, game_state):
        """Boundary cell whose maze distance from us best matches the noisy
        distance to the nearest opponent. Uses the *current* state, so every
        candidate action is scored against the same target."""
        my_pos = game_state.get_agent_position(self.index)
        noisy = game_state.get_agent_distances()
        opp_dists = [noisy[i] for i in self.get_opponents(game_state) if noisy[i] is not None]
        if not opp_dists:
            return min(self.boundary, key=lambda b: self.get_maze_distance(my_pos, b))

        nearest = min(opp_dists)
        # Primary: closest match to the noisy reading. Tie-break: nearer to us.
        return min(self.boundary,
                   key=lambda b: (abs(self.get_maze_distance(my_pos, b) - nearest),
                                  self.get_maze_distance(my_pos, b)))