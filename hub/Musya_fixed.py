# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='FixedOffensive', second='FixedDefensive', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class BaseAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    
    def get_enemies(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [e for e in enemies if e.get_position() is not None]

    def get_invaders(self, game_state):
        width = game_state.data.layout.width
        halfway = width // 2
        invaders = []
        for e in self.get_enemies(game_state):
            pos = e.get_position()
            if pos is None:
                continue
            on_our_side = (pos[0] < halfway) if self.red else (pos[0] >= halfway)
            if on_our_side:
                invaders.append(e)
        return invaders

    def _h(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def best_action_to(self, actions, game_state, target):
        my_pos = game_state.get_agent_state(self.index).get_position()
        my_pos_int = (int(my_pos[0]), int(my_pos[1]))
        target_int = (int(target[0]), int(target[1]))

        if my_pos_int == target_int:
            return actions[0] if actions else Directions.STOP

        walls = game_state.get_walls()
        width, height = walls.width, walls.height

        open_set = [(self._h(my_pos_int, target_int), 0, my_pos_int, None)]
        came_from = {}
        g_score = {my_pos_int: 0}
        visited = set()

        while open_set:
            open_set.sort(key=lambda x: x[0])
            f, g, current, parent = open_set.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if parent is not None:
                came_from[current] = parent

            if current == target_int:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                if len(path) >= 2:
                    next_pos = path[1]
                    dx = next_pos[0] - my_pos_int[0]
                    dy = next_pos[1] - my_pos_int[1]
                    if dx > 0:
                        return Directions.EAST
                    if dx < 0:
                        return Directions.WEST
                    if dy > 0:
                        return Directions.NORTH
                    if dy < 0:
                        return Directions.SOUTH
                return actions[0] if actions else Directions.STOP

            x, y = current
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if walls[nx][ny]:
                    continue
                neighbor = (nx, ny)
                tentative_g = g + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._h(neighbor, target_int)
                    open_set.append((f_score, tentative_g, neighbor, current))

        return self._greedy_action(actions, game_state, target)

    def _greedy_action(self, actions, game_state, target):
        best_action = actions[0]
        best_dist = 9999
        for action in actions:
            succ = self.get_successor(game_state, action)
            if succ is None:
                continue
            pos = succ.get_agent_position(self.index)
            if pos is None:
                continue
            try:
                d = self.get_maze_distance(pos, target)
            except Exception:
                continue
            if d < best_dist:
                best_dist = d
                best_action = action
        return best_action

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

class FixedOffensive(BaseAgent):
    MAX_CARRY = 5
    DANGER_DIST = 4
    SAFETY_WEIGHT = 20
    SCARED_ATTACK_DIST = 2

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        carrying = getattr(my_state, 'num_carrying', 0)

        if carrying == 0:
            for e in self.get_enemies(game_state):
                if e.is_pacman:
                    continue
                if e.get_position() is None:
                    continue
                if getattr(e, 'scared_timer', 0) > 0:
                    d = self.get_maze_distance(my_pos, e.get_position())
                    if d <= self.SCARED_ATTACK_DIST:
                        return self.best_action_to(actions, game_state, e.get_position())

        if carrying > 0:
            enemy_dist = self._nearest_enemy_dist(game_state, my_pos)
            if carrying >= self.MAX_CARRY or enemy_dist < self.DANGER_DIST:
                return self.best_action_to(actions, game_state, self.start)

        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return self.best_action_to(actions, game_state, self.start)

        enemy_dist = self._nearest_enemy_dist(game_state, my_pos)
        if enemy_dist < self.DANGER_DIST and carrying == 0:
            return self.best_action_to(actions, game_state, self.start)

        enemies = self.get_enemies(game_state)
        target = self._choose_safe_food(game_state, my_pos, food_list, enemies)
        return self.best_action_to(actions, game_state, target)

    def _nearest_enemy_dist(self, game_state, my_pos):
        dist = 999
        for e in self.get_enemies(game_state):
            if e.is_pacman:
                continue
            if e.get_position() is None:
                continue
            if getattr(e, 'scared_timer', 0) > 0:
                continue 
            try:
                d = self.get_maze_distance(my_pos, e.get_position())
            except Exception:
                continue
            if d < dist:
                dist = d
        return dist

    def _choose_safe_food(self, game_state, my_pos, food_list, enemies):
        enemy_positions = [e.get_position() for e in enemies
                           if not e.is_pacman and e.get_position() is not None
                           and getattr(e, 'scared_timer', 0) == 0]

        def score(f):
            d_me = self.get_maze_distance(my_pos, f)
            penalty = 0
            for ep in enemy_positions:
                d_enemy = self.get_maze_distance(f, ep)
                if d_enemy < 5:
                    penalty += (5 - d_enemy) * self.SAFETY_WEIGHT
            return -d_me - penalty

        return max(food_list, key=score)


class FixedDefensive(BaseAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.guard_pos = None

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        halfway = width // 2
        walls = game_state.get_walls()
        border_x = halfway - 1 if self.red else halfway
        for y in [height // 2, height // 2 - 1, height // 2 + 1, height // 3, 2 * height // 3]:
            if 0 <= y < height and not walls[border_x][y]:
                self.guard_pos = (border_x, y)
                break
        if self.guard_pos is None:
            self.guard_pos = self.start

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        if getattr(my_state, 'scared_timer', 0) > 0:
            return self.best_action_to(actions, game_state, self.start)

        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            return self.best_action_to(actions, game_state, self.start)

        invaders = self.get_invaders(game_state)
        if invaders:
            target = min(invaders, key=lambda e: self.get_maze_distance(my_pos, e.get_position()))
            enemy_pos = target.get_position()
            intercept = self._intercept_point(game_state, enemy_pos)
            return self.best_action_to(actions, game_state, intercept)

        return self.best_action_to(actions, game_state, self.guard_pos)

    def _intercept_point(self, game_state, enemy_pos):
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        halfway = width // 2
        walls = game_state.get_walls()
        border_x = halfway - 1 if self.red else halfway
        best = self.guard_pos
        best_d = 9999
        enemy_y = int(enemy_pos[1])
        y_min = max(0, enemy_y - 4)
        y_max = min(height, enemy_y + 5)
        for y in range(y_min, y_max):
            if 0 <= border_x < width and not walls[border_x][y]:
                try:
                    d = self.get_maze_distance(enemy_pos, (border_x, y))
                except Exception:
                    continue
                if d < best_d:
                    best_d = d
                    best = (border_x, y)
        return best