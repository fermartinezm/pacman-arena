# a_star_pro.py  (Oreo)
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# =============================================================================
# A_STAR_PRO  -  a_star.py + capsule timing, remote food first, flexible roles
# =============================================================================
# Everything of a_star.py, plus:
#
# 1. CAPSULE AT THE RIGHT TIME. A capsule gives no points: it enters the PDDL
#    plan only when at least REMOTE_MIN (2) food is out of safe reach (guarded
#    by an active ghost), to unlock that food.
# 2. NO GHOST HUNTING. Eating a ghost gives no points: the hunt-ghost goal is
#    never used and, while the ghosts are scared, our Pacman avoids them
#    (EAT_LATE = 0) and gathers food it can bring home before the scare ends,
#    preferring the most remote food (power play).
# 3. REMOTE FOOD FIRST. The food targets of the planner are chosen by
#    safe_distance - REMOTE_W * depth (depth = distance to their border), so
#    the deepest, hardest food is taken first and the easy food near the
#    border is left for later.
# 4. DEFEND THE LEAD. Winning: balanced mode (one attacks, one defends and
#    only raids when nobody threatens us); winning by LOCK_LEAD (4) or near the
#    end: both defend, each one marking an enemy, so they cannot bring food home.
# 5. NO FIXED ROLES (FlexAgent). Every turn the team decides: an agent on the
#    enemy side carrying food stays the attacker; otherwise the agent closer to
#    the threat (nearest invader, or our border) defends and the other attacks.
#    Roles swap only when the other agent is SWAP_GAIN (3) cells closer.
#
# Results (headless, 1200 moves, both colours):
#   vs a_star:    RANDOM 1-6 8-0-4, RANDOM 20-27 8-2-6 (total 16-2-10, +0.9),
#                 defaultCapture 2-0-0
#   vs alphabeta: RANDOM 1-8 12-1-3 (+3.7)
# Tested and discarded: LOCK_LEAD = 2 (defend from a 2-point lead): 14-2-12
# vs a_star; REMOTE_W = 0.5: same as 1.0.
#
# a_star.py description:
# -----------------------------------------------------------------------------
# =============================================================================
# A_STAR  -  alphabeta.py + pddl_astar.py merged, with fixes found in real games
# =============================================================================
# Base: alphabeta.py (PDDL planner + Alpha-Beta tactical filter + decoy, lure,
# loop brain) plus from pddl_astar.py: POWER PLAY, MAX-GAIN BANKING, QUOTA 12,
# HORIZON 4, LOCK_LEAD 4. Fixes (each one seen losing games vs alphabeta):
#
# 1. TIMED DEAD-END SAFETY (pocket_ok, ghost_eta, timed_astar). Food inside a
#    dead end is a target only if we get in, eat it and reach the exit before
#    any ghost can (a scared ghost counts from when its scare ends). Before,
#    the attacker went into corridors with 5 food and was trapped there.
# 2. CLOSEST INVADER. The defender chased the first Pacman of the list, even
#    when the other invader was the one eating our food.
# 3. LIMITED REFLEX (REFLEX_MAX = 8). The attacker chased an invader next to it
#    with no limit, so alphabeta's decoy dragged it around a loop forever.
# 4. PINCER (pincer). When the teammate already chases the invader, the second
#    ghost takes a path that avoids the teammate's lane, so in a loop it comes
#    from the other side; before, both ghosts followed it on the same cell.
# 5. LOCK MARKING (HELP_RANGE = 3). In LOCK mode each defender keeps its own
#    mark; before, both went for the decoy and the other enemy scored.
# 6. HELP (HELP_CARRY = 2). An attacker idle at home helps (pincer) against an
#    invader carrying 2+ food; a decoy carrying nothing does not trigger it.
#
# Results (headless, 1200 moves, both colours) against alphabeta:
#   RANDOM seeds 1-11: 14-3-5 (+1.7) / 20-28: 8-2-8 (-0.1) / 30-38: 11-2-5 (+2.5)
#   defaultCapture: 0-0-2 (-1).  Total 33-7-20.
# Also: vs pddl_astar 10-1-5, vs mcts 17-2-5, vs lama 5-0-3, vs learner 6-0-2,
# vs strategist 2-0-6 (alphabeta does worse there: -4.6 vs -1.1 avg).
# Move time: max ~0.09 s.
#
# alphabeta.py description:
# -----------------------------------------------------------------------------
# =============================================================================
# ALPHABETA  -  decoy.py + Minimax with Alpha-Beta pruning as a tactical filter
# =============================================================================
# Course techniques: classical planning with PDDL (lecture 6) for strategy,
# A*/BFS search (lectures 2-3) for paths, and ADVERSARIAL SEARCH (lecture 4):
# depth-limited Minimax with Alpha-Beta pruning, iterative deepening, move
# ordering and a hand-crafted evaluation function, for the close fights.
# Full view of the map makes the game one of perfect information, which is
# what Minimax assumes.
#
# How the layers work together (see the TACTICAL LAYER section in the code):
#   1. The strategy of decoy.py chooses a move (PDDL planner, roles driven by
#      the score, decoy plan at the end, lure, loop brain).
#   2. If an enemy is in close contact (an active ghost within 5 cells of our
#      Pacman, or an invader within 5 cells of our ghost), Alpha-Beta computes
#      the Minimax value of every move (us = MAX, that enemy = MIN), deepening
#      2, 4, 6... plies within 0.04 s (it reaches depth 10+).
#   3. The search works as a FILTER: it vetoes the planned move if it leads to
#      being eaten and takes a move that surely eats the enemy; otherwise the
#      plan is kept. The evaluation rewards progress towards the plan's goal.
#
# SPEED: the search does not copy the real game state (slow); it runs on a
# LIGHT SIMULATION of the fight with the same rules (positions, food, food
# carried and banked, capsules, scared timers, who eats whom). Move time:
# mean ~13 ms, max ~65-110 ms (before: mean ~65-75 ms, max 0.3-0.8 s).
#
# Results: level with decoy (40 games: 14 wins, 11 ties, 15 losses);
# league of 252 games (slow version): 3rd of 7, 45.5 points vs decoy's 47. Letting the search REPLACE the plan, or
# take any move it judged better, was clearly worse (it only sees a few
# moves ahead, the plan sees the whole game).
# =============================================================================

# decoy.py description:
# -----------------------------------------------------------------------------
# DECOY  -  planner_pro.py + a decoy plan for the end of the game
# =============================================================================
# Everything planner_pro.py does, plus:
#
# D. DECOY PLAN. In the last DECOY_MOVES moves (per agent) of the game, if we
#    are NOT winning, the team splits the work:
#      - DECOY: the agent closest to the enemy defenders crosses next to them
#        and hovers about DECOY_DISTANCE cells away, so they chase it; when a
#        ghost gets within 2 cells it steps back to our side and comes again.
#        As it stays near the border it also guards: it goes after any
#        invader within DECOY_GUARD cells (so our side is not left empty).
#      - SCORER: the other agent plans (with the PDDL planner) to bring home
#        just the food needed to take the lead (needed = 1 - lead), entering
#        far from the decoy and the defenders, and runs home with it.
#    Roles are fixed the first turn of the plan so they do not swap.
#    DECOY_MOVES = 30: tested 30, 50, 75 and 100 and "only when losing"; the
#    longer the plan, the longer our side is left without defense and strong
#    teams score more than we gain. 30 was clearly best.
#
# G. LURE (only when both defend). If an enemy ghost waits at the border in
#    front of our two defenders for LURE_AFTER (6) turns - everybody frozen -
#    we invite it in: the defender in front of it steps back LURE_DEPTH (5)
#    cells, the other hides beside the crossing point out of its reach. When
#    it crosses (it is a Pacman now) the closer defender chases it and the
#    other blocks its way home. Lasts up to LURE_MOVES (25) moves.
#
# F. LOOP BRAIN (loop_brain): memory of the last 12 positions. If an agent
#    attacking on the enemy side repeats the same cycle of 2, 3 or 4 cells
#    three times in a row ("ok, I am in a loop"), for BREAK_TURNS (4) turns
#    it avoids the cells of that cycle with another legal move that does not
#    step next to an active ghost (random, so two mirroring agents stop
#    mirroring). Defenders are not affected: breaking their loops opened the
#    defense in tests. With it: 1st of 7 in a 504-game league (80 won, 35
#    tied, 29 lost) vs 75-40-29 without it.
#
# Typical result (league of 864 games, 24 per pair): 1st of 9 (121 won,
# 39 tied, 32 lost), ahead of planner_pro (105-52-35). Head to head against
# planner_pro: 7 wins, 4 losses. Beats planner 17/24, hunter 13/24,
# strategist_pro 13/24, fortress 21/24, baseline_team 24/24.
#
# planner_pro.py description:
# -----------------------------------------------------------------------------
# PLANNER PRO  -  planner.py + roles driven by the score + dead-end safety
# =============================================================================
# Same as planner.py (PDDL strategic planning + BFS execution + full view),
# plus two features:
#
# A. ROLES DRIVEN BY THE SCORE (team_mode, recomputed every turn):
#      - losing or tied  -> ATTACK: the defender raids more (closer threats
#                           allowed, bigger quota) to catch up.
#      - winning         -> BALANCED: planner.py as it is (attacker + defender
#                           that raids only when nobody threatens our side).
#      - winning by LOCK_LEAD or more, or winning with LOCK_MOVES moves left
#                        -> LOCK: both defend, each one marking one enemy
#                           (an agent carrying food brings it home first).
#
# B. DEAD-END SAFETY (DEAD_END_SAFETY): at start-up the maze is analysed and
#    every cell inside a dead end gets its exit and depth. A dead-end cell
#    on the enemy side is forbidden (for the planner and for the paths) when
#    an active ghost reaches its exit before we can go in and come back out:
#        ghost_to_exit <= our_distance_to_cell + depth + 1
#    OFF by default (DEAD_END_SAFETY = False): in a 672-game league the
#    version without it came 1st and the one with it 2nd (it gives up food in
#    corridors). Set it to True to experiment.
#
# Typical result (league of 672 games, 24 per pair, with anti-loop): 1st of 8
# with the fewest losses (89 won, 59 tied, 20 lost). Beats planner 16/24,
# strategist_pro 13/24, hunter 10/24 (loses 3), fortress 18/24 (no losses),
# baseline_team 24/24; the version without anti-loop wins only 2/24 against it.
#
# C. ANTI-LOOP (check_loop): if the attacker spends LOOP_WINDOW (10) turns on
#    at most LOOP_CELLS (4) cells without eating anything - typically dancing
#    in front of a ghost that blocks the way - it changes tactics: RISK (for 6
#    turns it may pass right next to the ghost; only while carrying <= 2 food;
#    more likely each time it gets stuck: 60%, 80%, 90%) or DETOUR (retreat
#    and enter through the border crossing farthest from the ghosts).
#    Measured: time stuck drops from ~27% to ~19% of the attacker's turns.
#
# Original planner.py description:
# -----------------------------------------------------------------------------
# Two layers, as in a classical autonomous system (deliberative + reactive):
#
# 1. STRATEGIC LAYER (planning). Every turn the attacker turns the game state
#    into a classical planning problem, written in PDDL (domain PACMAN_DOMAIN
#    below, problem built by AttackPlanner.build_problem), and solves it with
#    an embedded A* planner over STRIPS states with action costs. No external
#    planner is needed (the contest cluster may not have one).
#      - Objects: targets on the map (a few nearest food, capsules, home,
#        scared ghosts) and counters n0..n6 for the food carried.
#      - Actions: eat-food, eat-capsule, hunt-ghost, return-home, and their
#        "powered" versions (usable after a capsule: enemies are scared, so
#        paths next to ghosts become allowed).
#      - Costs: real maze distances; paths next to an active ghost are
#        forbidden unless powered.
#      - Goal (chosen by the situation):
#          * a scared ghost is reachable in time  -> (ghost-eaten)
#          * otherwise                            -> (banked): bring home
#            QUOTA food. QUOTA = 6 in total, at most HORIZON (3) more than
#            what we carry, never more than can be reached; it drops to what
#            we carry right now when a ghost is close, time is running out
#            or only 2 food are left.
#    The optimal (cheapest) plan is a sequence like
#      eat-food f3 -> eat-food f1 -> eat-food f4 -> return-home   (normal)
#      eat-capsule c0 -> return-home-powered                      (home is blocked)
#    Only its first action is executed; the agent replans every turn, so the
#    plan adapts as ghosts move (receding-horizon planning). The target being
#    executed gets a small discount (COMMITMENT) so the agent does not dither
#    between two plans of almost the same cost.
#
# 2. TACTICAL LAYER (execution). The first action of the plan gives a target
#    cell; a BFS path that avoids cells next to active ghosts takes us there.
#    With no plan (every food guarded) it walks along our border to the
#    crossing point farthest from the enemy ghosts and tries again.
#
# Team roles:
#   - PlannerAttacker: the planning agent above.
#   - PlannerDefender: guards one step behind the border (no face-off with
#     enemy ghosts), chases invaders (keeps 2 cells away while scared) and,
#     when no enemy is near our border, RAIDS: it uses the same planner with a
#     small quota (2 food) and comes back as soon as an enemy approaches.
#
# Full view: observation_function returns the unfiltered state, so every
# opponent's position is known at every turn.
#
# Typical result: wins every game against baseline_team (avg about +18);
# about even against hunter.py.
#
# Debug: set the environment variable PACMAN_PDDL_DIR to a folder and every
# planning problem is written there as PDDL (with the domain), to inspect it
# or to solve it with an external planner (e.g. Fast Downward).
# =============================================================================

import heapq
import math
import itertools
import os
import random
import time
import traceback
from collections import deque

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='FlexAgent', second='FlexAgent', num_training=0):
    agents = {cls.__name__: cls for cls in (PlannerAttacker, PlannerDefender, FlexAgent)}
    return [agents[first](first_index), agents[second](second_index)]


STEPS = {(0, 1): Directions.NORTH, (0, -1): Directions.SOUTH,
         (1, 0): Directions.EAST, (-1, 0): Directions.WEST}
INF = float('inf')
MOVE_TIME_BUDGET = 0.8   # seconds; the contest warns at 1 s per move


# =============================================================================
# PDDL domain (documentation and debug export; the embedded planner below
# implements exactly these actions on grounded STRIPS states).
# =============================================================================

PACMAN_DOMAIN = """\
(define (domain pacman-strategy)
  (:requirements :strips :typing :action-costs)
  (:types target count)
  (:predicates
    (at ?t - target)                 ; where the attacker is (abstractly)
    (food ?t - target) (capsule ?t - target) (scared-ghost ?t - target)
    (home ?t - target)
    (safe ?a ?b - target)            ; path a->b avoids active ghosts
    (carrying ?n - count) (next ?n ?m - count) (enough ?n - count)
    (powered)                        ; a capsule was eaten: enemies are scared
    (banked) (ghost-eaten))
  (:functions (dist ?a ?b - target) (total-cost))

  (:action eat-food
    :parameters (?from ?to - target ?n ?m - count)
    :precondition (and (at ?from) (food ?to) (safe ?from ?to) (carrying ?n) (next ?n ?m))
    :effect (and (not (at ?from)) (at ?to) (not (food ?to))
                 (not (carrying ?n)) (carrying ?m) (increase (total-cost) (dist ?from ?to))))
  (:action eat-food-powered
    :parameters (?from ?to - target ?n ?m - count)
    :precondition (and (powered) (at ?from) (food ?to) (carrying ?n) (next ?n ?m))
    :effect (and (not (at ?from)) (at ?to) (not (food ?to))
                 (not (carrying ?n)) (carrying ?m) (increase (total-cost) (dist ?from ?to))))
  (:action eat-capsule
    :parameters (?from ?to - target)
    :precondition (and (at ?from) (capsule ?to) (safe ?from ?to))
    :effect (and (not (at ?from)) (at ?to) (not (capsule ?to)) (powered)
                 (increase (total-cost) (dist ?from ?to))))
  (:action hunt-ghost
    :parameters (?from ?to - target)
    :precondition (and (at ?from) (scared-ghost ?to))
    :effect (and (not (at ?from)) (at ?to) (not (scared-ghost ?to)) (ghost-eaten)
                 (increase (total-cost) (dist ?from ?to))))
  (:action return-home
    :parameters (?from ?h - target ?n - count)
    :precondition (and (at ?from) (home ?h) (safe ?from ?h) (carrying ?n) (enough ?n))
    :effect (and (not (at ?from)) (at ?h) (banked) (increase (total-cost) (dist ?from ?h))))
  (:action return-home-powered
    :parameters (?from ?h - target ?n - count)
    :precondition (and (powered) (at ?from) (home ?h) (carrying ?n) (enough ?n))
    :effect (and (not (at ?from)) (at ?h) (banked) (increase (total-cost) (dist ?from ?h)))))
"""


# =============================================================================
# Embedded classical planner: A* over grounded STRIPS with action costs
# =============================================================================

class Action:
    __slots__ = ('name', 'args', 'pre', 'add', 'delete', 'cost')

    def __init__(self, name, args, pre, add, delete, cost):
        self.name, self.args = name, args
        self.pre, self.add, self.delete = frozenset(pre), frozenset(add), frozenset(delete)
        self.cost = cost

    def __repr__(self):
        return f'({self.name} {" ".join(map(str, self.args))})'


def plan_astar(init, goal, actions, heuristic=lambda s: 0, max_expansions=20000, deadline=None):
    """Cheapest action sequence from `init` to a state containing all `goal` facts, or None."""
    init, goal = frozenset(init), frozenset(goal)
    counter = itertools.count()
    frontier = [(heuristic(init), next(counter), 0, init, ())]
    best = {init: 0}
    expansions = 0
    while frontier:
        _, _, g, state, plan = heapq.heappop(frontier)
        if goal <= state:
            return list(plan), g
        if g > best.get(state, INF):
            continue
        expansions += 1
        if expansions > max_expansions or (deadline and time.perf_counter() > deadline):
            return None
        for a in actions:
            if a.pre <= state:
                nxt = (state - a.delete) | a.add
                ng = g + a.cost
                if ng < best.get(nxt, INF):
                    best[nxt] = ng
                    heapq.heappush(frontier, (ng + heuristic(nxt), next(counter), ng, nxt, plan + (a,)))
    return None


# =============================================================================
# Shared helpers (geometry, BFS, full view, safety)
# =============================================================================

class PlannerAgentBase(CaptureAgent):
    DEAD_END_SAFETY = False
    LOCK_LEAD = 4      # winning by this much: both agents defend
    LOCK_MOVES = 60    # winning with this few moves left (per agent): both defend
    GUARD_DEPTH = 1    # defenders wait one step behind the border (no face-off)
    BREAK_TURNS = 4    # loop brain: turns spent avoiding the cells of a detected loop
    lure_state = {}    # shared: stalemate at the border and the lure in progress
    LURE_AFTER = 6     # turns of an enemy ghost waiting at the border before luring it in
    LURE_MOVES = 25    # how long (per agent) the lure lasts
    LURE_DEPTH = 5     # how far back the defender in front steps
    DECOY_MOVES = 30   # decoy plan in the last moves of the game (per agent) when not winning
    DECOY_DISTANCE = 3 # the decoy hovers this far from the defender it distracts
    DECOY_GUARD = 8    # the decoy goes after invaders this close to it
    decoy_roles = {}   # shared by the two agents of the team: team -> (timeleft, decoy index)
    # Anti-loop: stuck = last LOOP_WINDOW attacking turns on <= LOOP_CELLS cells without eating.
    LOOP_WINDOW = 10
    LOOP_CELLS = 4
    RISK_TURNS = 6      # turns of risk mode (may pass next to a ghost)
    DETOUR_TURNS = 10   # turns of detour (retreat and enter somewhere else)
    RISK_MAX_CARRY = 2  # take risks only while carrying at most this much food

    def observation_function(self, game_state):
        # Full world view: skip make_observation, which would hide distant opponents.
        return game_state

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()
        self.mid = self.walls.width // 2
        border_x = self.mid - 1 if self.red else self.mid
        self.home_border = [(border_x, y) for y in range(self.walls.height) if not self.walls[border_x][y]]
        self.turn_start = time.perf_counter()
        self.team = sorted(self.get_team(game_state))
        self.analyse_dead_ends()
        self.history = deque(maxlen=self.LOOP_WINDOW)  # (cell, food left) of recent attacking turns
        self.stuck_count = 0
        self.risk_turns = 0
        self.detour_turns = 0
        self.detour_target = None
        own = [(x, y) for x in range(self.walls.width) for y in range(self.walls.height)
               if not self.walls[x][y] and self.on_my_side((x, y))]
        depth = {c: self.distance_home(c) for c in own}
        best = min(abs(d - self.GUARD_DEPTH) for d in depth.values())
        self.guard_posts = [c for c, d in depth.items() if abs(d - self.GUARD_DEPTH) == best]
        self.own_depth = depth
        self.home_set = set(self.home_border)
        self.enemy_cells = [(x, y) for x in range(self.walls.width) for y in range(self.walls.height)
                            if not self.walls[x][y] and not self.on_my_side((x, y))]
        self.home_dist = {c: self.distance_home(c) for c in self.enemy_cells + own}

    # --- team roles -------------------------------------------------------

    def team_mode(self, game_state):
        """attack (losing or tied) / balanced (winning) / lock (winning clearly or near the end)."""
        lead = self.get_score(game_state)
        if lead <= 0 and game_state.data.timeleft // 4 <= self.DECOY_MOVES:
            return 'decoy'
        if lead <= 0:
            return 'attack'
        if lead >= self.LOCK_LEAD or game_state.data.timeleft // 4 <= self.LOCK_MOVES:
            return 'lock'
        return 'balanced'

    # --- decoy plan -------------------------------------------------------

    def decoy_index(self, game_state):
        """Which agent is the decoy: the one closest to the enemy defenders, fixed for the plan."""
        key = tuple(self.team)
        timeleft = game_state.data.timeleft
        stored = self.decoy_roles.get(key)
        if stored and stored[0] >= timeleft:  # same game, plan already started
            return stored[1]
        defenders = [p for p, e in self.enemy_states(game_state) if not e.is_pacman]

        def closeness(i):
            st = game_state.get_agent_state(i)
            pos = self.cell(st.get_position())
            # an agent already carrying food should be the scorer, not the decoy
            return (st.num_carrying, min((self.dist(pos, d) for d in defenders), default=0), i)
        decoy = min(self.team, key=closeness)
        self.decoy_roles[key] = (timeleft, decoy)
        return decoy

    def play_decoy_plan(self, game_state):
        if self.decoy_index(game_state) == self.index:
            return self.act_as_decoy(game_state)
        lead = self.get_score(game_state)
        needed = max(1, 1 - lead)
        return self.attack(game_state, needed)

    def act_as_decoy(self, game_state):
        """Show up next to the enemy defenders so they chase us, without getting eaten."""
        self.decoy_now = True
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        enemies = self.enemy_states(game_state)
        active = [p for p, e in enemies if not e.is_pacman and e.scared_timer <= 2]
        if not active:  # nothing to distract: just score
            return self.attack(game_state, max(1, 1 - self.get_score(game_state)))
        target = min(active, key=lambda g: self.dist(my_pos, g))
        ghost_dist = min(self.dist(my_pos, g) for g in active)

        # The decoy also guards: it stays near our border, so it eats any invader close by.
        invaders = [p for p, e in enemies if e.is_pacman]
        if invaders and me.scared_timer == 0:
            close = [p for p in invaders if self.dist(my_pos, p) <= self.DECOY_GUARD]
            if close:
                if me.is_pacman:
                    action, _ = self.bfs(my_pos, self.home_border, self.danger_zone(active, my_pos))
                else:
                    action, _ = self.bfs(my_pos, close, allowed=self.on_my_side)
                if self.legal(game_state, action) and action != Directions.STOP:
                    return action

        if me.is_pacman and ghost_dist <= 2:  # too close: step back home
            action, _ = self.bfs(my_pos, self.home_border, self.danger_zone(active, my_pos))
            if self.legal(game_state, action) and action != Directions.STOP:
                return action
            return self.flee(game_state, active)

        if not me.is_pacman and self.dist(my_pos, target) > self.DECOY_DISTANCE + 1:
            # Go to our border in front of the defender.
            crossing = min(self.home_border, key=lambda b: self.dist(target, b))
            if my_pos != crossing:
                action, _ = self.bfs(my_pos, [crossing], allowed=self.on_my_side)
                if self.legal(game_state, action) and action != Directions.STOP:
                    return action

        # Hover about DECOY_DISTANCE cells from the defender, never next to a ghost,
        # keeping away from our scorer so the defender is pulled away from it.
        scorer = [self.cell(game_state.get_agent_state(i).get_position()) for i in self.team if i != self.index]
        danger = {c for g in active for c in [g] + self.neighbours(g)}

        def score(action):
            pos = self.cell(game_state.generate_successor(self.index, action).get_agent_position(self.index))
            if pos in danger:
                return INF
            away = min((self.dist(pos, p) for p in scorer), default=0)
            return abs(self.dist(pos, target) - self.DECOY_DISTANCE) - 0.05 * away + (0.5 if action == Directions.STOP else 0)
        return min(game_state.get_legal_actions(self.index), key=score)

    def defend(self, game_state, lock=False):
        """Guard one step behind the border and chase invaders; in LOCK mode mark one enemy each."""
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        enemies = self.enemy_states(game_state)
        if not enemies:
            return Directions.STOP
        ghosts = [p for p, e in enemies if not e.is_pacman]
        if me.is_pacman:  # still on the enemy side: come back
            active = [p for p, e in enemies if not e.is_pacman and e.scared_timer <= 2]
            action, _ = self.bfs(my_pos, self.home_border, self.danger_zone(active, my_pos))
            if self.legal(game_state, action) and action != Directions.STOP:
                return action
            return self.flee(game_state, active)

        invaders = [p for p, e in enemies if e.is_pacman]
        if lock and me.scared_timer == 0:
            action = self.lure(game_state, my_pos, enemies)
            if action is not None:
                self.guarding = True  # the lure is a team plan: no tactical override
                return action
        mark_pos, mark = self.marked_enemy(game_state, enemies, lock)
        mate = self.mate_chaser(game_state)
        if mark.is_pacman:
            targets = [mark_pos]
        elif lock and mate is not None:
            # LOCK: my mark is still outside; the invader belongs to my teammate,
            # unless it comes close to me (no double-teaming one decoy).
            targets = [p for p in invaders if self.dist(my_pos, p) <= self.HELP_RANGE]
        else:
            targets = invaders
        if targets:
            if me.scared_timer > 0:
                return self.keep_distance(game_state, targets, 2)
            action = self.pincer(game_state, my_pos, targets, mate)
            if action is None:
                action, _ = self.bfs(my_pos, targets, allowed=self.on_my_side)
            if self.legal(game_state, action):
                return action

        self.guarding = True  # standing guard: the loop brain leaves this alone
        near = {c for g in ghosts for c in [g] + self.neighbours(g)}
        if my_pos in near:
            return self.keep_distance(game_state, ghosts, self.GUARD_DEPTH + 1)
        crossing = min(self.home_border, key=lambda b: self.dist(mark_pos, b))
        post = min(self.guard_posts, key=lambda c: (self.dist(crossing, c), self.dist(my_pos, c)))
        action, _ = self.bfs(my_pos, [post], blocked=near, allowed=self.on_my_side)
        if self.legal(game_state, action):
            return action
        return Directions.STOP

    HELP_RANGE = 3   # a defender leaves its own mark only for an invader this close

    def mate_chaser(self, game_state):
        """(position) of the teammate if it is a ghost able to chase (at home, not scared), else None."""
        for i in self.team:
            if i != self.index:
                st = game_state.get_agent_state(i)
                if not st.is_pacman and st.scared_timer == 0:
                    return self.cell(st.get_position())
        return None

    def pincer(self, game_state, my_pos, targets, mate):
        """
        PINCER: when the teammate is already chasing the invader (it is at least as
        close), do not follow it along the same corridor: take a path that avoids
        the teammate's way to the invader, so in a loop we come from the other
        side and trap it; if there is no such path, block its way home (the
        crossing of our border closest to it). None = just chase.
        """
        if mate is None:
            return None
        inv = min(targets, key=lambda p: self.dist(my_pos, p))
        d_me, d_mate = self.dist(my_pos, inv), self.dist(mate, inv)
        if d_mate > d_me or (d_mate == d_me and self.index == self.team[0]) or d_me == 1 and d_mate > 1:
            return None
        lane = {c for c in self.own_depth if c != inv
                and self.dist(mate, c) + self.dist(c, inv) == d_mate}
        lane.discard(my_pos)
        action, length = self.bfs(my_pos, [inv], blocked=lane, allowed=self.on_my_side)
        if action is not None and length <= d_me + 12:
            self.pincers = getattr(self, 'pincers', 0) + 1
            return action
        exit_cell = min(self.home_border, key=lambda b: self.dist(inv, b))
        if self.dist(my_pos, exit_cell) > self.dist(inv, exit_cell):
            return None
        if my_pos == exit_cell:
            return Directions.STOP
        action, _ = self.bfs(my_pos, [exit_cell], blocked=lane, allowed=self.on_my_side)
        return action

    def lure(self, game_state, my_pos, enemies):
        """
        LURE (both defending). If an enemy ghost waits at the border in front of our
        two defenders for LURE_AFTER turns (nobody moves: stalemate), we invite it
        in: the defender in front of it steps back LURE_DEPTH cells, the other one
        hides beside the crossing point, out of reach. When it crosses (it becomes
        a Pacman), the closer defender chases it and the other blocks its way home.
        Returns an action, or None when no lure is going on.
        """
        key = tuple(self.team)
        timeleft = game_state.data.timeleft
        st = self.lure_state.get(key)
        if st is None or timeleft > st['seen']:
            st = self.lure_state[key] = dict(seen=timeleft, count=0, target=None, until=0, back=None, crossing=None)
        first_call = timeleft < st['seen'] or st.get('counted') != timeleft
        st['seen'] = timeleft
        positions = {i: self.cell(game_state.get_agent_state(i).get_position()) for i in self.team}
        enemy_at = {i: (self.cell(game_state.get_agent_state(i).get_position()), game_state.get_agent_state(i))
                    for i in self.get_opponents(game_state) if game_state.get_agent_state(i).get_position()}

        # Active lure: trap the invader, or keep the bait open.
        if st['target'] is not None:
            pos, enemy = enemy_at.get(st['target'], (None, None))
            if pos is None or timeleft <= st['until'] or (not enemy.is_pacman and self.distance_home(pos) > 4):
                st.update(target=None, count=0)  # over: it left, it was eaten, or time is up
                return None
            if enemy.is_pacman:
                chaser = min(self.team, key=lambda i: (self.dist(positions[i], pos), i))
                if self.index == chaser:
                    goal = pos
                else:
                    goal = min(self.home_border, key=lambda b: self.dist(pos, b))
                    if self.dist(my_pos, pos) <= 2:
                        goal = pos
                if my_pos == goal:
                    return Directions.STOP
                action, _ = self.bfs(my_pos, [goal], allowed=self.on_my_side)
                return action if self.legal(game_state, action) else None
            crossing = st['crossing']
            if self.index == st['back']:
                spot = min((c for c, d in self.own_depth.items() if d == self.LURE_DEPTH),
                           key=lambda c: self.dist(c, crossing), default=None)
            else:
                spot = min((c for c, d in self.own_depth.items() if 2 <= d <= 3 and self.dist(c, pos) >= 4),
                           key=lambda c: (abs(self.dist(c, crossing) - 3), self.dist(my_pos, c)), default=None)
            if spot is None:
                return None
            if my_pos == spot:
                return Directions.STOP
            near = {c for c in [pos] + self.neighbours(pos)}
            action, _ = self.bfs(my_pos, [spot], blocked=near, allowed=self.on_my_side)
            return action if self.legal(game_state, action) else None

        # Detect the stalemate (counted once per turn of the team's first agent).
        if any(e.is_pacman for _, e in enemy_at.values()):
            st['count'] = 0
            return None
        waiting = [(i, p) for i, (p, e) in enemy_at.items()
                   if self.distance_home(p) <= 2 and all(self.dist(positions[a], p) <= 5 for a in self.team)]
        if self.index == self.team[0] and st.get('counted') != timeleft:
            st['counted'] = timeleft
            st['count'] = st['count'] + 1 if waiting else 0
        if waiting and st['count'] >= self.LURE_AFTER:
            target, pos = waiting[0]
            back = min(self.team, key=lambda i: (self.dist(positions[i], pos), i))
            st.update(target=target, until=timeleft - 4 * self.LURE_MOVES, back=back,
                      crossing=min(self.home_border, key=lambda b: self.dist(pos, b)), count=0)
            self.lures = getattr(self, 'lures', 0) + 1
            return self.lure(game_state, my_pos, enemies)
        return None

    def marked_enemy(self, game_state, enemies, lock):
        """The enemy to watch: the closest threat, or (LOCK) one enemy per agent."""
        if not lock or len(enemies) < 2:
            me_pos = self.cell(game_state.get_agent_state(self.index).get_position())
            return min(enemies, key=lambda pe: (0, self.dist(me_pos, pe[0])) if pe[1].is_pacman
                       else (1, self.distance_home(pe[0])))
        pos = {i: self.cell(game_state.get_agent_state(i).get_position()) for i in self.team}

        def cost(pairing):
            return sum(self.dist(pos[a], p) + (0 if e.is_pacman else 2 * self.distance_home(p))
                       for a, (p, e) in zip(self.team, pairing))
        pairing = min([enemies, enemies[::-1]], key=cost)
        return pairing[self.team.index(self.index)]

    def _base_get_action(self, game_state):
        # An exception would forfeit the game: log it and play a legal move instead.
        self.turn_start = time.perf_counter()
        self.guarding = False
        self.decoy_now = False
        self.plan_target = None
        try:
            action = super().get_action(game_state)
            action = self.loop_brain(game_state, action)
            return self.tactical(game_state, action)
        except Exception:
            traceback.print_exc()
            actions = game_state.get_legal_actions(self.index)
            return random.choice([a for a in actions if a != Directions.STOP] or actions)

    # --- POWER PLAY (after eating a capsule) -----------------------------------------
    EAT_LATE = 0        # eat a scared ghost only when its scare has this many moves left (or fewer)
    POWER_MARGIN = 2    # moves of margin to be home before the scare ends

    def get_action(self, game_state):
        action = self._base_get_action(game_state)
        try:
            return self.power_filter(game_state, action)
        except Exception:
            traceback.print_exc()
            return action

    def power_filter(self, game_state, action):
        """
        POWER PLAY. While every enemy ghost is scared (we ate a capsule), eating
        ghosts gives no points, so our Pacman does not hunt them:
          - it gathers as much food as possible, preferring the most remote food
            (deep in their side, normally guarded), but only food it can reach
            AND bring home before the scare ends;
          - it eats a scared ghost only near the end of the scare (EAT_LATE moves
            left, within 2 cells): the ghost then respawns far away just when it
            would have become dangerous again. Before that it avoids stepping
            on scared ghosts;
          - when the scare left only covers the way home, it goes home to bank.
        """
        me = game_state.get_agent_state(self.index)
        if not me.is_pacman:
            return action
        ghosts = [(p, e) for p, e in self.enemy_states(game_state) if not e.is_pacman]
        if not ghosts or any(e.scared_timer <= 0 for _, e in ghosts):
            return action                                   # not every ghost is scared
        scare = min(e.scared_timer for _, e in ghosts)
        my_pos = self.cell(me.get_position())
        home = self.distance_home(my_pos)

        # Late in the scare: eat a scared ghost that is right there.
        if scare <= self.EAT_LATE:
            close = [p for p, _ in ghosts if self.dist(my_pos, p) <= 2]
            if close:
                step = self.power_step(game_state, my_pos, set(close), set())
                if step:
                    return step
        avoid = set() if scare <= self.EAT_LATE else {p for p, _ in ghosts}

        # Time is up: bank what we carry.
        if scare <= home + self.POWER_MARGIN:
            if me.num_carrying > 0:
                return self.power_step(game_state, my_pos, set(self.home_border), avoid) or action
            return action

        # Gather: food we can reach and bring home in time, preferring remote food.
        food = self.get_food(game_state).as_list()
        feasible = [f for f in food
                    if self.dist(my_pos, f) + self.distance_home(f) + self.POWER_MARGIN <= scare]
        if not feasible:
            if me.num_carrying > 0:
                return self.power_step(game_state, my_pos, set(self.home_border), avoid) or action
            return action
        target = min(feasible, key=lambda f: self.dist(my_pos, f) - 0.5 * self.distance_home(f))
        return self.power_step(game_state, my_pos, {target}, avoid) or action

    def power_step(self, game_state, start, targets, blocked):
        """Legal first move of a shortest path from start to any target avoiding `blocked`, or None."""
        if not targets:
            return None


    # --- TIME-AWARE A* (escape safety) ------------------------------------------
    ESCAPE_MARGIN = 1   # we must reach every cell this many moves before any ghost can

    def ghost_eta(self, game_state):
        """Move at which an enemy ghost can be on each enemy-side cell (a scared one counts from when its scare ends)."""
        ghosts = [(self.cell(e.get_position()), e.scared_timer) for e in
                  (game_state.get_agent_state(i) for i in self.get_opponents(game_state))
                  if e.get_position() is not None and not e.is_pacman]
        if not ghosts:
            return {}
        return {c: min(max(self.dist(g, c), s) for g, s in ghosts) for c in self.enemy_cells}

    def pocket_ok(self, my_pos, cell, eta):
        """False if `cell` is in a dead end whose exit a ghost reaches before we get in and back out."""
        exit_cell = self.dead_exit.get(cell)
        if exit_cell is None or self.on_my_side(cell):
            return True
        out = self.dist(my_pos, cell) + self.dead_depth[cell]
        return out + self.ESCAPE_MARGIN < eta.get(exit_cell, INF)

    def cell_ok(self, c, t, eta):
        return self.on_my_side(c) or t + self.ESCAPE_MARGIN < eta.get(c, INF)

    def timed_reach(self, start, eta, t0=0):
        """Earliest arrival time at every cell reachable without a ghost getting there first."""
        times = {start: t0}
        frontier = deque([start])
        while frontier:
            c = frontier.popleft()
            t = times[c] + 1
            for n in self.neighbours(c):
                if n not in times and self.cell_ok(n, t, eta):
                    times[n] = t
                    frontier.append(n)
        return times

    def timed_astar(self, start, t0, eta, goals, h):
        """
        A* from `start` (reached at move t0) to any cell in `goals`, never entering a
        cell a ghost can reach first. h(cell) is an admissible estimate (maze distance).
        Returns the path length, or None if no safe path exists.
        """
        if not self.cell_ok(start, t0, eta):
            return None
        frontier = [(h(start), 0, start)]
        best = {start: 0}
        while frontier:
            f, g, c = heapq.heappop(frontier)
            if c in goals:
                return g
            if g > best.get(c, INF):
                continue
            for n in self.neighbours(c):
                ng = g + 1
                if ng < best.get(n, INF) and self.cell_ok(n, t0 + ng, eta):
                    best[n] = ng
                    heapq.heappush(frontier, (ng + h(n), ng, n))
        return None

    def can_escape(self, cell, t0, eta, capsules=()):
        """After being on `cell` at move t0, can we still get home (or to a capsule) before any ghost?"""
        if not eta or self.on_my_side(cell):
            return True
        goals = self.home_set | set(capsules)
        if capsules:
            h = lambda c: min([self.home_dist[c]] + [self.dist(c, k) for k in capsules])
        else:
            h = self.home_dist.__getitem__
        return self.timed_astar(cell, t0, eta, goals, h) is not None

    # --- loop brain -------------------------------------------------------

    def loop_brain(self, game_state, action):
        """
        Memory of the last positions. If the agent repeats the same cycle of 2, 3
        or 4 cells three times in a row ("ok, I am in a loop"), for BREAK_TURNS
        turns it avoids the cells of that cycle: it takes another legal move that
        does not step next to an active enemy ghost, chosen at random so that two
        agents mirroring each other stop mirroring. Only for agents attacking on
        the enemy side: a defender holding or chasing must not open the way.
        """
        if not hasattr(self, 'memory'):
            self.memory, self.break_turns, self.banned = deque(maxlen=12), 0, set()
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        self.memory.append(my_pos)
        enemies = self.enemy_states(game_state)
        invaders = [p for p, e in enemies if e.is_pacman]
        if not me.is_pacman and self.break_turns == 0:
            return action  # only attacking agents (on the enemy side) break loops

        mem = list(self.memory)
        if self.break_turns == 0:
            for p in (2, 3, 4):
                if len(mem) >= 3 * p and mem[-p:] == mem[-2 * p:-p] == mem[-3 * p:-2 * p] and len(set(mem[-p:])) > 1:
                    self.break_turns, self.banned = self.BREAK_TURNS, set(mem[-p:])
                    self.loops_broken = getattr(self, 'loops_broken', 0) + 1
                    break
        if self.break_turns == 0:
            return action
        self.break_turns -= 1

        active = [p for p, e in enemies if not e.is_pacman and e.scared_timer <= 2]
        danger = {c for g in active for c in [g] + self.neighbours(g) if not self.on_my_side(c)}
        invader_danger = set(invaders) if me.scared_timer > 0 else set()
        options = []
        for a in game_state.get_legal_actions(self.index):
            if a == Directions.STOP:
                continue
            nxt = self.cell(game_state.generate_successor(self.index, a).get_agent_position(self.index))
            if nxt in danger or nxt in invader_danger:
                continue
            options.append((nxt in self.banned, a))
        fresh = [a for banned, a in options if not banned]
        if fresh:
            return random.choice(fresh)
        return action

    def deadline(self):
        return self.turn_start + MOVE_TIME_BUDGET * 0.6

    # --- geometry -------------------------------------------------------

    def on_my_side(self, cell):
        return cell[0] < self.mid if self.red else cell[0] >= self.mid

    def neighbours(self, cell):
        x, y = cell
        return [(x + dx, y + dy) for dx, dy in STEPS if not self.walls[x + dx][y + dy]]

    def dist(self, a, b):
        return self.get_maze_distance(a, b)

    def distance_home(self, pos):
        return min(self.dist(pos, b) for b in self.home_border)

    @staticmethod
    def cell(pos):
        return tuple(int(c) for c in nearest_point(pos))

    def enemy_states(self, game_state):
        result = []
        for i in self.get_opponents(game_state):
            st = game_state.get_agent_state(i)
            if st.get_position() is not None:
                result.append((self.cell(st.get_position()), st))
        return result

    def analyse_dead_ends(self):
        """Dead ends: cells removed by repeatedly pruning cells with one open neighbour."""
        cells = [(x, y) for x in range(self.walls.width) for y in range(self.walls.height) if not self.walls[x][y]]
        degree = {c: len(self.neighbours(c)) for c in cells}
        removed = set()
        leaves = deque(c for c in cells if degree[c] <= 1)
        while leaves:
            c = leaves.popleft()
            if c in removed:
                continue
            removed.add(c)
            for n in self.neighbours(c):
                if n not in removed:
                    degree[n] -= 1
                    if degree[n] == 1:
                        leaves.append(n)
        self.dead_exit, self.dead_depth = {}, {}
        for root in cells:
            if root in removed:
                continue
            frontier = deque((n, 1) for n in self.neighbours(root) if n in removed)
            while frontier:
                c, d = frontier.popleft()
                if c in self.dead_exit:
                    continue
                self.dead_exit[c], self.dead_depth[c] = root, d
                frontier.extend((n, d + 1) for n in self.neighbours(c) if n in removed and n not in self.dead_exit)

    def danger_zone(self, ghosts, my_pos=None):
        """
        Cells on the enemy side where an active ghost can catch us next turn, plus
        (DEAD_END_SAFETY) dead-end cells whose exit a ghost reaches before we get out.
        """
        if self.risk_turns > 0:
            # Risk mode: only the ghost's own cell is forbidden; passing next to it is allowed.
            return {g for g in ghosts if not self.on_my_side(g)}
        zone = {c for g in ghosts for c in [g] + self.neighbours(g) if not self.on_my_side(c)}
        if not self.DEAD_END_SAFETY or my_pos is None or not ghosts:
            return zone
        my_exit = self.dead_exit.get(my_pos)
        ghost_to = {}
        for c, exit_cell in self.dead_exit.items():
            if self.on_my_side(c) or exit_cell == my_exit:
                continue
            if exit_cell not in ghost_to:
                ghost_to[exit_cell] = min(self.dist(g, exit_cell) for g in ghosts)
            if ghost_to[exit_cell] <= self.dist(my_pos, c) + self.dead_depth[c] + 1:
                zone.add(c)
        return zone

    def check_loop(self, game_state, my_pos, active):
        """
        Anti-loop. Called on every attacking turn. If the agent has spent the last
        LOOP_WINDOW turns on at most LOOP_CELLS cells without eating (typically
        dancing in front of a ghost that blocks the way), it changes tactics:
          - RISK: for RISK_TURNS turns it may pass right next to the ghost
            (only while carrying little food). The more often it got stuck,
            the more likely it takes the risk.
          - DETOUR: otherwise it retreats and, for DETOUR_TURNS turns, goes to
            the border crossing farthest from the ghosts to enter elsewhere.
        """
        food_left = len(self.get_food(game_state).as_list())
        self.history.append((my_pos, food_left))
        if self.risk_turns > 0:
            self.risk_turns -= 1
        if self.detour_turns > 0:
            self.detour_turns -= 1
            if my_pos == self.detour_target:
                self.detour_turns = 0
        if len(self.history) < self.LOOP_WINDOW or self.risk_turns or self.detour_turns:
            return
        cells = {c for c, _ in self.history}
        ate = self.history[0][1] != food_left
        if len(cells) > self.LOOP_CELLS or ate:
            return
        self.stuck_count += 1
        self.history.clear()
        carrying = game_state.get_agent_state(self.index).num_carrying
        p_risk = min(0.9, 0.4 + 0.2 * self.stuck_count)
        if carrying <= self.RISK_MAX_CARRY and random.random() < p_risk:
            self.risk_turns = self.RISK_TURNS
        else:
            ghosts = active or [my_pos]
            self.detour_target = max(self.home_border,
                                     key=lambda b: (min(self.dist(b, g) for g in ghosts), -self.dist(my_pos, b)))
            self.detour_turns = self.DETOUR_TURNS

    # =========================================================================
    # TACTICAL LAYER: adversarial search in close contact (lectures 4 and 5)
    # =========================================================================
    # The strategic layer (PDDL planner, decoy plan, lure...) chooses an action
    # every turn. When an enemy is in close contact - an active ghost near our
    # Pacman, or an invader near our ghost - that action is CHECKED by an
    # adversarial search over the REAL game (full view = perfect information),
    # which works as a filter (see tactical_filter): it vetoes a planned move
    # that gets us eaten and takes a move that surely eats the enemy;
    # otherwise the plan is kept. Search algorithm:
    #   * TACTIC = 'alphabeta': depth-limited Minimax with Alpha-Beta pruning,
    #     iterative deepening until the time budget ends, best move first.
    #   * TACTIC = 'mcts': Monte Carlo Tree Search with UCT (UCB1) selection,
    #     random rollouts of ROLLOUT_DEPTH plies and the same evaluation at
    #     the end of each rollout.
    # Players: this agent (MAX) and the enemy in contact (MIN). The others stay
    # where they are during the search (they are far away).
    # Evaluation (from our side): score, food carried and eaten, being eaten
    # (very bad), eating the enemy (very good), distance to the enemy, and
    # progress towards the goal of the STRATEGIC PLAN (the PDDL planner's next
    # target, or home), so the search keeps the plan's intention and only
    # changes the move when it gets there more safely.

    TACTIC = 'alphabeta'    # tactical search algorithm: 'alphabeta' or 'mcts'
    CONTACT = 5           # maze distance that counts as close contact
    TACTIC_BUDGET = 0.04  # seconds of tactical search per move (light simulation: fast)
    ROLLOUT_DEPTH = 10    # MCTS: plies simulated after expanding a node
    UCT_C = 1.2           # MCTS: exploration constant of UCB1
    DEATH = -500          # filter: a move worth this little means we get eaten
    CATCH = 400           # filter: a move worth this much means we eat the enemy
    MARGIN = 200          # filter: how much better the alternative must be
    IMPROVE = 10000          # filter: a move this much better than the plan (worst case) is taken

    class _Timeout(Exception):
        pass

    def tactical(self, game_state, action):
        """Check `action` with adversarial search when an enemy is in close contact."""
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        if self.guarding or getattr(self, 'decoy_now', False):
            return action  # standing guard / acting as decoy: the strategy knows better
        opp = None
        best_d = self.CONTACT + 1
        for i in self.get_opponents(game_state):
            st = game_state.get_agent_state(i)
            if st.get_position() is None:
                continue
            p = self.cell(st.get_position())
            d = self.dist(my_pos, p)
            hunted = me.is_pacman and not st.is_pacman and st.scared_timer <= 2
            hunting = not me.is_pacman and st.is_pacman and me.scared_timer == 0
            if (hunted or hunting) and d < best_d:
                opp, best_d = i, d
        if opp is None:
            return action

        enemy = game_state.get_agent_state(opp)
        opp_pos = self.cell(enemy.get_position())
        food = self.get_food(game_state).as_list()
        self.tac = dict(
            my_start=self.start, opp_start=game_state.get_initial_agent_position(opp),
            attacking=me.is_pacman, target=self.plan_target, carrying=me.num_carrying,
            opp_carrying=enemy.num_carrying, food=len(food),
            bank=me.num_carrying >= 3 or len(food) <= 2
                 or game_state.data.timeleft // 4 <= self.distance_home(my_pos) + 6)
        # Root of the LIGHT SIMULATION: only what matters in a fight between two agents.
        root = (my_pos, opp_pos, me.num_carrying, enemy.num_carrying,
                frozenset(food), frozenset(self.get_food_you_are_defending(game_state).as_list()),
                frozenset(self.get_capsules(game_state)), frozenset(self.get_capsules_you_are_defending(game_state)),
                me.scared_timer, enemy.scared_timer, 0, 0)
        deadline = time.perf_counter() + self.TACTIC_BUDGET
        try:
            if self.TACTIC == 'mcts':
                values = self.mcts(root, deadline)
            else:
                values = self.alphabeta_root_values(root, deadline)
        except Exception:
            traceback.print_exc()
            return action
        return self.tactical_filter(game_state, action, values)

    def tactical_filter(self, game_state, action, values):
        """
        The search works as a FILTER on the strategic action (the long-term plan
        knows better where to go; a shallow search knows better what happens in
        the next few moves):
          - VETO: if the planned move leads to being eaten (value <= DEATH) and
            another move is clearly better, take the best move instead.
          - OPPORTUNITY: if some move surely eats the enemy (value >= CATCH)
            and the planned one does not, take it.
          - IMPROVEMENT: if the best move is worth IMPROVE more than the planned
            one in the worst case (Minimax assumes the enemy answers perfectly),
            take it.
          - Otherwise keep the planned move.
        """
        if not values:
            return action
        legal = game_state.get_legal_actions(self.index)
        values = {a: v for a, v in values.items() if a in legal}
        if not values:
            return action
        best = max(values, key=values.get)
        planned = values.get(action)
        if planned is None:
            return action
        if planned <= self.DEATH and values[best] > planned + self.MARGIN:
            self.vetoes = getattr(self, 'vetoes', 0) + 1
            return best
        if values[best] >= self.CATCH and planned < self.CATCH - self.MARGIN:
            self.catches = getattr(self, 'catches', 0) + 1
            return best
        if values[best] >= planned + self.IMPROVE:
            self.improves = getattr(self, 'improves', 0) + 1
            return best
        return action

    # --- light simulation of a two-agent fight ----------------------------------
    # State: (my_pos, opp_pos, my_carry, opp_carry, enemy_food, our_food,
    #         enemy_capsules, our_capsules, my_scared, opp_scared, gain, dead)
    # gain = food banked by us minus food banked by the enemy during the search;
    # dead = 0 alive, 1 we were eaten, 2 the enemy was eaten (ends the search).
    # Same rules as the game: a Pacman eats food and capsules on the other
    # side, banks its food when it crosses home, and a Pacman and a ghost on
    # the same cell -> the Pacman dies, unless the ghost is scared.

    def sim_moves(self, pos):
        return [pos] + self.neighbours(pos)

    def sim_step(self, s, mine, nxt):
        mp, op, mc, oc, ef, of, ec, oca, ms, os_, gain, dead = s
        if mine:
            mp = nxt
            if not self.on_my_side(mp):
                if mp in ef:
                    ef = ef - {mp}
                    mc += 1
                if mp in ec:
                    ec = ec - {mp}
                    os_ = 40
            elif mc:
                gain += mc
                mc = 0
            if ms:
                ms -= 1
        else:
            op = nxt
            if self.on_my_side(op):
                if op in of:
                    of = of - {op}
                    oc += 1
                if op in oca:
                    oca = oca - {op}
                    ms = 40
            elif oc:
                gain -= oc
                oc = 0
            if os_:
                os_ -= 1
        if mp == op:
            if not self.on_my_side(mp):   # we are a Pacman on their side
                dead = 2 if os_ else 1
            else:                         # they are a Pacman on our side
                dead = 1 if ms else 2
        return (mp, op, mc, oc, ef, of, ec, oca, ms, os_, gain, dead)

    def tactical_value(self, s):
        """Evaluation function Eval(s): how good the simulated state is for us."""
        mp, op, mc, oc, ef, of, ec, oca, ms, os_, gain, dead = s
        t = self.tac
        if dead == 1:
            return -1000 - 30 * (t['carrying'] + mc)                    # we were eaten
        v = 100.0 * gain + 15 * (t['food'] - len(ef)) + 5 * mc
        if dead == 2:
            return v + 600 + 30 * oc                                   # we ate the enemy
        if t['attacking']:
            if not self.on_my_side(mp) and not os_:
                d = self.dist(mp, op)
                v += 8 * min(d, 4) - (400 if d <= 1 else 0)
            if t['target'] == 'home' or t['bank'] or mc >= 3:
                v -= 3 * self.distance_home(mp)
            elif t['target'] is not None:
                v -= 3 * self.dist(mp, t['target'])                    # progress along the plan
            elif ef:
                v -= 1.5 * min(self.dist(mp, f) for f in ef)
        else:
            if self.on_my_side(op):
                v -= 12 * self.dist(mp, op) + 2 * oc
            else:
                v -= 80                                                # it got away home
        return v

    # --- Minimax with Alpha-Beta pruning (iterative deepening) -----------------

    def alphabeta_root_values(self, root, deadline):
        """Minimax value of every root move (us = MAX, the enemy = MIN), deepening until the deadline."""
        moves = self.sim_moves(root[0])
        values, depth = {}, 2
        while depth <= 20:
            try:
                current = {}
                for nxt in moves:
                    v = self.alphabeta(self.sim_step(root, True, nxt), depth - 1, -INF, INF, False, deadline)
                    current[self.direction(root[0], nxt)] = v
            except self._Timeout:
                break
            values = current
            depth += 2
        self.last_tactic_depth = depth - 2
        return values

    def alphabeta(self, s, depth, alpha, beta, maximizing, deadline):
        if s[11] or depth == 0:
            return self.tactical_value(s)
        if time.perf_counter() > deadline:
            raise self._Timeout()
        if maximizing:
            value = -INF
            for nxt in self.sim_moves(s[0]):
                value = max(value, self.alphabeta(self.sim_step(s, True, nxt), depth - 1, alpha, beta, False, deadline))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break                                    # beta cut
            return value
        value = INF
        for nxt in self.sim_moves(s[1]):
            value = min(value, self.alphabeta(self.sim_step(s, False, nxt), depth - 1, alpha, beta, True, deadline))
            beta = min(beta, value)
            if alpha >= beta:
                break                                        # alpha cut
        return value

    def direction(self, a, b):
        if a == b:
            return Directions.STOP
        return STEPS[(b[0] - a[0], b[1] - a[1])]

    # --- Monte Carlo Tree Search with UCT ----------------------------------------

    def mcts(self, root_state, deadline):
        root = dict(s=root_state, mine=True, n=0, w=0.0, children={}, parent=None,
                    untried=self.sim_moves(root_state[0]))
        while time.perf_counter() < deadline:
            node = root
            # 1. Selection (UCB1)
            while not node['untried'] and node['children']:
                node = self.uct_child(node)
            # 2. Expansion
            if node['untried'] and not node['s'][11]:
                nxt = node['untried'].pop(random.randrange(len(node['untried'])))
                s = self.sim_step(node['s'], node['mine'], nxt)
                child = dict(s=s, mine=not node['mine'], n=0, w=0.0, children={}, parent=node,
                             untried=[] if s[11] else self.sim_moves(s[0] if not node['mine'] else s[1]))
                node['children'][nxt] = child
                node = child
            # 3. Simulation (random rollout, then evaluation)
            value = self.rollout(node['s'], node['mine'])
            # 4. Backpropagation (values from our side, squashed to [-1, 1])
            reward = max(-1.0, min(1.0, value / 600.0))
            while node is not None:
                node['n'] += 1
                node['w'] += reward
                node = node['parent']
        return {self.direction(root_state[0], nxt): 600.0 * c['w'] / c['n']
                for nxt, c in root['children'].items() if c['n'] > 0}

    def uct_child(self, node):
        log_n = math.log(node['n'] + 1)
        sign = 1 if node['mine'] else -1   # the enemy picks what is worst for us

        def ucb(child):
            if child['n'] == 0:
                return INF
            return sign * child['w'] / child['n'] + self.UCT_C * math.sqrt(log_n / child['n'])
        return max(node['children'].values(), key=ucb)

    def rollout(self, s, mine):
        for _ in range(self.ROLLOUT_DEPTH):
            if s[11]:
                break
            pos = s[0] if mine else s[1]
            s = self.sim_step(s, mine, random.choice(self.neighbours(pos) or [pos]))
            mine = not mine
        return self.tactical_value(s)

    def legal(self, game_state, action):
        return action is not None and action in game_state.get_legal_actions(self.index)

    # --- search ---------------------------------------------------------

    def bfs_distances(self, start, blocked=frozenset()):
        """Maze distance from start to every reachable cell, never entering `blocked`."""
        dist = {start: 0}
        frontier = deque([start])
        while frontier:
            c = frontier.popleft()
            for n in self.neighbours(c):
                if n not in dist and n not in blocked:
                    dist[n] = dist[c] + 1
                    frontier.append(n)
        return dist

    def bfs(self, start, targets, blocked=frozenset(), allowed=None):
        """First move of a shortest path to any target avoiding `blocked`; (None, INF) if unreachable."""
        targets = set(targets)
        if not targets:
            return None, INF
        if start in targets:
            return Directions.STOP, 0
        frontier = deque([(start, None, 0)])
        seen = {start}
        while frontier:
            (x, y), first, d = frontier.popleft()
            for (dx, dy), direction in STEPS.items():
                nxt = (x + dx, y + dy)
                if self.walls[nxt[0]][nxt[1]] or nxt in seen or nxt in blocked:
                    continue
                if allowed is not None and not allowed(nxt):
                    continue
                seen.add(nxt)
                step = first or direction
                if nxt in targets:
                    return step, d + 1
                frontier.append((nxt, step, d + 1))
        return None, INF

    def flee(self, game_state, threats):
        actions = [a for a in game_state.get_legal_actions(self.index) if a != Directions.STOP] \
            or game_state.get_legal_actions(self.index)
        if not threats:
            return random.choice(actions)

        def score(action):
            pos = self.cell(game_state.generate_successor(self.index, action).get_agent_position(self.index))
            return min(self.dist(pos, t) for t in threats)
        return max(actions, key=score)

    def keep_distance(self, game_state, threats, target):
        def score(action):
            pos = self.cell(game_state.generate_successor(self.index, action).get_agent_position(self.index))
            if not self.on_my_side(pos):
                return INF
            return abs(min(self.dist(pos, t) for t in threats) - target)
        return min(game_state.get_legal_actions(self.index), key=score)


# =============================================================================
# Strategic layer: builds and solves the planning problem for an attacker
# =============================================================================

class AttackPlanner:
    MAX_FOOD_TARGETS = 6     # nearest food considered as objects of the problem
    HUNT_RANGE = 10          # a scared ghost this close (and in time) becomes the goal
    COMMITMENT = 3           # the current target looks this many steps cheaper (avoids dithering)
    HORIZON = 4              # plan up to 4 more food before going home
    REMOTE_MIN = 2           # capsule only when this much food is guarded
    REMOTE_W = 1.0           # remote-first: weight of the food's depth in its side
    MAX_HAZARD = 0.15        # banking risk model: hazard per step right next to a ghost
    HAZARD_RANGE = 2.0       # banking risk model: how fast the hazard fades with distance

    def __init__(self, agent):
        self.agent = agent
        self.last_plan = []
        self.committed = None    # target cell of the plan being executed

    def solve(self, game_state, quota_limit):
        """Plan and return (target cell, use_danger_free_paths) for the next step, or (None, False)."""
        a = self.agent
        me = game_state.get_agent_state(a.index)
        my_pos = a.cell(me.get_position())
        enemies = a.enemy_states(game_state)
        active = [p for p, e in enemies if not e.is_pacman and e.scared_timer <= 2]
        scared = [(p, e.scared_timer) for p, e in enemies if not e.is_pacman and e.scared_timer > 2]

        danger = a.danger_zone(active, my_pos)

        # Map targets: the nearest food (by safe distance), capsules, home, reachable scared ghosts.
        safe_from_me = a.bfs_distances(my_pos, danger)
        food = a.get_food(game_state).as_list()
        capsules = a.get_capsules(game_state)
        eta = a.ghost_eta(game_state)
        if eta:
            # DEAD-END SAFETY (timed): food inside a dead end is a target only if we can
            # get in, eat it and be back at the exit before any ghost reaches that exit.
            food_pool = [f for f in food if a.pocket_ok(my_pos, f, eta)]
            capsules = [c for c in capsules if a.pocket_ok(my_pos, c, eta)]
            self.trapped_food = len(food) - len(food_pool)
        else:
            food_pool = food
        # CAPSULE AT THE RIGHT TIME: a capsule gives no points; it is a target only when
        # at least REMOTE_MIN food is out of safe reach (guarded), to unlock that food.
        remote = [f for f in food if safe_from_me.get(f, INF) == INF]
        if len(remote) < self.REMOTE_MIN:
            capsules = []
        # REMOTE FIRST: the deepest food (far from their border, the hardest to reach
        # later) is preferred; the easy food near the border is left for later.
        foods = sorted(food_pool, key=lambda f: (safe_from_me.get(f, INF) - self.REMOTE_W * a.distance_home(f),
                                                 a.dist(my_pos, f)))[:self.MAX_FOOD_TARGETS]
        prey = []  # NO HUNTING: eating a ghost gives no points

        targets = {'me': my_pos}
        targets.update({f'f{i}': f for i, f in enumerate(foods)})
        targets.update({f'c{i}': c for i, c in enumerate(capsules)})
        targets.update({f'g{i}': g for i, g in enumerate(prey)})

        # Distances between targets: safe (avoiding ghosts) and free (powered).
        safe, free = {}, {}
        for name, cell in targets.items():
            ds = a.bfs_distances(cell, danger)
            for other, ocell in targets.items():
                if other != name:
                    safe[name, other] = ds.get(ocell, INF)
                    free[name, other] = a.dist(cell, ocell)
            safe[name, 'home'] = min((ds.get(b, INF) for b in a.home_border), default=INF)
            free[name, 'home'] = a.distance_home(cell)

        # Quota of food to bring home.
        carrying = me.num_carrying
        ghost_dist = min((a.dist(my_pos, g) for g in active), default=INF)
        moves_left = game_state.data.timeleft // 4
        quota = min(quota_limit, max(carrying, len(food) - 2))
        if carrying > 0 and (ghost_dist <= 2 or moves_left <= free['me', 'home'] + 4 or len(food) <= 2
                             or self.bank_now(my_pos, carrying, active, foods, safe_from_me, safe, free)):
            quota = carrying
        # Never ask for more food than can be reached: safely, or after a capsule.
        reachable = sum(1 for f in foods if safe_from_me.get(f, INF) < INF)
        if capsules:
            reachable = len(foods)
        quota = min(quota, carrying + reachable, carrying + self.HORIZON)
        if quota <= carrying and carrying == 0:
            return None, False  # nothing reachable: the caller looks for another way in

        goal = {('ghost-eaten',)} if prey else {('banked',)}
        init, actions = self.ground(targets, safe, free, carrying, quota, bool(prey))
        result = plan_astar(init, goal, actions, deadline=a.deadline())
        if result is None and prey:  # cannot reach the ghost: plan to score instead
            goal = {('banked',)}
            init, actions = self.ground(targets, safe, free, carrying, quota, False)
            result = plan_astar(init, goal, actions, deadline=a.deadline())

        self.export_pddl(targets, init, goal, safe, free, carrying, quota)
        if not result:
            self.last_plan = []
            return None, False
        plan, _ = result
        self.last_plan = plan
        first = plan[0]
        powered = first.name.endswith('powered') or first.name == 'hunt-ghost'
        dest = first.args[1]
        if dest == 'home':
            self.committed = None
            return 'home', powered
        self.committed = targets[dest]
        return targets[dest], powered

    def bank_now(self, my_pos, carrying, ghosts, foods, safe_from_me, safe, free):
        """
        MAX-GAIN BANKING: go home now, or eat one more first? Expected value of
        the food that reaches home in each case, with a simple risk model: each
        step taken near an active ghost has a hazard h(g) that grows as the
        ghost gets closer (g = maze distance), and a path of L steps survives
        with probability (1 - h)^L.
            bank now:      carrying       * (1 - h(ghost to me))^(steps home)
            one more food: (carrying + 1) * (1 - h(ghost to food))^(steps to food + food to home)
        With no ghost around the risk is zero, so it keeps eating; with a ghost
        close the trip gets risky and it banks what it carries.
        """
        if carrying == 0 or not ghosts:
            return False
        a = self.agent

        def hazard(cell):
            g = min(a.dist(cell, x) for x in ghosts)
            return min(self.MAX_HAZARD, self.MAX_HAZARD * math.exp(-(g - 2) / self.HAZARD_RANGE))

        steps_home = safe['me', 'home'] if safe['me', 'home'] < INF else free['me', 'home']
        bank = carrying * (1 - hazard(my_pos)) ** steps_home
        best_more = 0.0
        for f in foods:
            d = safe_from_me.get(f, INF)
            if d == INF:
                continue
            steps = d + a.distance_home(f)
            best_more = max(best_more, (carrying + 1) * (1 - max(hazard(f), hazard(my_pos))) ** steps)
        return bank >= best_more

    def ground(self, targets, safe, free, carrying, quota, hunting):
        """Initial state and grounded actions of the PDDL problem."""
        counts = list(range(carrying, max(quota, carrying) + 1))
        init = {('at', 'me'), ('carrying', carrying)}
        init |= {('food', t) for t in targets if t[0] == 'f'}
        init |= {('capsule', t) for t in targets if t[0] == 'c'}
        if hunting:
            init |= {('scared-ghost', t) for t in targets if t[0] == 'g'}
        enough = {n for n in counts if n >= quota}

        actions = []
        places = list(targets)

        def cost(frm, to, d):
            # Plan commitment: moving to the target we are already heading to is a
            # bit cheaper, so a new plan must be clearly better to make us turn.
            if frm == 'me' and targets.get(to) == self.committed:
                return max(0.5, d - self.COMMITMENT)
            return d

        for frm in places:
            for to in places:
                if frm == to:
                    continue
                if to[0] == 'f':
                    for n in counts[:-1]:
                        pre_n, eff_n = ('carrying', n), ('carrying', n + 1)
                        if safe[frm, to] < INF:
                            actions.append(Action('eat-food', (frm, to, n, n + 1),
                                                  {('at', frm), ('food', to), pre_n},
                                                  {('at', to), eff_n}, {('at', frm), ('food', to), pre_n},
                                                  cost(frm, to, safe[frm, to])))
                        actions.append(Action('eat-food-powered', (frm, to, n, n + 1),
                                              {('powered',), ('at', frm), ('food', to), pre_n},
                                              {('at', to), eff_n}, {('at', frm), ('food', to), pre_n},
                                              cost(frm, to, free[frm, to])))
                elif to[0] == 'c' and safe[frm, to] < INF:
                    actions.append(Action('eat-capsule', (frm, to), {('at', frm), ('capsule', to)},
                                          {('at', to), ('powered',)}, {('at', frm), ('capsule', to)},
                                          cost(frm, to, safe[frm, to])))
                elif to[0] == 'g' and hunting:
                    actions.append(Action('hunt-ghost', (frm, to), {('at', frm), ('scared-ghost', to)},
                                          {('at', to), ('ghost-eaten',)}, {('at', frm), ('scared-ghost', to)},
                                          cost(frm, to, free[frm, to])))
            for n in enough:
                if safe[frm, 'home'] < INF:
                    actions.append(Action('return-home', (frm, 'home', n), {('at', frm), ('carrying', n)},
                                          {('at', 'home'), ('banked',)}, {('at', frm)}, safe[frm, 'home']))
                actions.append(Action('return-home-powered', (frm, 'home', n),
                                      {('powered',), ('at', frm), ('carrying', n)},
                                      {('at', 'home'), ('banked',)}, {('at', frm)}, free[frm, 'home']))
        return init, actions

    def export_pddl(self, targets, init, goal, safe, free, carrying, quota):
        """Write the problem as PDDL when PACMAN_PDDL_DIR is set (debugging only)."""
        folder = os.environ.get('PACMAN_PDDL_DIR')
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        names = list(targets) + ['home']
        counts = range(carrying, max(quota, carrying) + 1)

        def fact(f):
            if f[0] == 'carrying':
                return f'(carrying n{f[1]})'
            return '(' + ' '.join(map(str, f)) + ')'
        lines = ['(define (problem pacman-turn) (:domain pacman-strategy)',
                 '  (:objects ' + ' '.join(names) + ' - target ' + ' '.join(f'n{n}' for n in counts) + ' - count)',
                 '  (:init', '    (home home)']
        lines += ['    ' + fact(f) for f in sorted(init, key=str)]
        lines += [f'    (next n{n} n{n + 1})' for n in list(counts)[:-1]]
        lines += [f'    (enough n{n})' for n in counts if n >= quota]
        for (x, y), d in free.items():
            lines.append(f'    (= (dist {x} {y}) {d})')
            if safe[x, y] < INF:
                lines.append(f'    (safe {x} {y})')
        lines += ['    (= (total-cost) 0))', '  (:goal (and ' + ' '.join(fact(g) for g in goal) + '))',
                  '  (:metric minimize (total-cost)))']
        with open(os.path.join(folder, 'domain.pddl'), 'w') as f:
            f.write(PACMAN_DOMAIN)
        with open(os.path.join(folder, f'problem_agent{self.agent.index}.pddl'), 'w') as f:
            f.write('\n'.join(lines) + '\n')


# =============================================================================
# Agents
# =============================================================================

class PlannerAttacker(PlannerAgentBase):
    QUOTA = 12
    HELP_CARRY = 2   # help the defender against an invader carrying at least this much food
    REFLEX_MAX = 8   # turns the attacker may chase an invader next to it before going back to attack

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.planner = AttackPlanner(self)

    def choose_action(self, game_state):
        me = game_state.get_agent_state(self.index)
        if self.team_mode(game_state) == 'decoy':
            return self.play_decoy_plan(game_state)
        if self.team_mode(game_state) == 'lock' and not (me.is_pacman and me.num_carrying > 0):
            return self.defend(game_state, lock=True)
        return self.attack(game_state, self.QUOTA)

    def attack(self, game_state, quota):
        """Run the strategic planner and execute the first step of its plan."""
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        enemies = self.enemy_states(game_state)
        active = [p for p, e in enemies if not e.is_pacman and e.scared_timer <= 2]
        self.check_loop(game_state, my_pos, active)
        danger = self.danger_zone(active, my_pos)

        # Detour after a loop: retreat and enter through another border crossing.
        if self.detour_turns > 0 and self.detour_target is not None:
            if me.num_carrying > 0 and me.is_pacman:
                action, _ = self.bfs(my_pos, self.home_border, danger)
            else:
                action, _ = self.bfs(my_pos, [self.detour_target], danger)
            if self.legal(game_state, action) and action != Directions.STOP:
                return action
            self.detour_turns = 0

        # Reactive reflex: still a ghost at home and an invader next to us -> eat it.
        if not me.is_pacman and me.scared_timer == 0:
            close = [p for p, e in enemies if e.is_pacman and self.dist(my_pos, p) <= 2]
            self.reflex_turns = getattr(self, 'reflex_turns', 0) + 1 if close else 0
            if close and self.reflex_turns <= self.REFLEX_MAX:
                action, _ = self.bfs(my_pos, close, allowed=self.on_my_side)
                if self.legal(game_state, action):
                    return action

        # HELP: still at home and an invader is carrying food on our side -> help catch it,
        # coming from the other side (pincer) while the teammate chases it. A decoy that
        # carries nothing does not trigger this, so it cannot drag us around.
        if not me.is_pacman and me.scared_timer == 0:
            loaded = [p for p, e in enemies if e.is_pacman and e.num_carrying >= self.HELP_CARRY]
            if loaded:
                mate = self.mate_chaser(game_state)
                action = self.pincer(game_state, my_pos, loaded, mate)
                if action is None:
                    action, _ = self.bfs(my_pos, loaded, allowed=self.on_my_side)
                if self.legal(game_state, action):
                    self.helps = getattr(self, 'helps', 0) + 1
                    return action

        target, powered = self.planner.solve(game_state, quota)
        self.plan_target = target  # what the strategic plan is heading to (cell or 'home')
        blocked = frozenset() if powered else danger
        if target == 'home':
            action, _ = self.bfs(my_pos, self.home_border, blocked)
        elif target is not None:
            action, _ = self.bfs(my_pos, [target], blocked)
        else:
            action = None
        if self.legal(game_state, action) and action != Directions.STOP:
            return action

        # No plan. Carrying food: bring it home. Empty: move along our border to the
        # crossing point farthest from the enemy ghosts and try again from there.
        if me.num_carrying > 0 or me.is_pacman:
            action, _ = self.bfs(my_pos, self.home_border, danger)
            if self.legal(game_state, action) and action != Directions.STOP:
                return action
        if active:
            entry = max(self.home_border, key=lambda b: (min(self.dist(b, g) for g in active), -self.dist(my_pos, b)))
            action, _ = self.bfs(my_pos, [entry], danger)
            if self.legal(game_state, action) and action != Directions.STOP:
                return action
        return self.flee(game_state, active)


class PlannerDefender(PlannerAttacker):
    # Raids (going to eat while nobody threatens our side), per team mode.
    RAID = {'attack':   dict(safe=5, back=3, quota=4),
            'balanced': dict(safe=8, back=5, quota=2)}

    def choose_action(self, game_state):
        me = game_state.get_agent_state(self.index)
        enemies = self.enemy_states(game_state)
        mode = self.team_mode(game_state)
        if not enemies:
            return Directions.STOP
        if mode == 'decoy':
            return self.play_decoy_plan(game_state)
        if mode == 'lock':
            if me.is_pacman and me.num_carrying > 0:
                return self.attack(game_state, me.num_carrying)
            return self.defend(game_state, lock=True)

        invaders = [p for p, e in enemies if e.is_pacman]
        threat = min(0 if e.is_pacman else self.distance_home(p) for p, e in enemies)
        raid = self.RAID[mode]
        raiding = me.is_pacman or me.num_carrying > 0
        if not invaders and (threat >= raid['safe'] or (raiding and threat >= raid['back'])):
            return self.attack(game_state, raid['quota'])
        return self.defend(game_state)


class FlexAgent(PlannerDefender):
    """
    NO FIXED ROLES. Every turn the team decides who attacks and who defends:
      - an agent on the enemy side carrying food stays the attacker;
      - otherwise the defender is the agent closer to the threat (the nearest
        invader, or else our own border), the attacker is the other one;
      - hysteresis: roles only swap when the other agent is SWAP_GAIN cells
        closer to the threat, so they do not swap every turn.
    The attacker plays PlannerAttacker's strategy and the defender
    PlannerDefender's (guard, chase, raid when safe); LOCK and the decoy plan
    work as before for both.
    """
    SWAP_GAIN = 3
    roles = {}   # shared by the team: team -> (timeleft, defender index)

    def defender_index(self, game_state):
        key = tuple(self.team)
        tl = game_state.data.timeleft
        stored = self.roles.get(key)
        if stored and stored[0] == tl + 1 or stored and stored[0] == tl:
            return stored[1]
        enemies = self.enemy_states(game_state)
        invaders = [p for p, e in enemies if e.is_pacman]
        st = {i: game_state.get_agent_state(i) for i in self.team}
        pos = {i: self.cell(st[i].get_position()) for i in self.team}

        def threat_cost(i):
            if invaders:
                return min(self.dist(pos[i], p) for p in invaders)
            return self.distance_home(pos[i])
        loaded = [i for i in self.team if st[i].is_pacman and st[i].num_carrying > 0]
        current = stored[1] if stored else self.team[-1]
        if len(loaded) == 1:
            defender = [i for i in self.team if i not in loaded][0]
        else:
            other = [i for i in self.team if i != current][0]
            defender = other if threat_cost(other) + self.SWAP_GAIN <= threat_cost(current) else current
        self.roles[key] = (tl, defender)
        return defender

    def choose_action(self, game_state):
        if self.defender_index(game_state) == self.index:
            return PlannerDefender.choose_action(self, game_state)
        return PlannerAttacker.choose_action(self, game_state)
