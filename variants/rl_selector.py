# rl_selector.py  (Oreo)  -  frankenstein + a deep RL plan selector (see the end of the file)
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# =============================================================================
# APEX  -  plan-based capture-the-flag team that plays by the observation rule
# =============================================================================
# HOW IT THINKS (one turn, < 0.1 s):
#   1. SEE: the normal observation of rule 2.5. Enemies more than 5 cells away
#      are hidden; their public fields (is_pacman, num_carrying, scared_timer,
#      num_returned) are still read, as the course staff showed.
#   2. TRACK (ENEMY TRACKER): an exact HMM filter per enemy, shared by the
#      team. Only the enemy that just moved is advanced; then every hidden
#      enemy is conditioned on its noisy distance (+-6), on NOT being within 5
#      cells of us, on its side of the map (is_pacman), and on our food or
#      capsule that just vanished (it was eaten exactly there). Kills reset it
#      to its start. Hidden enemies are filled in at their most likely cell,
#      never within sight, so the search below only ever fights real contacts.
#   3. DECIDE ROLES (FlexAgent): the agent closer to the threat defends, the
#      other attacks; an attacker carrying food stays attacker. Team modes by
#      score and time: attack / balanced / LOCK (lead >= 4 or near the end:
#      both defend) / DECOY (end of a game we are not winning).
#   4. ATTACK: a PDDL-style planner picks the food to take (deep food first,
#      timed dead-end safety: a pocket is entered only if we can get out
#      before any ghost reaches its exit), when to bank, and when a capsule is
#      worth it (power play: collect remote food during the scare).
#   5. DEFEND: guard one step behind the border facing the enemy most likely
#      to cross; chase invaders, pincer with the teammate from the other side,
#      lure ghosts that camp at the border.
#   6. FIGHT: when an enemy is within 5 cells, depth-limited Alpha-Beta on a
#      light simulation vetoes moves that get us eaten.
#
# ADDED ON TOP OF a_star_pro (each one measured, see below):
#   - Legal tracking instead of the full-view observation_function.
#   - SCARED ATTACK: a defender scared for more than SCARED_ATTACK (10) moves
#     attacks instead of shadowing invaders it cannot eat.
#   - CONTAIN: an invader carrying >= CONTAIN_CARRY (4) food and >= 3 cells
#     away is not followed (a chaser behind it never catches it): the defender
#     takes the move that leaves it the smallest region it reaches before us,
#     above all cutting it off from its side, so the haul never gets home.
#   - LOCK HELP: protecting a lead, both defenders hunt an invader carrying
#     >= LOCK_HELP_CARRY (3) food instead of each marking one enemy.
#   - COUNTER-RAID: when both enemies are on our side nobody guards their
#     food, so the attacker attacks instead of becoming a third chaser.
#   - MORE VOLUME: measured, apex was safe (banked 82% of what it ate, lost
#     1.4 food/game to deaths) but ate too little (13 food/game). So it now
#     keeps attacking until a 6-point lead (LOCK_LEAD 6, was 4) and the
#     defender raids more (quota 6 / 4 instead of 4 / 2, from closer to
#     the border). Banked food 10.8 -> 12.5 per game, average margin
#     +4.0 -> +4.9 (112 games); 144 other games: 76% -> 78% points.
#     Tested and dropped: bolder banking (HORIZON 8, MAX_HAZARD 0.10): it
#     ate and banked less.
#   - SCARED SAFETY: after an invader eats our capsule, our scared ghosts
#     used to walk next to it and get eaten (~40 moves lost each). A scared
#     ghost on our side now never steps on or next to a seen invader.
#   - SIEGE: when we are behind and the enemy has not attacked for 30 moves
#     (both its agents parked at home to defend the lead), one of ours pulls
#     a defender away (decoy) while the other scores.
#     Scared safety + siege, league set (your a_star_pro, pddl_astar,
#     alphabeta; default + RANDOM; both colours): 68% -> 73% points on seeds
#     1-9 (108 games), 61% -> 66% on fresh seeds 10-27 (216 games).
#     Tested and dropped: faster siege (15 moves), siege on ties, a capsule
#     guard (stand on our capsule when an invader nears it): no gain.
#   - OPEN SIDE / RACE (big maps): when both enemies are on our side as
#     Pacmen, nobody guards their food. On maps with more than 40 food left
#     on their side, our attacker no longer makes short 4-food trips (it
#     keeps eating, up to 60, until a ghost shows up) and the defender
#     joins the raid instead of chasing (a lone ghost rarely catches a
#     Pacman in a big open maze). Against a strong all-in attacking team on
#     jumboCapture: 0-0-32 (avg -86) -> 9-0-23 (avg -24); default, RANDOM
#     and office unchanged; your league set unchanged (73%).
#   - TRAP (two ghosts together): when both our agents are ghosts at home
#     within TRAP_RANGE (10) of an invader carrying food, they choose their
#     moves jointly (every pair of moves scored by the region the invader
#     still controls, huge if it can still reach its side) and each plays
#     its half, closing the exits from both ends. Against a strong all-in
#     team (128 games): 59 -> 64 wins (default 8 -> 12, office 27 -> 29);
#     your league set (216 games): 69% -> 70%.
#   - CAPSULE BLITZ (big maps, > 40 food per side): both agents set up on
#     their side, the closer one eats the capsule once both are there, both
#     collect food (deepest first, each its own targets, only what they can
#     still bring home before the scare ends), run home, then defend while
#     leading. Also fixes power_step (inherited empty from a_star_pro, so the
#     power play after a capsule never ran). Against a strong all-in team on
#     jumboCapture: 19-1-44 -> 37-1-26 (avg -24.6 -> -4.4, 64 games); small
#     maps unchanged (the plan leaves our side open too long there); league
#     set unchanged (70%).
#
# RESULTS (headless, 1200 moves, both colours; see below)
#   Against 12 teams (your full-view a_star_pro, pddl_astar, alphabeta, a_star,
#   mcts and 7 external teams) on default/RANDOM/office/jumbo, seeds 3-4:
#     first legal version 145-24-23 (82% points); full-view a_star_pro on
#     the same games 148-8-36 (79%).
#   Tuning, 228 games on 7 layouts (jumbo, office, RANDOM, default, alley,
#   distant) vs 9 teams, seeds 5-8:
#     before CONTAIN/LOCK HELP/COUNTER-RAID  167-22-39, 8 losses by 10+
#     this version                           169-22-37, 4 losses by 10+
#     big maps only (jumbo, office, RANDOM): 76% -> 83% points.
#   Tested and dropped: SCARED_ATTACK = 5 (same); CONTAIN on any invader
#   (71%: full-view invaders slip past); CONTAIN_CARRY = 8 (77%, as many
#   heavy losses as before).
# Move time: max ~0.09 s (limit 1 s); start-up well under 15 s.
#
# The original descriptions of a_star_pro and its ancestors follow.
#
# a_star_pro.py description:
# -----------------------------------------------------------------------------
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
# 6. TWO ROLES vs DOUBLE ATTACKERS (RoleAttacker + RoleGuard, small maps).
#    Once both enemies have been on our side at the same time for ROLES_AFTER
#    (15) turns, the team stops swapping roles: the attacker always attacks and
#    eats their capsule alone; the guard stays home, cuts invaders off from
#    their side, stands between an invader and our capsule (never on it) and,
#    when we are not ahead, snipes the one food it can bring back before any
#    ghost can intercept it. If our ghosts are scared or an invader is about to
#    eat our capsule, both attack. Big maps keep FlexAgent + capsule blitz.
#    vs the external double-attacker (64 games/map): default 21-43 -> 36-27,
#    total 143 -> 155 of 256; league (216 games) 70% -> 69% points.
#    Discarded: fixed top/bottom lanes (jumbo 37 -> 17), roles from move 1
#    (league 53%), roles on big maps (jumbo 15), lock from +2 (RANDOM -3).
#
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
from contest.game import Directions, Configuration, Actions
from contest.util import nearest_point


# =============================================================================
# ENEMY TRACKER (legal observation, rule 2.5)
# =============================================================================
# The planner below was written for a full view of the board. Here every agent
# receives the normal observation (make_observation: enemies further than 5
# cells from both of us are hidden) and the hidden enemies are filled in with
# their most likely cell according to an exact HMM filter shared by the team:
#   - only the enemy that moved since our team's last observation is advanced
#     (uniform over its legal moves, staying included);
#   - every hidden enemy is conditioned on the noisy distance (true +- 6), on
#     NOT being within 5 cells of either of us, and on its side of the board
#     (is_pacman is part of every observation);
#   - food or a capsule of ours that disappeared was eaten by the enemy that
#     just moved, exactly at that cell;
#   - an enemy we eat (or that becomes inconsistent with the evidence) is reset
#     to its start cell.
# A filled-in enemy is never placed within sight of us, so the tactical search
# (which works on contacts within 5 cells) only ever uses real positions.

TRACK_SONAR = 6
TRACK_SIGHT = 5


class EnemyTracker(object):

    def __init__(self, agent, game_state):
        walls = game_state.get_walls()
        self.cells = [(x, y) for x in range(walls.width) for y in range(walls.height) if not walls[x][y]]
        self.idx = {c: i for i, c in enumerate(self.cells)}
        self.V = len(self.cells)
        self.nb = []
        for (x, y) in self.cells:
            self.nb.append([self.idx[p] for p in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if p in self.idx])
        half = walls.width // 2
        red = game_state.is_on_red_team(agent.index)
        self.red = red
        # True where an enemy is a Pacman (our side)
        self.our_side = [(c[0] < half) == red for c in self.cells]
        self.team = sorted(agent.get_team(game_state))
        self.opps = sorted(agent.get_opponents(game_state))
        self.start = {o: self.idx[game_state.get_initial_agent_position(o)] for o in self.opps}
        self.belief = {}
        for o in self.opps:
            b = [0.0] * self.V
            b[self.start[o]] = 1.0
            self.belief[o] = b
        self.updates = 0
        self.last_food = None
        self.last_caps = None
        self.last_turn = None

    def elapse(self, o):
        b, nb = self.belief[o], self.nb
        new = [0.0] * self.V
        for i in range(self.V):
            p = b[i]
            if p > 0:
                ns = nb[i]
                share = p / (len(ns) + 1)
                new[i] += share
                for n in ns:
                    new[n] += share
        self.belief[o] = new

    def condition(self, b, nd, my_pos, team_pos, is_pacman):
        cells, side = self.cells, self.our_side
        mx, my = my_pos
        new = [0.0] * self.V
        total = 0.0
        for i in range(self.V):
            p = b[i]
            if p <= 0 or side[i] != is_pacman:
                continue
            x, y = cells[i]
            if nd is not None and abs(abs(x - mx) + abs(y - my) - nd) > TRACK_SONAR:
                continue
            seen = False
            for tp in team_pos:
                if tp is not None and abs(x - tp[0]) + abs(y - tp[1]) <= TRACK_SIGHT:
                    seen = True
                    break
            if seen:
                continue
            new[i] = p
            total += p
        if total <= 0:
            return None
        inv = 1.0 / total
        return [p * inv for p in new]

    def update(self, agent, obs):
        """Observation of `agent`, just before it moves."""
        turn = obs.data.timeleft
        if self.last_turn == turn:
            return
        self.last_turn = turn
        me = agent.index
        if self.updates > 0 or me != 0:
            mover = (me - 1) % 4
            if mover in self.opps:
                self.elapse(mover)
        self.updates += 1
        mover = (me - 1) % 4

        if self.red:
            food = set(obs.get_red_food().as_list())
            caps = set(obs.get_red_capsules())
        else:
            food = set(obs.get_blue_food().as_list())
            caps = set(obs.get_blue_capsules())
        eaten = []
        if self.last_food is not None:
            eaten = [self.idx[f] for f in self.last_food - food] + [self.idx[c] for c in self.last_caps - caps]
        self.last_food, self.last_caps = food, caps

        team_pos = [obs.get_agent_position(t) for t in self.team]
        noisy = obs.get_agent_distances()
        my_pos = obs.get_agent_position(me)
        for o in self.opps:
            pos = obs.get_agent_position(o)
            st = obs.get_agent_state(o)
            if pos is not None:
                b = [0.0] * self.V
                b[self.idx[nearest_point(pos)]] = 1.0
                self.belief[o] = b
                continue
            if eaten and o == mover:
                b = [0.0] * self.V
                for e in eaten:
                    b[e] = 1.0 / len(eaten)
                self.belief[o] = b
                continue
            nd = noisy[o] if noisy is not None else None
            new = self.condition(self.belief[o], nd, my_pos, team_pos, st.is_pacman)
            if new is None:
                b0 = [0.0] * self.V
                b0[self.start[o]] = 1.0
                new = self.condition(b0, nd, my_pos, team_pos, st.is_pacman)
                if new is None:
                    new = self.condition([1.0] * self.V, nd, my_pos, team_pos, st.is_pacman)
                if new is None:
                    new = [1.0 / self.V] * self.V
            self.belief[o] = new

    def mark_dead(self, o):
        b = [0.0] * self.V
        b[self.start[o]] = 1.0
        self.belief[o] = b

    def estimate(self, o):
        b = self.belief[o]
        return self.cells[max(range(self.V), key=b.__getitem__)]

    def fill(self, obs):
        """Place every hidden enemy at its most likely cell."""
        for o in self.opps:
            st = obs.data.agent_states[o]
            if st.configuration is None:
                st.configuration = Configuration(self.estimate(o), Directions.STOP)
        return obs


TRACKERS = {}   # one tracker per team and game, shared by the two agents


def get_tracker(agent, game_state):
    key = (id(agent.team_key), agent.red)
    tr = TRACKERS.get(key)
    if tr is None:
        tr = EnemyTracker(agent, game_state)
        TRACKERS[key] = tr
    return tr


#################
# Team creation #
#################

# =============================================================================
# FRANKENSTEIN: every Oreo plan in one file, switchable with one setting.
#   PLAN = 'apex'          (default) best all-rounder: FlexAgent roles + capsule
#                          blitz on big maps; switches to attacker + guard on small
#                          maps when both enemies invade together.
#                          vs friend 155/256, league ~69%.
#   PLAN = 'capsule_race'  counter for two-agent capsule rushers (the friend's
#                          agent): rusher eats their capsule from move 1, keeper
#                          holds the ring around ours (small maps, 1 capsule per
#                          side; elsewhere = apex). vs friend 183/256 (default
#                          64-0), league ~52%.
#   PLAN = 'two_roles'     fixed attacker + guard on small maps from move 1.
#                          vs friend 147/256, league ~53%.
#   PLAN = 'flex'          the previous apex (no role switch): FlexAgent x2.
#                          vs friend 143/256, league ~70%.
#   PLAN = 'adaptive'      (default here) watch-and-switch loop: starts with the
#                          capsule keeper, first decision at move FIRST (100)
#                          from the last OBS (50) moves, then every HOLD (150).
#                          Enemy averaging >= AGGR_PACMEN (1.2) Pacmen on our
#                          side = attacking -> keep the capsule lock; otherwise
#                          -> apex. vs friend: default 51-10; league ~59%.
#                          FIRST trades one for the other (measured):
#                            FIRST=100: friend default 51-10, league 59%
#                            FIRST= 70: friend default 45-19, league 62%
#                            FIRST= 80: friend default 39-24, league 65%
#                            (apex:       friend default 36-27, league 69%)
# FAILURE FIXES (each one an "if" for a situation where we lost, found by
# replaying the games frankenstein lost and apex did not):
#   FIX 1  a lone invader already carrying >= KEEP_CHASE_CARRY (2) food is not a
#          capsule runner: the keeper leaves its ring and defends normally.
#   FIX 2  attacker with no safe plan for NOPLAN_RISK (12) turns in a row (stuck
#          circling at their border): take the risk mode and go in.
#   FIX 3  guard/keeper, behind and no invader for QUIET_JOIN (20) turns: join
#          the attack.
#   FIX 4  siege decoy that the defenders ignore (no food eaten for DECOY_STALL
#          (25) turns): the decoy attacks too for DECOY_PUSH (40) turns.
#   FIX 5  our capsule eaten and both enemies looting our side: the scared guard
#          loots their empty side instead of watching.
#   Measured (league = 108 games per map vs a_star_pro, pddl_astar, alphabeta;
#   friend = 64 games per map):
#                 league default  league RANDOM  friend default  friend RANDOM
#     before          45%             -              51-10          25-38
#     FIX 1-2         54%             -              46-15          25-38
#     FIX 1-4         57%            69%             46-17          25-38
#     FIX 1-5         56%            71%             48-14          25-38
#   Tried and dropped: keeper on maps with two capsules per side (friend RANDOM
#   21-41, league RANDOM 52%).
# The plan can also be picked without editing the file with the environment
# variable OREO_PLAN, or with create_team(..., plan='capsule_race').
# Built from: alphabeta.py (tactical alpha-beta), pddl_astar.py (A* PDDL
# planner), a_star_pro.py (both merged + tracker), apex.py, capsule_race.py.
# =============================================================================
PLAN = 'adaptive'
PLANS = {
    'apex':         ('RoleAttacker', 'RoleGuard', {}),
    'capsule_race': ('CapsuleRusher', 'CapsuleKeeper', {}),
    'two_roles':    ('RoleAttacker', 'RoleGuard', {'ROLES_AFTER': 0}),
    'flex':         ('FlexAgent', 'FlexAgent', {}),
    'adaptive':     ('AdaptiveRusher', 'AdaptiveKeeper', {}),
}
PLAN_DEFAULTS = {'ROLES_AFTER': 15}


def create_team(first_index, second_index, is_red,
                first=None, second=None, num_training=0, plan=None):
    plan = plan or os.environ.get('OREO_PLAN') or PLAN
    if plan not in PLANS:
        plan = 'apex'
    p_first, p_second, overrides = PLANS[plan]
    for k, v in PLAN_DEFAULTS.items():
        setattr(PlannerAgentBase, k, overrides.get(k, v))
    agents = {cls.__name__: cls for cls in (PlannerAttacker, PlannerDefender, FlexAgent, RoleAttacker, RoleGuard,
                                            CapsuleRusher, CapsuleKeeper, AdaptiveRusher, AdaptiveKeeper)}
    team = [agents[first or p_first](first_index), agents[second or p_second](second_index)]
    key = object()   # one shared tracker for this team in this game
    for a in team:
        a.team_key = key
    return team


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
    BIG_MAP_FOOD = 40  # more food than this on their side at the start: a big map
    LOCK_LEAD = 6      # winning by this much: both agents defend
    POST_LOCK = True   # after the capsule blitz, with any lead: both defend
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
    DECOY_STALL = 25   # FIX 4: decoy turns without progress before it attacks too
    DECOY_PUSH = 40    # ... for this many turns, then decoy again
    decoy_roles = {}   # shared by the two agents of the team: team -> (timeleft, decoy index)
    # Anti-loop: stuck = last LOOP_WINDOW attacking turns on <= LOOP_CELLS cells without eating.
    LOOP_WINDOW = 10
    LOOP_CELLS = 4
    RISK_TURNS = 6      # turns of risk mode (may pass next to a ghost)
    DETOUR_TURNS = 10   # turns of detour (retreat and enter somewhere else)
    RISK_MAX_CARRY = 2
    NOPLAN_RISK = 12    # FIX 2: turns without a safe plan before taking the risk mode
    # take risks only while carrying at most this much food

    team_key = None

    def observation_function(self, game_state):
        # Legal observation (rule 2.5); hidden enemies are filled in by the tracker.
        obs = game_state.make_observation(self.index)
        try:
            tracker = get_tracker(self, obs)
            tracker.update(self, obs)
            self.real_obs = obs.deep_copy()
            return tracker.fill(obs)
        except Exception:
            traceback.print_exc()
            return obs

    def note_kills(self, action):
        # An enemy we eat with this move goes back to its start cell.
        obs = getattr(self, 'real_obs', None)
        if obs is None:
            return
        tracker = get_tracker(self, obs)
        succ = obs.generate_successor(self.index, action)
        for o in tracker.opps:
            p0 = obs.get_agent_position(o)
            p1 = succ.get_agent_position(o)
            if p0 is not None and p1 is not None and nearest_point(p1) == obs.get_initial_agent_position(o) \
                    and nearest_point(p0) != obs.get_initial_agent_position(o):
                tracker.mark_dead(o)

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
        self.big_map = len(self.get_food(game_state).as_list()) > self.BIG_MAP_FOOD
        # LANES: their food is split in a bottom and a top half (same amount each);
        # the agent that starts higher takes the top lane.
        ys = sorted(y for _, y in self.get_food(game_state).as_list())
        self.lane_split = ys[len(ys) // 2] if ys else self.walls.height // 2
        starts = {i: game_state.get_initial_agent_position(i)[1] for i in self.team}
        self.lane_top = max(self.team, key=lambda i: (starts[i], i)) == self.index

    LANES = False
    CAP_GUARD = 8       # an invader this close to our capsule: a defender blocks it
    CAP_LOST_SCARE = 99  # our ghosts scared at least this long: both attack

    ROLES_AFTER = 15    # turns with both enemies on our side at once: switch to attacker + guard
    both_in = {}        # shared by the team: team -> [last timeleft, count]

    def roles_on(self, game_state):
        key, tl = tuple(self.team), game_state.data.timeleft
        st = self.both_in.get(key)
        if st is None or st[0] < tl:
            st = [tl + 1, 0]
            self.both_in[key] = st
        if st[0] != tl:
            st[0] = tl
            if all(e.is_pacman for e in (game_state.get_agent_state(i) for i in self.get_opponents(game_state))):
                st[1] += 1
        return not self.big_map and st[1] >= self.ROLES_AFTER

    def lane_food(self, game_state, food):
        """Food of my lane while the teammate is also on their side, else all of it."""
        if not self.LANES:
            return food
        mate = [i for i in self.team if i != self.index][0]
        if not game_state.get_agent_state(mate).is_pacman:
            return food
        lane = [f for f in food if (f[1] >= self.lane_split) == self.lane_top]
        return lane or food

    def capsule_threat(self, game_state):
        """(invader, capsule) for the invader closest to one of our capsules, within CAP_GUARD, or None."""
        caps = self.get_capsules_you_are_defending(game_state)
        if not caps or not self.roles_on(game_state):
            return None
        best = None
        for p, e in self.enemy_states(game_state):
            if e.is_pacman:
                for c in caps:
                    d = self.dist(p, c)
                    if d <= self.CAP_GUARD and (best is None or d < best[0]):
                        best = (d, p, c)
        return best and best[1:]

    def capsule_lost(self, game_state):
        """Our capsule is eaten (our ghosts scared) or an invader will get it before any of us."""
        if not self.roles_on(game_state):
            return False
        for i in self.team:
            if game_state.get_agent_state(i).scared_timer >= self.CAP_LOST_SCARE:
                return True
        t = self.capsule_threat(game_state)
        if t is None:
            return False
        inv, cap = t
        d_inv = self.dist(inv, cap)
        if d_inv > 2:
            return False
        for i in self.team:
            st = game_state.get_agent_state(i)
            if not st.is_pacman and self.dist(self.cell(st.get_position()), cap) < d_inv:
                return False
        return True

    def guard_capsule(self, game_state):
        """
        CAPSULE DENIAL. An invader near our capsule: the ghost that gets there first
        stays between them, always closer to the capsule than the invader (never on
        it: a Pacman eating it would also eat us), and moves in on the invader.
        """
        t = self.capsule_threat(game_state)
        if t is None:
            return None
        inv, cap = t
        me = game_state.get_agent_state(self.index)
        if me.is_pacman or me.scared_timer > 0:
            return None
        my_pos = self.cell(me.get_position())
        mates = [self.cell(game_state.get_agent_state(i).get_position()) for i in self.team
                 if i != self.index and not game_state.get_agent_state(i).is_pacman
                 and game_state.get_agent_state(i).scared_timer == 0]
        mine = self.dist(my_pos, cap)
        if any(self.dist(m, cap) < mine or (self.dist(m, cap) == mine and i < self.index)
               for i, m in zip([i for i in self.team if i != self.index], mates)):
            return None      # the teammate guards it
        d_inv = self.dist(inv, cap)
        best, best_v = None, None
        for a in game_state.get_legal_actions(self.index):
            dx, dy = Actions.direction_to_vector(a)
            c = (int(my_pos[0] + dx), int(my_pos[1] + dy))
            if not self.on_my_side(c):
                continue
            if c == inv:
                return a
            if c == cap or self.dist(c, cap) >= d_inv:
                continue
            v = self.dist(c, inv)
            if best_v is None or v < best_v:
                best, best_v = a, v
        if best is None:
            return None
        self.cap_guards = getattr(self, 'cap_guards', 0) + 1
        return best

    # --- team roles -------------------------------------------------------

    def team_mode(self, game_state):
        """attack (losing or tied) / balanced (winning) / lock (winning clearly or near the end)."""
        lead = self.get_score(game_state)
        if lead <= 0 and game_state.data.timeleft // 4 <= self.DECOY_MOVES:
            return 'decoy'
        if lead < 0 and isinstance(self, PlannerDefender) and self.siege(game_state, self.enemy_states(game_state)):
            return 'decoy'   # SIEGE: they stopped attacking to defend their lead
        if lead <= 0:
            return 'attack'
        if self.POST_LOCK and getattr(self, 'phase', None) == 'post':
            return 'lock'      # after the blitz, with the lead: both defend
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
        # FIX 4: the defenders do not take the bait (siege stalemate: no food
        # eaten by our team for DECOY_STALL turns): the decoy attacks too, so
        # the two marked defenders face two attackers at once.
        progress = (self.get_score(game_state),
                    sum(game_state.get_agent_state(i).num_carrying for i in self.team))
        if progress != getattr(self, 'decoy_progress', None):
            self.decoy_progress, self.decoy_stall = progress, 0
        self.decoy_stall = getattr(self, 'decoy_stall', 0) + 1
        if self.decoy_stall % (self.DECOY_STALL + self.DECOY_PUSH) > self.DECOY_STALL:
            self.decoy_now = False
            self.decoy_pushes = getattr(self, 'decoy_pushes', 0) + 1
            return self.attack(game_state, max(1, 1 - self.get_score(game_state)))
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
        action = self.guard_capsule(game_state)
        if action is not None:
            return action
        if lock and me.scared_timer == 0:
            action = self.lure(game_state, my_pos, enemies)
            if action is not None:
                self.guarding = True  # the lure is a team plan: no tactical override
                return action
        if me.scared_timer == 0:
            inv_t = self.trap_target(game_state)
            if inv_t is not None:
                action = self.trap(game_state, inv_t)
                if action is not None and self.legal(game_state, action):
                    return action
        mark_pos, mark = self.marked_enemy(game_state, enemies, lock)
        mate = self.mate_chaser(game_state)
        if mark.is_pacman:
            targets = [mark_pos]
        elif lock and mate is not None:
            # LOCK: my mark is still outside; the invader belongs to my teammate,
            # unless it comes close to me (no double-teaming one decoy) or it is
            # already carrying enough food to hurt our lead (then both hunt it).
            loaded = [p for p, e in enemies if e.is_pacman and e.num_carrying >= self.LOCK_HELP_CARRY]
            targets = loaded or [p for p in invaders if self.dist(my_pos, p) <= self.HELP_RANGE]
        else:
            targets = invaders
        if targets:
            if me.scared_timer > 0:
                return self.keep_distance(game_state, targets, 2)
            inv = min(targets, key=lambda p: self.dist(my_pos, p))
            inv_carry = max([e.num_carrying for p, e in enemies if p == inv] or [0])
            if self.dist(my_pos, inv) >= self.CONTAIN_DIST and inv_carry >= self.CONTAIN_CARRY:
                action = self.contain(game_state, my_pos, inv, mate)
                if action is not None and self.legal(game_state, action):
                    return action
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

    LOCK_HELP_CARRY = 3   # LOCK: an invader carrying this much is hunted by both defenders
    HELP_RANGE = 3   # a defender leaves its own mark only for an invader this close

    CONTAIN_CARRY = 4  # ... and carrying at least this much food
    CONTAIN_DIST = 3   # an invader at least this far: contain it instead of following it

    def invader_region(self, inv, ghosts):
        """Score of the cells the invader reaches strictly before our ghosts
        (size + food there; huge if it can still get back to its side)."""
        food = set(self.get_food_you_are_defending(self.cur_state).as_list())
        caps = set(self.get_capsules_you_are_defending(self.cur_state))
        seen = {inv: 0}
        frontier = deque([inv])
        score = 0
        while frontier:
            c = frontier.popleft()
            d = seen[c]
            score += 1 + (3 if c in food else 0) + (40 if c in caps else 0)
            if not self.on_my_side(c):
                score += 500
                continue
            for n in self.neighbours(c):
                if n not in seen and all(d + 1 < self.dist(g, n) for g in ghosts):
                    seen[n] = d + 1
                    frontier.append(n)
        return score

    def contain(self, game_state, my_pos, inv, mate):
        """CONTAIN: the move that leaves the invader the smallest region it controls,
        above all cutting it off from its side (a chaser behind it never catches it)."""
        self.cur_state = game_state
        others = [mate] if mate is not None else []
        best, best_v = None, None
        for action in game_state.get_legal_actions(self.index):
            pos = self.cell(game_state.generate_successor(self.index, action).get_agent_position(self.index))
            if not self.on_my_side(pos):
                continue
            if pos == inv:
                return action
            v = self.invader_region(inv, [pos] + others) * 10 + self.dist(pos, inv)
            if best_v is None or v < best_v:
                best, best_v = action, v
        return best

    # --- TRAP: two ghosts corner an invader together -------------------------
    TRAP_RANGE = 10    # both our ghosts within this many cells of the invader
    TRAP_CARRY = 1     # ... carrying at least this much food

    def trap_target(self, game_state):
        """The invader both our ghosts should corner now, or None."""
        enemies = self.enemy_states(game_state)
        invaders = [(p, e) for p, e in enemies if e.is_pacman and e.num_carrying >= self.TRAP_CARRY]
        if not invaders:
            return None
        ours = []
        for i in self.team:
            st = game_state.get_agent_state(i)
            if st.is_pacman or st.scared_timer > 0 or st.num_carrying > 0:
                return None
            ours.append(self.cell(st.get_position()))
        best = None
        for p, e in invaders:
            if all(self.dist(o, p) <= self.TRAP_RANGE for o in ours):
                key = (e.num_carrying, -sum(self.dist(o, p) for o in ours))
                if best is None or key > best[0]:
                    best = (key, p)
        return best and best[1]

    def trap(self, game_state, inv):
        """
        TRAP. A lone ghost chasing a good Pacman in a maze with loops never
        catches it. Here both ghosts choose their next cells TOGETHER: every
        pair of moves is scored by the region the invader still controls (the
        cells it reaches strictly before both ghosts, with a huge cost if that
        region still touches its own side, plus our food and capsule in it),
        then by how close the ghosts get. Both agents run the same search and
        each plays its own half, so they close the exits from both ends.
        """
        self.cur_state = game_state
        mate = [i for i in self.team if i != self.index][0]
        my_pos = self.cell(game_state.get_agent_state(self.index).get_position())
        mate_pos = self.cell(game_state.get_agent_state(mate).get_position())
        mine = []
        for a in game_state.get_legal_actions(self.index):
            dx, dy = Actions.direction_to_vector(a)
            c = (int(my_pos[0] + dx), int(my_pos[1] + dy))
            if self.on_my_side(c):
                if c == inv:
                    return a
                mine.append((a, c))
        theirs = [mate_pos] + [n for n in self.neighbours(mate_pos) if self.on_my_side(n)]
        best, best_v = None, None
        for a, c in mine:
            for m in theirs:
                v = self.invader_region(inv, [c, m]) * 10 + self.dist(c, inv) + self.dist(m, inv)
                if best_v is None or v < best_v or (v == best_v and a != Directions.STOP and best == Directions.STOP):
                    best, best_v = a, v
        if best is not None:
            self.traps = getattr(self, 'traps', 0) + 1
        return best

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
        try:
            phase = self.blitz(game_state)
        except Exception:
            traceback.print_exc()
            phase = None
        self.phase = phase
        action = None
        if phase == 'blitz':
            try:
                action = self.blitz_action(game_state)
            except Exception:
                traceback.print_exc()
        if action is None:
            action = self._base_get_action(game_state)
            try:
                action = self.power_filter(game_state, action)
            except Exception:
                traceback.print_exc()
        try:
            action = self.scared_filter(game_state, action)
        except Exception:
            traceback.print_exc()
        try:
            self.note_kills(action)
        except Exception:
            traceback.print_exc()
        return action

    def scared_filter(self, game_state, action):
        """
        SCARED SAFETY. While we are a scared ghost on our side, an invading
        Pacman eats us (a ~40 move respawn). Never step on or next to a seen
        invader; pick the safe move that stays farthest from it.
        """
        me = game_state.get_agent_state(self.index)
        if me.is_pacman or me.scared_timer <= 0:
            return action
        obs = getattr(self, 'real_obs', None) or game_state
        inv = []
        for i in self.get_opponents(obs):
            st = obs.get_agent_state(i)
            if st.is_pacman and st.get_position() is not None:
                inv.append(self.cell(st.get_position()))
        if not inv:
            return action
        my_pos = self.cell(me.get_position())

        def nxt(a):
            dx, dy = Actions.direction_to_vector(a)
            return (int(my_pos[0] + dx), int(my_pos[1] + dy))

        def gap(a):
            n = nxt(a)
            return min(self.dist(n, p) for p in inv)
        if gap(action) >= 2:
            return action
        legal = game_state.get_legal_actions(self.index)
        safe = [a for a in legal if gap(a) >= 2]
        if not safe:
            return max(legal, key=gap)
        return max(safe, key=lambda a: (gap(a) >= 3, -abs(gap(a) - 3), a != Directions.STOP))

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
        if not me.is_pacman or not self.big_map:
            return action          # small maps: measured better without the power play
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

    # --- CAPSULE BLITZ (team plan) ------------------------------------------------
    # 1. SETUP: both agents go to the enemy side; the capsule is eaten only once
    #    both are there (by the one closer to it).
    # 2. BLITZ: while their ghosts are scared, both agents collect food, the
    #    hardest first (deepest in their side), each its own targets, only food
    #    they can still bring home before the scare ends.
    # 3. HOME: when the scare is about to end, both run home (only the ghost's
    #    own cell is avoided: it may follow one cell behind).
    # 4. DEFEND: after the blitz, with the lead, both agents defend.
    BLITZ_MARGIN = 2        # moves of margin to be home when the scare ends
    BLITZ_FROM = 0          # first move (whole game) at which the setup may start
    BLITZ_MIN_FOOD = 40     # (see BIG_MAP_FOOD)
    BLITZ_MIN_FOOD = 40     # only on big maps: on small ones the setup leaves our side open too long
    BLITZ_ABORT = 4         # an invader carrying this much cancels the setup for our defender
    blitz_state = {}        # shared by the team: team -> dict(tl, phase, caps, targets)

    def blitz(self, game_state):
        """Current phase of the team plan: 'setup', 'blitz', 'post' or None."""
        if not self.big_map:
            return None        # small maps: the plan leaves our side open too long (measured)
        key, tl = tuple(self.team), game_state.data.timeleft
        st = self.blitz_state.get(key)
        if st is None or st['tl'] < tl:
            st = dict(tl=tl, phase=None, caps=len(self.get_capsules(game_state)), targets={})
            self.blitz_state[key] = st
        if st['tl'] == tl and st.get('done_tl') == tl:
            return st['phase']
        st['tl'], st['done_tl'] = tl, tl
        caps = len(self.get_capsules(game_state))
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        scare = max([e.scared_timer for e in enemies] or [0])
        if caps < st['caps'] and scare > 0:
            st['phase'] = 'blitz'                   # we just ate their capsule
        elif st['phase'] == 'blitz' and scare == 0:
            st['phase'] = 'post'
            st['targets'] = {}
        elif st['phase'] in (None, 'setup'):
            st['phase'] = 'setup' if caps > 0 and 1200 - tl >= self.BLITZ_FROM and \
                self.big_map and len(self.get_food(game_state).as_list()) > self.BLITZ_MIN_FOOD else None
        st['caps'] = caps
        return st['phase']

    def blitz_action(self, game_state):
        """BLITZ phase move for either agent (also a ghost still at home)."""
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [(self.cell(e.get_position()), e.scared_timer) for e in enemies
                  if not e.is_pacman and e.get_position() is not None]
        if not ghosts:
            return None
        scare = min(t for _, t in ghosts)
        if scare <= 0:
            return None
        avoid = {p for p, _ in ghosts}                  # never eat a scared ghost (it respawns unscared)
        home = self.distance_home(my_pos) if not self.on_my_side(my_pos) else 0
        st = self.blitz_state[tuple(self.team)]
        if scare <= home + self.BLITZ_MARGIN:
            st['targets'].pop(self.index, None)
            if me.is_pacman:
                return self.power_step(game_state, my_pos, set(self.home_border), avoid)
            return None
        food = self.get_food(game_state).as_list()
        mate_t = [t for i, t in st['targets'].items() if i != self.index]
        feasible = [f for f in food if f not in mate_t
                    and self.dist(my_pos, f) + self.distance_home(f) + self.BLITZ_MARGIN <= scare]
        if not feasible:
            st['targets'].pop(self.index, None)
            if me.is_pacman and me.num_carrying > 0:
                return self.power_step(game_state, my_pos, set(self.home_border), avoid)
            return None
        # hardest first: the deepest food, but not a long detour for it
        target = max(feasible, key=lambda f: (self.distance_home(f) - 0.5 * self.dist(my_pos, f)))
        st['targets'][self.index] = target
        return self.power_step(game_state, my_pos, {target}, avoid)

    def power_step(self, game_state, start, targets, blocked):
        """Legal first move of a shortest path from start to any target avoiding `blocked`, or None."""
        if not targets:
            return None
        action, _ = self.bfs(start, targets, frozenset(blocked) - set(targets))
        if action is None or action == Directions.STOP or not self.legal(game_state, action):
            return None
        return action


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
    OPEN_QUOTA = 60          # their side unguarded: food per trip
    OPEN_MIN_FOOD = 40       # ... only on big maps
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
        if getattr(a, 'phase', None) == 'setup' and not all(
                game_state.get_agent_state(i).is_pacman for i in a.team):
            capsules = []      # BLITZ: the capsule waits until both of us are on their side
        # REMOTE FIRST: the deepest food (far from their border, the hardest to reach
        # later) is preferred; the easy food near the border is left for later.
        food_pool = a.lane_food(game_state, food_pool)
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
        # OPEN SIDE: when no enemy is home as a ghost, nobody guards their food:
        # no trip limit, keep eating until a ghost shows up (big maps: long trips).
        open_side = len(food) > self.OPEN_MIN_FOOD and not any(not e.is_pacman for _, e in enemies)
        if open_side:
            quota = max(quota, min(self.OPEN_QUOTA, len(food) - 2))
        quota = min(quota, carrying + reachable, carrying + (self.OPEN_QUOTA if open_side else self.HORIZON))
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
        # COUNTER-RAID: when every enemy is on our side, nobody guards their food:
        # attacking is worth more than a third chaser.
        if not me.is_pacman and me.scared_timer == 0 and me.num_carrying == 0:
            inv_t = self.trap_target(game_state)
            if inv_t is not None:
                action = self.trap(game_state, inv_t)
                if action is not None and self.legal(game_state, action):
                    return action
        unguarded = all(e.is_pacman for p, e in enemies)
        if not me.is_pacman and me.scared_timer == 0 and not unguarded:
            loaded = [p for p, e in enemies if e.is_pacman and e.num_carrying >= self.HELP_CARRY]
            if loaded:
                mate = self.mate_chaser(game_state)
                action = self.pincer(game_state, my_pos, loaded, mate)
                if action is None:
                    action, _ = self.bfs(my_pos, loaded, allowed=self.on_my_side)
                if self.legal(game_state, action):
                    self.helps = getattr(self, 'helps', 0) + 1
                    return action

        if getattr(self, 'phase', None) == 'setup':
            caps = self.get_capsules(game_state)
            mate = [i for i in self.team if i != self.index][0]
            mst = game_state.get_agent_state(mate)
            mpos = self.cell(mst.get_position())
            if caps and me.is_pacman and mst.is_pacman:
                mine = min(self.dist(my_pos, c) for c in caps)
                theirs = min(self.dist(mpos, c) for c in caps)
                if mine < theirs or (mine == theirs and self.index < mate):
                    action, _ = self.bfs(my_pos, caps, danger)
                    if self.legal(game_state, action) and action != Directions.STOP:
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
            self.no_plan = 0
            return action

        # FIX 2: no safe plan for NOPLAN_RISK turns in a row (a defender blocks every
        # way in, outside the 4-cell loop detector): take the risk mode to get in.
        self.no_plan = getattr(self, 'no_plan', 0) + 1
        if self.no_plan >= self.NOPLAN_RISK and me.num_carrying == 0 and self.risk_turns == 0:
            self.no_plan = 0
            self.risk_turns = self.RISK_TURNS
            self.noplan_risks = getattr(self, 'noplan_risks', 0) + 1
            target, powered = self.planner.solve(game_state, quota)
            if target is not None and target != 'home':
                action, _ = self.bfs(my_pos, [target], self.danger_zone(active, my_pos))
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
    SCARED_ATTACK = 10   # scared for longer than this: attack instead of shadowing invaders
    RAID = {'attack':   dict(safe=4, back=3, quota=6),
            'balanced': dict(safe=6, back=4, quota=4)}

    OPEN_RACE = 60
    SIEGE_QUIET = 30     # behind and no invader for this many moves (per agent): both attack
    SIEGE_QUOTA = 4
    quiet_since = {}     # shared by the team: team -> timeleft when an invader was last seen

    def siege(self, game_state, enemies):
        """
        SIEGE. When we are behind and the enemy has stopped attacking (both its
        agents stay home to defend the lead), a defender at home is wasted:
        both our agents attack, entering at different points.
        """
        key, tl = tuple(self.team), game_state.data.timeleft
        if key not in self.quiet_since or self.quiet_since[key] < tl:
            self.quiet_since[key] = tl          # new game
        if any(e.is_pacman for _, e in enemies):
            self.quiet_since[key] = tl
            return False
        return self.get_score(game_state) < 0 and self.quiet_since[key] - tl >= 4 * self.SIEGE_QUIET
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
        # SCARED: a scared ghost cannot stop invaders; while the scare is long,
        # attack instead (as a Pacman on their side we are not scared).
        if me.scared_timer > self.SCARED_ATTACK:
            return self.attack(game_state, self.RAID['attack']['quota'])
        if invaders and all(e.is_pacman for _, e in enemies) and len(self.get_food(game_state).as_list()) > 40:
            return self.attack(game_state, self.OPEN_RACE)   # RACE: both invade on a big map
        if self.siege(game_state, enemies):
            return self.attack(game_state, self.SIEGE_QUOTA)
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
        if self.capsule_lost(game_state):
            self.cap_lost_moves = getattr(self, 'cap_lost_moves', 0) + 1
            return self.attack(game_state, self.RAID['attack']['quota'])
        action = self.guard_capsule(game_state)
        if action is not None:
            return action
        phase = getattr(self, 'phase', None)
        if phase == 'setup' and self.team_mode(game_state) in ('attack', 'balanced'):
            enemies = self.enemy_states(game_state)
            loaded = any(e.is_pacman and e.num_carrying >= self.BLITZ_ABORT for _, e in enemies)
            if not loaded:
                return self.attack(game_state, self.RAID['attack']['quota'])
        if self.defender_index(game_state) == self.index:
            return PlannerDefender.choose_action(self, game_state)
        return PlannerAttacker.choose_action(self, game_state)


class RoleAttacker(FlexAgent):
    """
    ROLE 1, THE ATTACKER. Always on their side: plans the most food per trip
    with the A* planner, eats their capsule when food is guarded and then
    clears the deepest food while their ghosts are scared (blitz). It only
    stops attacking to defend a clear lead (lock) with nothing to bank.
    """
    def blitz(self, game_state):
        phase = super().blitz(game_state)
        if not self.roles_on(game_state):
            return phase       # big maps: both attack together (FlexAgent + capsule blitz)
        return None if phase == 'setup' else phase   # no waiting for the guard

    def choose_action(self, game_state):
        if not self.roles_on(game_state):
            return FlexAgent.choose_action(self, game_state)
        me = game_state.get_agent_state(self.index)
        if self.team_mode(game_state) == 'lock' and not (me.is_pacman and me.num_carrying > 0):
            return self.defend(game_state, lock=True)
        return self.attack(game_state, self.QUOTA)


class RoleGuard(FlexAgent):
    """
    ROLE 2, THE GUARD. Lives on our side. Invaders may come in, but:
      - none goes back home with food (chase, contain: cut it off from its side,
        trap it together with the attacker when that one is home);
      - none eats our capsule (the guard stays between the invader and it).
    When we are not ahead and the coast is clear, it snipes the single safest
    food on their side (one it can reach and bring back before any ghost can
    get there, by the timed reachability of the A* safety code) and returns.
    """
    SNIPE_MAX = 14      # longest round trip (moves) for a snipe
    SNIPE_LEAD = 0      # snipe only while our lead is at most this

    def blitz(self, game_state):
        phase = super().blitz(game_state)
        if not self.roles_on(game_state):
            return phase       # big maps: both attack together (FlexAgent + capsule blitz)
        return None if phase == 'setup' else phase

    def snipe_target(self, game_state, my_pos):
        eta = self.ghost_eta(game_state)
        food = self.get_food(game_state).as_list()
        if not eta:
            times = self.bfs_distances(my_pos)
            cands = [(times.get(f, INF) + self.home_dist.get(f, INF), f) for f in food]
        else:
            times = self.timed_reach(my_pos, eta)
            cands = [(times[f] + self.home_dist[f], f) for f in food
                     if f in times and self.can_escape(f, times[f], eta)]
        cands = [c for c in cands if c[0] <= self.SNIPE_MAX]
        return min(cands)[1] if cands else None

    # FIX 3: behind, and the enemy camps at home (no invader for QUIET_JOIN
    # turns in a row): guarding an empty side is useless, join the attack.
    QUIET_JOIN = 20

    def join_attack(self, game_state):
        enemies = self.enemy_states(game_state)
        me = game_state.get_agent_state(self.index)
        if any(e.is_pacman for _, e in enemies):
            self.quiet = 0
            return None
        self.quiet = getattr(self, 'quiet', 0) + 1
        if self.quiet >= self.QUIET_JOIN and self.get_score(game_state) < 0 and me.scared_timer == 0:
            self.joins = getattr(self, 'joins', 0) + 1
            return self.attack(game_state, self.QUOTA)
        return None

    def choose_action(self, game_state):
        if not self.roles_on(game_state):
            return FlexAgent.choose_action(self, game_state)
        action = self.join_attack(game_state)
        if action is not None:
            return action
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        enemies = self.enemy_states(game_state)
        if not enemies:
            return Directions.STOP
        active = [p for p, e in enemies if not e.is_pacman and e.scared_timer <= 2]
        danger = self.danger_zone(active, my_pos)
        invaders = [p for p, e in enemies if e.is_pacman]
        if me.is_pacman:
            target = getattr(self, 'snipe', None)
            food = set(self.get_food(game_state).as_list())
            if me.num_carrying == 0 and not invaders and target in food:
                action, _ = self.bfs(my_pos, [target], danger)
                if self.legal(game_state, action) and action != Directions.STOP:
                    return action
            self.snipe = None
            action, _ = self.bfs(my_pos, self.home_border, danger)
            if self.legal(game_state, action) and action != Directions.STOP:
                return action
            return self.flee(game_state, active)
        if me.scared_timer > self.SCARED_ATTACK and not invaders:
            return self.attack(game_state, 1)
        # FIX 5: our capsule is eaten (we are scared) and both enemies are on our
        # side looting: a scared guard cannot stop them, and their side is empty.
        # Loot it too instead of watching.
        if me.scared_timer > self.SCARED_ATTACK and len(invaders) == len(enemies):
            self.counter_loots = getattr(self, 'counter_loots', 0) + 1
            return self.attack(game_state, self.QUOTA)
        if not invaders and self.get_score(game_state) <= self.SNIPE_LEAD:
            target = self.snipe_target(game_state, my_pos)
            if target is not None:
                self.snipe = target
                self.snipes = getattr(self, 'snipes', 0) + 1
                action, _ = self.bfs(my_pos, [target], danger)
                if self.legal(game_state, action) and action != Directions.STOP:
                    return action
        return self.defend(game_state)


def capsule_race(agent, game_state):
    """CAPSULE RACE only on small maps with exactly one capsule per side."""
    return (not agent.big_map and len(agent.get_capsules(game_state)) == 1
            and len(agent.get_capsules_you_are_defending(game_state)) == 1)


class CapsuleRusher(RoleAttacker):
    """
    CAPSULE RACE, ATTACKER. From the first move it takes the fastest safe path
    to their capsule (avoiding cells next to an active ghost) and eats it as
    soon as possible. While their ghosts are scared it clears the food it can
    still bring home in time (deepest first) and banks before the scare ends.
    Afterwards it plays the normal attacker.
    """
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.race = capsule_race(self, game_state)

    def roles_on(self, game_state):
        return True if self.race else RoleAttacker.roles_on(self, game_state)

    def blitz(self, game_state):
        return None if self.race else RoleAttacker.blitz(self, game_state)

    def choose_action(self, game_state):
        if not self.race:
            return RoleAttacker.choose_action(self, game_state)
        me = game_state.get_agent_state(self.index)
        my_pos = self.cell(me.get_position())
        key = tuple(self.team)
        st = self.blitz_state.get(key)
        if st is None or st['tl'] < game_state.data.timeleft:
            st = dict(tl=game_state.data.timeleft, phase=None, caps=0, targets={})
            self.blitz_state[key] = st
        st['tl'] = game_state.data.timeleft
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        if any(not e.is_pacman and e.scared_timer > 0 for e in enemies):
            action = self.blitz_action(game_state)
            if action is not None:
                return action
        caps = self.get_capsules(game_state) if getattr(self, 'RUSH', True) else []
        if caps and me.num_carrying <= self.RUSH_CARRY:
            active = [p for p, e in self.enemy_states(game_state) if not e.is_pacman and e.scared_timer <= 2]
            danger = self.danger_zone(active, my_pos)
            action, _ = self.bfs(my_pos, caps, danger)
            if self.legal(game_state, action) and action != Directions.STOP:
                self.rush_moves = getattr(self, 'rush_moves', 0) + 1
                return action
        return RoleAttacker.choose_action(self, game_state)

    RUSH_CARRY = 3   # carrying more than this: bank first (the planner decides)


class CapsuleKeeper(RoleGuard):
    """
    CAPSULE RACE, DEFENDER. Parks next to our capsule (on the side the enemy
    comes from, never on it). When an invader is on our side, it always stays
    closer to the capsule than the invader and moves in on it, so the invader
    can never eat the capsule. Without capsules left it is the normal guard.
    """
    CAP_GUARD = 99
    SNIPE_LEAD = -99

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.race = capsule_race(self, game_state)
        if not self.race:
            self.CAP_GUARD, self.SNIPE_LEAD = RoleGuard.CAP_GUARD, RoleGuard.SNIPE_LEAD

    def roles_on(self, game_state):
        return True if self.race else RoleGuard.roles_on(self, game_state)

    def blitz(self, game_state):
        return None if self.race else RoleGuard.blitz(self, game_state)

    KEEP_RADIUS = 2
    KEEP_CHASE_CARRY = 2

    def reach_cells(self, cap, r):
        return [c for c in self.home_dist if self.dist(c, cap) <= r]

    def park_cell(self, cap):
        near = [c for c in self.neighbours(cap) if self.on_my_side(c)] or [cap]
        return min(near, key=lambda c: (self.distance_home(c), c))

    def choose_action(self, game_state):
        if not self.race:
            return RoleGuard.choose_action(self, game_state)
        action = self.join_attack(game_state)
        if action is not None:
            return action
        caps = self.get_capsules_you_are_defending(game_state)
        me = game_state.get_agent_state(self.index)
        if not caps or me.is_pacman or me.scared_timer > 0:
            return RoleGuard.choose_action(self, game_state)
        my_pos = self.cell(me.get_position())
        enemies = self.enemy_states(game_state)
        invaders = [p for p, e in enemies if e.is_pacman]
        cap = min(caps, key=lambda c: min((self.dist(p, c) for p, _ in enemies), default=0))
        # an invader next to us: eat it
        for a in game_state.get_legal_actions(self.index):
            dx, dy = Actions.direction_to_vector(a)
            c = (int(my_pos[0] + dx), int(my_pos[1] + dy))
            if c in invaders and self.on_my_side(c):
                return a
        # FIX 1: a lone invader that is already eating (carrying food) while the
        # other enemy stays home is not a capsule runner: defend normally.
        loaded = [e for _, e in enemies if e.is_pacman and e.num_carrying >= self.KEEP_CHASE_CARRY]
        if len(invaders) == 1 and loaded:
            self.keep_fix1 = getattr(self, 'keep_fix1', 0) + 1
            return self.defend(game_state)
        ring = [c for c in self.reach_cells(cap, self.KEEP_RADIUS) if c != cap and self.on_my_side(c)]
        if invaders and ring:
            inv = min(invaders, key=lambda p: self.dist(p, cap))
            post = min(ring, key=lambda c: (self.dist(c, inv), self.dist(c, cap)))
        else:
            post = self.park_cell(cap)
        if my_pos == post:
            return Directions.STOP
        action, _ = self.bfs(my_pos, [post], allowed=self.on_my_side)   # walking over our capsule is safe
        if self.legal(game_state, action):
            self.guarding = True
            return action
        return self.defend(game_state)


class _AdaptiveMixin:
    """
    ADAPTIVE LOOP. Watch the enemy for OBS moves, classify it, play the plan
    that answers it for HOLD moves, watch again, and so on.
      - attacking: on average at least AGGR_PACMEN enemies on our side during
        the window (two-agent raids, like the friend's agent) -> capsule plan:
        the keeper locks our capsule (small maps, one capsule per side)
      - defensive / normal -> apex.
    Only public information is used (is_pacman of each enemy).
    """
    OBS = 50
    FIRST = 100        # first decision (moves 0-50 are the walk to the border: no signal)
    HOLD = 150
    AGGR_PACMEN = 1.2
    START_AGGR = True
    adapt_state = {}   # shared by the team

    def _adapt(self, game_state):
        key, tl = tuple(self.team), game_state.data.timeleft
        st = self.adapt_state.get(key)
        if st is None or st['tl'] < tl:
            st = dict(tl=tl, seen={}, aggr=self.START_AGGR, next=self.FIRST, log=[])
            self.adapt_state[key] = st
        st['tl'] = tl
        mv = (1200 - tl) // 4
        if tl not in st['seen']:
            st['seen'][tl] = sum(1 for o in self.get_opponents(game_state)
                                 if game_state.get_agent_state(o).is_pacman)
            if mv >= st['next']:
                window = [n for t, n in st['seen'].items() if (1200 - t) // 4 > mv - self.OBS]
                level = sum(window) / max(1, len(window))
                st['aggr'] = level >= self.AGGR_PACMEN
                st['log'].append((mv, round(level, 2), st['aggr']))
                st['next'] = mv + self.HOLD + self.OBS if False else mv + self.HOLD
        return st['aggr']

    @property
    def race(self):
        gs = getattr(self, '_gs', None)
        return bool(getattr(self, '_race_static', False) and gs is not None and self._adapt(gs))

    @race.setter
    def race(self, value):
        self._race_static = value

    def get_action(self, game_state):
        self._gs = game_state
        return super().get_action(game_state)


class AdaptiveRusher(_AdaptiveMixin, CapsuleRusher):
    RUSH = False


class AdaptiveKeeper(_AdaptiveMixin, CapsuleKeeper):
    @property
    def CAP_GUARD(self):
        return CapsuleKeeper.CAP_GUARD if self.race else RoleGuard.CAP_GUARD

    @CAP_GUARD.setter
    def CAP_GUARD(self, v):
        pass

    @property
    def SNIPE_LEAD(self):
        return CapsuleKeeper.SNIPE_LEAD if self.race else RoleGuard.SNIPE_LEAD

    @SNIPE_LEAD.setter
    def SNIPE_LEAD(self, v):
        pass


# =============================================================================
# RL PLAN SELECTOR  (deep reinforcement learning on top of frankenstein)
# =============================================================================
# The moves are still played by the Oreo planners above. A small neural
# network (actor-critic MLP, 2 hidden layers of 64 tanh units) only decides
# the TEAM PLAN, every DECIDE_EVERY (25) moves per agent (~12 times a game):
#   apex       FlexAgent roles, capsule blitz on big maps, attacker + guard
#              when both enemies invade together (frankenstein PLAN='apex')
#   capsule    capsule lock: the keeper holds the ring around our capsule
#              (only offered on small maps with one capsule per side)
#   two_roles  fixed attacker + guard from now on
#   attack     like apex, but never slow down (no balanced / lock mode) while
#              ahead: keep scoring
#   lock       both agents defend (an agent carrying food banks it first)
# Input (23 numbers, public information only): score, time left, food left on
# each side, food carried by us and by them, how often the enemies invade
# (last window and whole game), our scared timer and theirs, capsules, map
# size, score change in the last window and the plan in use.
# Training: PPO (clipped policy gradient, GAE) on thousands of games against
# the other Oreo plans, the class agents and older teams, on default, RANDOM
# and other layouts. Reward = change of the score during each plan
# (food delivered minus food conceded) + 1 for a win / -1 for a loss at the end.
# Inference is plain Python (no numpy): the contest only installs frozendict.
# One decision costs ~0.1 ms.
# RESULTS (weights below = iteration 33 of 50, ~4300 training games; the
# reward stopped improving after iteration ~23):
#   The trained network almost always picks 'attack' (greedy play gives the
#   same games as RL_FIXED=attack). 288 games, fresh seeds 201-208, default +
#   RANDOM, both colours, vs frankenstein, apex, capsule_race, two_roles,
#   a_star_pro, hunter, fortress, Musya/fixed, Bimo/iw1:
#     frankenstein (adaptive)  170-34-84  64.9% points  avg +2.84
#     rl_selector (this file)  175-40-73  67.7% points  avg +4.73
#     RL_FIXED=apex            171-45-72  67.2% points  avg +3.14
#   So the gain comes from never slowing down while ahead and from not using
#   frankenstein's automatic capsule switch, not from switching plans by
#   opponent. Not tested against the friend's compiled agent (Windows only).
# Env vars (testing only): RL_FIXED=<plan> plays one plan, RL_WEIGHTS=<json>
# loads other weights, RL_MODE=sample samples plans (training).
# =============================================================================

import json as _json

RL_OPTIONS = ('apex', 'capsule', 'two_roles', 'attack', 'lock')
RL_DECIDE_EVERY = 25
RL_FEATURES = 23
RL_WEIGHTS = None   # set at the end of the file
RL_SEL = {}         # shared by the two agents of a team: id(team_key) -> state


def _rl_forward(w, x):
    """MLP forward pass in plain Python: returns (logits, value)."""
    h = x
    for W, b in ((w['W1'], w['b1']), (w['W2'], w['b2'])):
        h = [math.tanh(bj + sum(wij * hi for wij, hi in zip(col, h))) for col, bj in zip(W, b)]
    logits = [bj + sum(wij * hi for wij, hi in zip(col, h)) for col, bj in zip(w['Wp'], w['bp'])]
    value = w['bv'][0] + sum(wij * hi for wij, hi in zip(w['Wv'][0], h))
    return logits, value


def _rl_load_weights():
    path = os.environ.get('RL_WEIGHTS')
    if path:
        try:
            with open(path) as f:
                return _json.load(f)
        except Exception:
            traceback.print_exc()
    return RL_WEIGHTS


class _RLMixin:
    def _rl_state(self, game_state):
        key = id(self.team_key)
        st = RL_SEL.get(key)
        if st is None:
            st = dict(tl0=game_state.data.timeleft, seen={}, opt=None, next=0, traj=[], last_score=0,
                      weights=_rl_load_weights(), sample=os.environ.get('RL_MODE') == 'sample',
                      fixed=os.environ.get('RL_FIXED'), init_food=None)
            RL_SEL[key] = st
        self._rl = st
        return st

    def _rl_option(self, game_state):
        st = self._rl_state(game_state)
        tl = game_state.data.timeleft
        if tl not in st['seen']:
            opp = [game_state.get_agent_state(o) for o in self.get_opponents(game_state)]
            mine = [game_state.get_agent_state(i) for i in self.team]
            st['seen'][tl] = (sum(e.is_pacman for e in opp), sum(m.is_pacman for m in mine))
        mv = (st['tl0'] - tl) // 4
        if st['opt'] is None or mv >= st['next']:
            try:
                self._rl_decide(game_state, st, mv)
            except Exception:
                traceback.print_exc()
                st['opt'] = 'apex'
            st['next'] = mv + RL_DECIDE_EVERY
        return st['opt']

    def _rl_features(self, game_state, st, mv):
        tl = game_state.data.timeleft
        score = self.get_score(game_state)
        food_att = len(self.get_food(game_state).as_list())
        food_def = len(self.get_food_you_are_defending(game_state).as_list())
        if st['init_food'] is None:
            st['init_food'] = (max(1, food_att), max(1, food_def))
        opp = [game_state.get_agent_state(o) for o in self.get_opponents(game_state)]
        mine = [game_state.get_agent_state(i) for i in self.team]
        seen = st['seen']
        window = [v for t, v in seen.items() if (st['tl0'] - t) // 4 > mv - RL_DECIDE_EVERY] or [(0, 0)]
        allv = list(seen.values()) or [(0, 0)]
        prev = st['opt']
        race_ok = bool(getattr(self, '_race_static', False))
        x = [
            max(-3.0, min(3.0, score / 10.0)),
            tl / float(max(1, st['tl0'])),
            food_def / float(st['init_food'][1]),
            food_att / float(st['init_food'][0]),
            sum(m.num_carrying for m in mine) / 10.0,
            sum(e.num_carrying for e in opp) / 10.0,
            sum(v[0] for v in window) / (2.0 * len(window)),
            sum(v[0] for v in allv) / (2.0 * len(allv)),
            sum(1 for v in allv if v[0] == 2) / float(len(allv)),
            sum(v[1] for v in window) / (2.0 * len(window)),
            max(m.scared_timer for m in mine) / 40.0,
            max(e.scared_timer for e in opp) / 40.0,
            len(self.get_capsules_you_are_defending(game_state)) / 2.0,
            len(self.get_capsules(game_state)) / 2.0,
            1.0 if self.big_map else 0.0,
            1.0 if race_ok else 0.0,
            max(-3.0, min(3.0, (score - st['last_score']) / 10.0)),
        ] + [1.0 if prev == o else 0.0 for o in RL_OPTIONS] + [st['init_food'][0] / 60.0]
        mask = [1, 1 if race_ok else 0, 1, 1, 1]
        return x, mask, score

    def _rl_decide(self, game_state, st, mv):
        x, mask, score = self._rl_features(game_state, st, mv)
        w = st['weights']
        if st['fixed']:
            st['opt'] = st['fixed'] if mask[RL_OPTIONS.index(st['fixed'])] else 'apex'
            return
        if w is None:
            st['opt'] = 'apex'
            return
        logits, value = _rl_forward(w, x)
        z = [l if m else -1e9 for l, m in zip(logits, mask)]
        mx = max(z)
        ex = [math.exp(v - mx) if m else 0.0 for v, m in zip(z, mask)]
        tot = sum(ex)
        probs = [e / tot for e in ex]
        if st['sample']:
            r, a = random.random(), 0
            acc = 0.0
            for a, p in enumerate(probs):
                acc += p
                if r < acc and mask[a]:
                    break
        else:
            a = max(range(len(probs)), key=lambda i: probs[i])
        st['traj'].append(dict(x=x, mask=mask, a=a, logp=math.log(max(probs[a], 1e-12)), v=value,
                               score=score, mv=mv))
        st['opt'] = RL_OPTIONS[a]
        st['last_score'] = score

    # --- how the chosen plan drives the Oreo agents ---------------------------------
    def _adapt(self, game_state):           # capsule lock on/off (used by the race property)
        return self._rl_option(game_state) == 'capsule'

    def roles_on(self, game_state):
        opt = self._rl_option(game_state)
        if opt == 'two_roles':
            return True
        return super().roles_on(game_state)

    def team_mode(self, game_state):
        mode = super().team_mode(game_state)
        opt = self._rl_option(game_state)
        if opt == 'attack' and mode in ('balanced', 'lock'):
            return 'attack'
        if opt == 'lock' and mode != 'decoy':
            return 'lock'
        return mode

    def get_action(self, game_state):
        self._gs = game_state
        try:
            self._rl_option(game_state)
        except Exception:
            traceback.print_exc()
        return super().get_action(game_state)


class RLRusher(_RLMixin, AdaptiveRusher):
    pass


class RLKeeper(_RLMixin, AdaptiveKeeper):
    pass


def create_team(first_index, second_index, is_red,
                first='RLRusher', second='RLKeeper', num_training=0, **kwargs):
    PlannerAgentBase.ROLES_AFTER = PLAN_DEFAULTS['ROLES_AFTER']
    agents = {'RLRusher': RLRusher, 'RLKeeper': RLKeeper}
    team = [agents[first](first_index), agents[second](second_index)]
    key = object()   # one shared tracker and plan selector for this team in this game
    for a in team:
        a.team_key = key
    return team


# Trained weights (PPO, iteration 33).
RL_WEIGHTS = {"W1":[[-0.00515,-0.05058,0.117,0.07136,-0.14782,0.15383,0.29389,0.21151,-0.13643,-0.26716,-0.12523,-0.02743,-0.4842,-0.06438,-0.31754,-0.1492,-0.08581,-0.05489,0.11703,0.23286,-0.02744,0.25833,-0.15688],[0.16785,0.22962,0.0631,-0.17374,-0.11581,-0.14389,0.04363,-0.20674,-0.05475,0.02713,0.10519,0.13262,0.11457,-0.10057,0.01914,0.16647,0.30598,-0.24213,0.2655,0.28336,0.19361,0.10816,-0.04845],[0.35289,0.44202,0.37907,0.2059,0.15829,-0.30299,-0.00102,0.11553,-0.2662,0.08335,0.0798,0.24343,-0.2779,-0.16314,-0.094,-0.25743,0.29848,-0.11371,0.02709,-0.07438,0.31241,0.30031,0.08714],[-0.58062,0.02086,0.10399,0.27393,-0.2727,0.46375,-0.21884,-0.09582,0.22788,-0.06247,0.48808,-0.03894,-0.13349,-0.08656,-0.28259,-0.28992,0.08858,0.14787,0.2779,-0.11656,0.28595,-0.08771,0.29937],[-0.06632,-0.16217,0.05807,0.15291,0.12112,-0.1771,-0.31244,-0.30404,0.08478,0.24784,-0.07875,-0.2007,0.1638,-0.265,-0.07885,0.13905,-0.42217,0.08704,-0.13505,0.01004,-0.02685,0.05297,0.17937],[-0.05731,0.33224,0.20179,0.14684,0.30875,0.11963,0.18756,0.03616,-0.29782,0.01495,-0.15729,-0.20626,0.07773,-0.09806,-0.18714,-0.21418,0.06287,0.09256,0.23962,-0.01463,0.26399,0.33811,0.24958],[-0.42994,0.28048,0.08226,0.0414,0.10504,0.00661,0.001,-0.15928,-0.46466,-0.0451,-0.18666,0.23139,-0.08464,0.00605,-0.10618,-0.09809,0.01103,-0.33439,0.06144,-0.06348,-0.25613,-0.49562,0.10985],[-0.05633,-0.07646,-0.01219,0.36836,0.04799,-0.05399,-0.32965,0.35611,0.17602,0.25821,0.00823,0.2682,0.08163,0.13378,-0.02267,-0.31082,0.16476,-0.38791,-0.08161,-0.04,-0.18692,0.16639,-0.04458],[-0.08883,0.14055,-0.0703,0.27755,0.13721,-0.07583,-0.37152,-0.2709,0.2353,-0.01531,-0.04023,0.30178,-0.30466,-0.11171,-0.03146,0.23158,-0.13055,-0.14045,-0.29948,0.13635,0.17256,-0.07448,0.06154],[-0.23221,-0.10214,0.29049,-0.02699,0.49972,-0.26428,0.10105,-0.05157,0.08267,-0.00298,-0.14114,-0.14699,0.59982,-0.02481,-0.38716,-0.12975,0.10028,-0.11703,0.2492,0.17289,-0.03407,-0.05545,-0.22994],[-0.17184,-0.31876,0.23814,0.36277,-0.34605,-0.26999,-0.37029,-0.17662,-0.63558,-0.25009,0.27959,-0.14923,0.18071,-0.13387,0.34377,0.0209,-0.07055,0.54448,-0.03832,-0.1888,0.01845,-0.04834,0.20781],[-0.15749,0.19452,0.1858,-0.17649,0.13645,-0.16504,0.48972,-0.14226,-0.07206,-0.18392,-0.09044,0.06448,0.14257,-0.12564,0.01689,-0.27546,-0.15213,0.58184,0.18819,-0.15639,-0.30277,-0.18215,0.01852],[-0.11871,-0.13778,-0.26712,0.28942,0.19699,-0.13161,-0.11399,-0.11504,-0.59977,0.05966,-0.30047,-0.1606,-0.20115,0.11518,-0.25034,-0.3135,0.06488,0.15306,-0.22126,0.10469,-0.06925,0.07536,-0.26151],[0.17518,0.22357,0.12814,0.17483,-0.8437,0.17213,0.09572,0.07792,-0.07242,0.03305,0.15374,-0.08656,-0.03621,0.29036,-0.26395,0.17792,0.0543,-0.11024,-0.04167,-0.13396,0.15105,0.07091,-0.08904],[-0.18838,0.11499,0.24859,-0.0338,0.13992,-0.11735,0.02653,-0.04444,0.0605,-0.29115,0.13015,0.00873,0.09013,-0.0415,0.10237,-0.20386,0.21751,-0.32403,-0.24327,0.05996,0.33659,0.08996,-0.03759],[-0.26571,-0.07125,-0.03171,0.26209,0.25771,-0.32174,0.36532,-0.17449,-0.21375,0.35927,-0.05527,-0.05014,0.00663,0.14631,0.27787,-0.28801,0.42684,0.15561,-0.10181,-0.20689,-0.23593,0.01849,-0.10573],[-0.24714,0.15213,0.06295,-0.06997,0.16466,0.29722,-0.26286,-0.1861,0.14927,0.12126,-0.01687,0.17238,-0.2423,-0.30749,-0.16825,0.0394,-0.13776,-0.14059,-0.17109,-0.14675,-0.20161,0.03115,0.17118],[-0.11207,-0.04999,-0.12112,0.17318,-0.06766,0.36645,-0.19234,0.10543,0.10043,-0.09642,0.1637,-0.35758,0.42907,-0.31015,0.10684,-0.22389,0.2307,-0.05781,0.06645,0.0405,0.22026,-0.08702,-0.64524],[-0.16462,0.00041,-0.12879,0.18046,0.0849,0.00621,-0.32021,0.25058,0.18165,-0.13862,0.44039,-0.15397,-0.25802,-0.07328,0.12392,-0.20862,0.441,-0.18136,0.21632,0.1107,-0.04884,0.18227,-0.38515],[0.35635,0.03342,-0.08699,0.12632,0.27259,0.15756,0.45146,0.21964,0.27125,-0.06979,0.0754,0.23454,0.04601,0.10238,0.18012,0.14667,-0.03275,-0.0768,-0.21539,-0.17343,0.26697,0.01868,0.20173],[-0.35402,-0.44755,-0.26157,0.27003,0.16499,0.1437,-0.14656,-0.01426,-0.03955,-0.12885,-0.52263,-0.28117,-0.05638,0.29484,0.00701,0.2782,-0.07301,-0.04986,-0.74157,0.09867,0.07974,0.32018,-0.09389],[-0.01793,-0.20167,-0.27864,-0.13032,-0.15976,0.29608,-7e-05,-0.18078,0.02173,0.04072,-0.10746,0.1688,-0.37578,-0.05317,0.08579,-0.31217,0.08424,0.27141,0.11721,-0.33364,-0.1561,0.21691,0.04327],[0.03427,0.09992,0.17245,-0.12823,-0.09103,0.0688,-0.04683,-0.03233,0.27369,-0.21266,0.4564,0.35676,-0.32601,0.00154,0.10974,-0.10694,-0.1252,-0.02918,0.19926,-0.06611,0.37503,0.04071,-0.00078],[0.30717,0.16054,0.08292,-0.11652,0.42189,0.13937,-0.0218,-0.33053,0.09725,-0.22459,-0.32896,0.02172,0.02955,0.27504,0.12617,0.19186,-0.2879,-0.0315,0.00787,0.13799,0.0303,0.19057,-0.08735],[0.01865,0.07722,-0.28298,0.07786,-0.14966,-0.30417,0.23757,-0.0838,-0.17113,-0.13599,0.04745,0.23658,-0.04324,0.0941,0.29621,0.12754,0.23953,-0.08753,0.19267,0.00995,0.18912,-0.23414,-0.15831],[0.23243,-0.03463,-0.07515,0.07098,-0.20643,0.30046,-0.23374,-0.01046,0.0913,-0.00177,-0.11805,-0.19983,0.09105,-0.23278,0.05534,-0.35047,-0.12153,0.02486,-0.2473,0.1518,-0.00551,-0.24065,-0.34087],[-0.38592,-0.06529,-0.29273,-0.2723,-0.13122,0.53009,0.06692,0.15212,0.03216,0.11144,-0.2833,-0.19126,0.04201,-0.02377,-0.10006,-0.12866,-0.03148,-0.16879,-0.44181,-0.24953,0.06966,0.27685,0.35049],[-0.00136,0.13747,0.1474,-0.13351,-0.44238,0.06839,-0.34772,-0.06452,0.14635,-0.34704,0.01746,0.18227,0.07008,0.086,0.15943,0.34653,0.05587,0.25002,0.01667,-0.1041,0.03433,-0.16956,-0.12206],[-0.15189,-0.52543,-0.01092,-0.22008,-0.11488,0.33974,-0.1039,-0.10765,0.28405,0.08791,0.21662,-0.15932,0.16481,-0.14762,-0.15999,0.09509,-0.1,0.16546,0.51694,-0.32772,-0.16918,0.19984,-0.01688],[0.25319,-0.20904,0.0847,-0.04555,0.05624,0.03921,-0.25707,-0.14552,0.33532,0.1176,0.0492,0.03729,0.08212,-0.23592,-0.27017,0.26202,-0.01684,0.17719,-0.11705,0.27005,0.10709,0.13933,0.08554],[-0.18004,-0.3977,-0.22077,0.40292,0.24043,-0.00952,-0.04967,0.22597,-0.04417,-0.28651,0.25384,0.03187,-0.47631,-0.05095,0.02983,0.08953,0.08767,-0.10723,-0.00576,0.09772,-0.04225,-0.10917,0.29256],[-0.25868,-0.10594,-0.45042,0.15748,0.12998,0.18253,0.20594,0.10431,0.08343,0.07167,-0.05278,0.16839,-0.06219,-0.29617,0.1556,0.37771,-0.17029,-0.02204,-0.11215,-0.0652,-0.00112,-0.29043,-0.04861],[-0.36302,-0.1393,-0.25564,-0.16051,-0.38661,0.32879,0.13996,-0.36503,-0.09654,-0.13429,-0.11012,-0.34744,0.18269,-0.07076,0.07586,0.12157,0.30826,-0.35001,0.39285,0.29445,0.13056,0.45655,0.06879],[0.20189,-0.15485,0.22814,0.00024,-0.075,0.37438,-0.14635,-0.04565,-0.10268,-0.14684,0.0137,0.19112,0.06666,-0.51255,0.14053,-0.16571,-0.29675,0.13699,0.07224,-0.25579,0.5088,-0.23751,-0.39917],[-0.20238,0.32688,-0.03094,-0.13496,0.12276,-0.14995,-0.19511,-0.19138,0.14279,0.23325,-0.25864,0.10638,0.14636,-0.22902,-0.03269,-0.23177,-0.26001,-0.01216,-0.20382,0.10445,-0.01224,0.1145,-0.04084],[-0.09664,0.01255,0.01212,-0.15155,0.1003,0.16292,0.04322,0.31814,-0.1893,0.09199,-0.09733,0.04426,-0.14572,0.2674,0.14343,0.27179,-0.20488,-0.12461,-0.17561,-0.13169,-0.11902,-0.02263,0.03835],[0.09499,-0.1195,0.05515,0.24046,-0.26206,0.05686,0.03711,0.15597,-0.00593,0.0013,-0.14768,-0.08742,-0.13326,0.13115,-0.07601,-0.01369,-0.12172,0.19455,0.15587,0.25969,0.46205,-0.0271,0.25085],[-0.09377,0.13638,0.09296,0.04468,0.22685,-0.06963,0.00291,0.08642,0.3786,-0.1666,-0.39074,-0.03768,-0.00448,0.04118,0.08719,0.27833,0.15511,-0.01885,-0.11407,-0.09711,-0.07066,-0.02091,-0.29946],[-0.11306,-0.09308,-0.02954,-0.0483,0.11604,-0.12818,-0.07304,0.09846,-0.07183,-0.00038,0.15221,0.03129,0.06747,-0.00155,0.24107,0.10463,-0.04757,-0.03919,-0.2216,-0.18504,-0.39172,-0.16552,0.30962],[-0.13021,0.15208,-0.02711,-0.23077,0.31746,0.01907,-0.43865,0.14262,-0.27318,-0.04568,0.11147,-0.01655,0.17732,0.12437,-0.31856,-0.10717,0.0115,-0.15682,-0.08086,0.11228,0.04041,-0.18272,-0.00907],[-0.13167,-0.31169,-0.26756,0.09091,-0.10496,0.07017,-0.21244,-0.00823,-0.2002,-0.22915,-0.01991,-0.38714,0.13194,0.50211,0.05104,-0.19351,-0.02063,-0.30895,-0.02494,0.12912,0.09245,-0.26106,0.05382],[0.30001,0.379,-0.10812,-0.22544,0.15186,-0.71197,-0.1437,-0.0895,-0.13578,-0.24634,-0.07344,-0.06358,0.15653,-0.19006,0.00096,-0.00944,0.13438,-0.51385,-0.06605,0.11816,-0.04304,-0.28438,0.08814],[0.05618,-0.00993,-0.2143,-0.18211,0.03307,-0.22648,-0.41963,-0.17809,-0.00434,-0.07876,-0.01564,0.12153,0.1531,0.45768,0.21561,-0.25338,-0.57392,-0.01348,0.00298,-0.23599,0.03635,-0.13481,0.10857],[-0.13902,0.09907,-0.37238,-0.11733,0.05039,-0.33743,0.33698,-0.19588,0.25404,-0.15153,-0.12617,0.25393,0.01996,0.03574,0.15752,-0.23644,-0.09103,-0.21761,0.06724,-0.11331,-0.06972,-0.03858,0.02803],[-0.10506,-0.15295,-0.22778,-0.09733,0.13165,0.02559,0.14619,0.35441,-0.22877,0.04836,-0.13983,-0.1576,0.12465,0.38223,-0.12269,-0.03491,0.21335,-0.3085,0.07083,-0.04054,-0.17329,0.02544,0.13715],[-0.15347,0.10562,0.04867,-0.30509,0.40244,0.18705,-0.19359,-0.14507,0.14014,-0.08257,-0.39835,0.25889,-0.01695,0.40357,0.12354,0.04346,0.57654,-0.03948,-0.22007,0.03597,0.23676,-0.20195,-0.19129],[0.05749,-0.70643,-0.21773,0.22823,-0.18834,-0.33403,0.39123,-0.26568,-0.11855,-0.06896,0.00236,-0.17753,0.02338,0.02923,-0.05136,0.0098,-0.45037,0.13788,-0.25258,-0.41973,0.29784,-0.27522,0.06211],[-0.11192,-0.16478,0.09431,-0.29094,0.15131,0.07379,0.24981,-0.08216,0.21152,0.39472,0.30749,0.18167,0.09955,0.61269,0.59177,-0.16086,0.23463,0.10808,-0.01878,0.02664,0.10064,0.22545,0.12609],[0.16238,0.33988,0.1143,0.10434,0.25808,0.31853,0.29592,0.1208,0.05469,-0.20318,0.077,-0.10936,-0.18047,0.15405,0.00408,0.11765,-0.11476,0.26917,0.00116,0.25596,0.18151,0.07607,-0.09186],[-0.00912,0.21775,0.1738,-0.27955,0.27546,-0.15479,0.11353,-0.18446,0.09598,-0.14277,-0.01241,0.02706,-0.06771,-0.10808,0.00044,-0.34001,0.14057,0.18809,-0.11355,-0.29141,-0.12445,0.22968,0.11729],[-0.05586,0.1883,-0.23966,-0.12168,0.03546,0.05693,-0.07061,-0.52727,0.17489,0.03027,-0.29695,0.07351,0.0148,0.05774,0.18486,-0.15335,0.14436,0.11732,0.17153,0.37864,0.13183,-0.17495,0.35048],[-0.30793,-0.13299,-0.23178,0.02451,0.15627,0.35043,0.13289,-0.10052,-0.08909,-0.17303,0.22904,-0.2674,0.19461,0.00082,0.00165,0.31048,-0.09396,0.34122,-0.2597,-0.16753,-0.33479,-0.19438,0.13141],[0.11807,0.06219,0.06433,-0.17592,-0.27875,0.14219,0.32156,-0.02658,-0.23118,-0.29423,-0.13204,0.2882,0.00202,-0.19084,-0.13991,-0.11211,0.07622,-0.10225,-0.16817,0.01256,0.11038,0.1306,0.15033],[-0.21231,-0.35305,-0.17044,0.13126,-0.31651,0.10825,-0.1234,-0.10177,0.24545,0.02354,0.29125,-0.26438,0.20454,0.14055,-0.64803,-0.00361,0.31577,0.17447,-0.05209,-0.02265,-0.2767,0.01256,-0.37172],[-0.24901,0.21368,-0.00356,-0.00631,-0.09568,0.66766,0.26427,-0.12395,0.36926,-0.04738,0.18052,-0.18747,-0.17435,0.17366,-0.20735,-0.14311,0.2094,-0.21872,-0.25564,0.11621,-0.38314,-0.18199,-0.06419],[-0.30968,-0.27446,0.10367,0.09429,-0.31057,0.16693,0.33668,-0.26092,0.01518,0.35599,0.14808,0.06926,-0.04479,-0.17914,0.19642,0.13245,0.02389,-0.10181,-0.32284,0.04402,-0.16786,-0.07263,0.1248],[0.21595,0.21111,0.49555,0.21204,-0.08686,-0.21663,0.49504,-0.13177,0.00015,0.38927,0.36594,0.10919,0.04163,-0.34624,-0.16969,-0.09594,-0.04902,-0.06316,-0.01753,0.44792,0.09895,0.1111,-0.0139],[-0.04765,0.2308,-0.22477,0.02931,-0.28428,0.11089,0.06266,0.23209,0.28448,0.10432,0.3374,0.23734,0.07781,0.08026,-0.42555,0.05416,0.0201,-0.33033,-0.06634,0.20504,-0.26158,0.2545,0.11998],[-0.06321,-0.03332,0.16536,-0.36592,0.45771,-0.22731,-0.20891,0.18509,0.14447,0.24907,0.18929,0.11372,-0.29119,-0.08544,0.11329,0.03828,-0.44264,0.16854,0.34544,0.24281,-0.21017,0.10228,-0.42029],[0.40648,-0.3831,0.12412,0.2361,0.29198,0.0264,-0.26549,0.04162,-0.02221,0.14966,0.25508,0.19587,0.49879,0.19022,-0.10635,0.32602,-0.11115,0.06268,0.00742,-0.45319,0.10023,0.10438,-0.04059],[0.04104,-0.29688,0.02226,0.3263,-0.0412,-0.06626,-0.11052,0.48395,-0.24149,0.2407,0.07127,0.35795,0.31113,-0.02779,-0.3878,0.22834,0.31747,-0.22413,0.13933,-0.13055,-0.12566,-0.34186,0.19513],[0.08764,0.12128,0.29877,0.15051,0.02515,0.16205,-0.19592,0.00782,-0.22415,-0.25904,-0.00029,0.24849,0.46122,-0.21591,-0.02681,-0.06276,0.10586,0.12453,0.04489,-0.20767,0.08981,0.18139,-0.12693],[0.41635,-0.00492,-0.09374,0.03037,0.03918,-0.32099,0.01376,0.35523,0.45799,-0.068,0.21644,0.14049,0.08198,0.31098,0.27562,0.06727,-0.23399,-0.22369,-0.01363,0.07343,0.11823,0.18387,0.26222],[-0.66002,0.16254,-0.28268,0.27983,-0.00185,0.22095,-0.07901,0.08129,-0.33732,0.16117,-0.02319,-0.61411,0.19979,0.33323,-0.22795,0.1803,-0.18835,-0.43675,-0.08109,0.06999,-0.02345,-0.02111,0.05519]],"b1":[0.01962,0.01597,-0.03945,0.00986,-0.02406,0.01942,-0.05031,0.02184,-0.00993,-0.02427,0.02021,-0.03154,0.00124,0.06016,0.01678,-0.0864,-0.02063,0.03924,-0.01088,-0.01295,-0.00427,-0.00429,0.00683,-0.0309,-0.00927,0.03577,-0.01276,-0.01371,0.00899,0.05058,0.03207,0.00899,0.03091,-0.00585,-0.04673,-0.02629,0.06315,-0.01748,-0.03141,-0.07102,0.0095,-0.00483,-0.01286,-0.06576,0.00588,-0.02432,0.05483,-0.04363,0.02841,-0.01903,-0.02775,0.01018,0.04942,0.03567,0.04038,0.00043,-0.00566,0.02824,-0.05912,0.04877,0.05341,0.01041,-0.00321,-0.0129],"W2":[[0.09784,-0.03597,-0.11238,0.02286,-0.06596,-0.06935,-0.08037,-0.06791,0.0266,0.00987,-0.03378,-0.03565,0.13898,0.11295,0.05723,-0.23988,-0.00484,0.17441,0.13997,0.00712,0.05225,-0.04808,0.15789,-0.24252,0.12918,0.07613,-0.14726,-0.05012,0.1892,-0.09913,-0.12522,-0.00599,-0.05586,-0.1016,0.0585,0.05681,-0.10611,0.06586,-0.09487,-0.03158,0.03765,0.20412,0.04286,-0.09087,0.03727,-0.07004,-0.03452,-0.20518,-0.04319,-0.01003,-0.06813,-0.15226,-0.17305,0.06745,0.07277,0.27661,-0.06813,0.14005,0.13959,-0.07723,-0.02149,-0.07831,0.12304,-0.16111],[0.05426,0.16629,-0.03177,-0.0024,-0.0467,0.07887,0.01348,0.03043,-0.01346,0.07461,-0.1307,-0.08762,0.10725,0.15176,-0.03156,0.02627,0.07011,-0.03738,-0.15968,-0.14439,0.23015,-0.10874,-0.03128,-0.09143,0.11152,-0.13648,0.03337,0.1761,0.16819,0.05749,-0.03756,0.2102,-0.15075,0.17824,-0.03943,0.04406,0.09062,-0.04024,0.02979,-0.21936,-0.10029,-0.0374,-0.03507,-0.02172,0.20726,-0.16077,0.15341,0.01132,0.31871,-0.25493,-0.11674,-0.01689,0.25273,-0.06743,0.27418,0.22023,0.11607,0.11972,-0.28394,0.0273,-0.13372,-0.04064,0.01398,-0.00412],[0.08077,-0.02849,-0.16286,-0.17713,-0.14062,-0.01872,0.11735,0.04831,0.05672,-0.138,0.30648,0.01758,0.2455,-0.12427,-0.06224,0.34279,-0.14538,-0.02042,-0.07685,-0.04417,0.25285,-0.17125,0.11394,0.1933,-0.13605,-0.30829,-0.26758,-0.10011,0.00467,0.08062,-0.0133,-0.01979,0.06955,0.04574,0.04469,-0.03765,0.11961,0.0406,-0.04607,-0.04808,0.1045,0.21269,-0.07561,-0.29566,0.08375,-0.08798,0.06179,0.02149,0.31568,0.0326,0.12558,-0.16501,-0.40652,-0.01772,-0.01031,0.08763,-0.03317,-0.18326,-0.158,0.11274,0.10262,-0.10162,0.00628,-0.12931],[-0.27826,0.03886,0.14212,-0.12703,-0.16978,-0.02361,0.12263,0.22736,0.05794,-0.09699,-0.2625,-0.17791,-0.06869,-0.00129,0.05113,-0.04345,-0.18416,-0.2709,-0.00693,-0.05354,0.12018,-0.20215,0.24571,0.02237,0.03751,-0.02033,-0.09056,-0.2494,-0.08459,0.21558,-0.18422,-0.24496,-0.09089,-0.07155,-0.1491,-0.05914,-0.07369,-0.03032,0.0688,0.10831,-0.01898,0.07838,0.2264,0.13371,-0.36962,-0.00204,-0.24342,-0.02434,0.11271,0.03029,-0.02793,0.11693,-0.20558,-0.06754,0.01108,-0.03312,-0.13115,-0.04746,0.15562,0.07609,-0.00014,0.05501,0.07924,-0.23342],[0.01318,0.03988,-0.4135,0.18675,0.03274,-0.1359,-0.06526,0.01373,0.02107,-0.05863,0.15187,0.01842,-0.03397,-0.04219,0.00382,0.09284,0.09339,0.06239,0.073,-0.06061,0.13944,-0.08445,0.0503,-0.33366,0.04137,0.08953,0.03551,-0.27907,0.23501,0.1077,0.04271,0.06847,-0.05738,0.09448,0.17306,-0.18224,0.34306,-0.08497,0.04412,-0.11979,0.10091,-0.1789,-0.10296,-0.0291,-0.08107,-0.06905,0.16391,-0.22511,-0.1729,-0.02575,-0.0923,-0.09494,0.03132,-0.02697,0.07864,0.01904,-0.04799,0.01591,0.08144,0.12704,0.00336,-0.14648,-0.26915,-0.20004],[-0.19421,-0.11119,0.00055,-0.31538,0.19251,0.1207,0.14253,-0.07952,0.05192,-0.10435,0.19339,-0.12775,-0.09448,0.13083,0.33758,0.01245,-0.04638,-0.37035,0.20624,0.05871,-0.24082,0.09865,0.08038,0.01483,-0.05247,0.03643,0.00791,0.06268,0.14589,0.14116,0.08423,0.06547,-0.18469,-0.19036,0.13534,0.11551,0.03562,0.13579,0.00161,0.11492,-0.147,0.04051,0.15399,-0.06541,0.26454,0.20133,0.14023,0.0337,-0.18456,-0.03628,0.10841,-0.05889,0.03912,-0.00847,-0.11137,0.09372,0.15517,-0.13547,0.12489,0.02648,-0.00889,-0.02018,-0.11482,0.12965],[-0.1729,-0.09264,0.11902,-0.08052,-0.08873,-0.07551,-0.07455,-0.08499,0.1081,-0.13732,0.00891,0.2016,-0.03551,-0.14359,-0.07381,-0.01584,-0.06165,0.08363,-0.26388,0.25913,-0.29884,-0.02761,-0.09408,0.22102,-0.1056,-0.18181,-0.09125,-0.06798,-0.26285,0.09978,-0.33261,0.13496,-0.12514,0.1086,-0.05872,-0.05843,0.06948,-0.05637,-0.15693,0.01938,-0.24522,0.03561,0.19051,0.11318,0.00398,0.02765,-0.08965,-0.01393,0.06643,0.03721,-0.02574,-0.01879,-0.27216,0.18801,0.02314,-0.21839,0.03563,-0.17732,-0.01274,-0.06986,0.09792,-0.22096,0.02164,-0.28032],[-0.02733,-0.04226,-0.15025,0.11744,0.19312,-0.05931,0.01352,-0.05286,0.04485,0.1913,-0.03499,0.23602,-0.08366,-0.03532,0.24125,0.03771,-0.24775,-0.02598,-0.18621,-0.1572,-0.13805,0.02211,-0.1296,-0.09493,-0.0547,-0.00101,-0.11813,-0.16497,-0.28528,-0.06492,0.10476,0.11849,0.05554,-0.04757,0.00894,0.19693,-0.10757,0.07396,-0.19039,0.09031,-0.11079,-0.14277,0.0812,-0.1679,-0.09945,0.22932,-0.17676,-0.02413,-0.05663,0.13566,-0.2071,0.13814,-0.05308,-0.07601,-0.13481,-0.03123,0.20218,-0.00886,-0.07952,-0.01031,0.04325,0.02643,0.01698,0.01813],[0.09831,-0.15535,-0.31015,0.15884,0.20846,-0.13275,-0.00552,0.16271,0.19092,-0.01139,0.17088,0.03306,-0.02152,-0.23142,-0.16869,0.02566,0.03994,0.00363,-0.07511,-0.25906,-0.12202,0.01785,-0.23079,-0.11339,-0.07508,-0.01253,0.33039,0.01275,0.02053,-0.02844,-0.12172,0.2151,0.21328,-0.05051,-0.03133,-0.07021,0.0654,-0.18362,-0.00142,0.15207,0.14602,-0.11511,0.09676,0.2476,0.11997,-0.03669,-0.00665,-0.01321,0.20547,-0.35344,0.08641,0.0845,0.19805,0.05763,-0.21715,0.31802,-0.18989,0.1203,-0.06402,-0.03698,0.05744,-0.13097,-0.04226,0.12802],[0.06865,-0.00355,0.06005,0.1029,-0.09866,0.02097,0.03867,0.12055,0.01558,0.02858,-0.05186,0.18651,-0.08393,0.06099,0.01341,0.01532,-0.14113,-0.04837,-0.02629,-0.14242,-0.10609,-0.08816,-0.17329,-0.0724,0.16994,-0.17761,-0.13423,-0.18367,0.12571,0.11696,0.02666,0.00708,-0.11407,-0.07211,-0.05513,0.12108,0.08162,0.13805,0.30157,0.11934,0.03226,0.12795,-0.17673,-0.0026,-0.18621,0.22593,-0.10965,0.28219,0.16161,-0.02591,-0.18929,-0.1589,0.09719,-0.07676,-0.2088,-0.04954,0.22915,0.18434,-0.00884,0.06479,0.03957,-0.16955,-0.00752,-0.09401],[0.05997,-0.09726,-0.21517,0.15516,-0.06697,-0.13927,-0.10703,0.01578,0.14418,-0.02617,0.16791,-0.30747,-0.06397,-0.15853,-0.06846,0.11857,-0.06911,-0.10739,0.12968,-0.13517,0.09074,-0.06312,0.00077,-0.15424,0.06689,0.09825,0.33493,-0.02367,-0.07877,-0.14212,-0.00695,0.06972,0.00107,0.18422,-0.00248,-0.13374,-0.0325,-0.11864,0.01193,-0.14562,0.23105,-0.09102,-0.05043,0.10056,-0.07238,-0.03687,0.26322,0.01304,0.02101,-0.1081,-0.11279,0.28773,-0.03154,0.276,-0.05425,0.2231,-0.04555,0.06875,0.09436,0.03735,0.16318,-0.20948,-0.11349,0.34434],[0.0101,0.02499,-0.07759,-0.00107,-0.21273,-0.16497,0.25273,0.00593,-0.17497,-0.1357,0.11118,-0.14445,-0.02896,-0.11612,-0.19162,0.19837,-0.00653,0.0355,-0.0342,0.03257,0.04968,0.08986,-0.02208,-0.14005,-0.06187,-0.15779,-0.0042,0.28168,-0.09862,-0.0132,0.07794,0.17199,0.41685,-0.02513,-0.14334,0.09942,0.01339,0.22898,0.06613,-0.0997,0.0487,0.04588,0.04935,-0.12826,-0.21631,-0.2257,0.11817,-0.08733,-0.26704,-0.1039,-0.24482,0.02067,-0.08243,-0.00867,0.05437,0.04723,-0.04541,0.17946,0.25797,0.06601,-0.00668,0.07062,-0.16574,-0.04191],[-0.13943,0.04616,0.22575,0.2061,0.17448,0.10779,-0.11685,0.0032,0.08441,0.01337,0.23451,-0.07813,-0.1358,0.05867,0.3003,-0.23021,-0.12815,0.09349,-0.12691,0.30803,-0.14025,0.07897,0.02996,0.12733,-0.12075,-0.24633,0.05168,-0.07315,-0.18468,-0.14132,-0.17645,0.12495,-0.19883,-0.09164,0.26951,-0.04877,0.01257,0.06506,-0.01445,-0.06455,0.11318,0.05899,-0.05345,-0.17203,-0.15233,-0.11671,-0.29114,0.1945,0.04107,0.10244,-0.18351,-0.11613,-0.03369,-0.04403,-0.07682,-0.20815,0.18139,0.07859,0.05006,0.17124,0.13474,0.20117,0.11306,-0.09454],[-0.12577,0.11353,-0.05623,-0.19262,0.08177,0.08363,0.06054,-0.10551,0.04723,0.12827,-0.05978,0.00123,-0.1016,0.16792,0.14151,-0.15476,-0.19794,0.00877,-0.06349,0.1212,-0.01217,0.08848,-0.18629,0.01658,0.02662,-0.02871,-0.16668,-0.06071,-0.11302,0.01027,0.05301,-0.05375,-0.0811,-0.30552,-0.11113,0.26686,-0.21341,0.18287,0.022,-0.20959,0.13615,0.18108,-0.03511,0.09985,-0.04063,0.04607,-0.03448,0.05335,0.09609,0.09442,-0.03722,0.00377,-0.10177,-0.03246,-0.10665,-0.01937,0.22839,-0.0931,-0.14075,0.2394,0.02462,-0.05309,0.0486,-0.16858],[0.13847,0.06967,0.05988,0.13761,-0.23243,-0.12962,0.13222,-0.28042,0.02097,-0.10239,0.09471,-0.03061,0.18005,0.04567,0.01968,-0.09273,0.10577,0.35548,-0.17729,-0.1679,0.24903,0.06535,-0.06538,0.02371,0.02358,-0.06379,-0.06461,0.09385,0.04683,0.11406,0.11679,-0.05308,0.23216,-0.09018,-0.03792,-0.12573,-0.16618,0.08347,-0.17103,-0.07113,0.14769,0.0776,-0.10152,-0.16681,-0.03093,-0.20257,-0.11605,-0.10844,-0.01914,-0.10605,-0.03055,0.20557,0.04998,0.20148,0.14199,0.06977,-0.1515,-0.14612,-0.03344,-0.07448,-0.07748,0.05902,0.07116,0.07385],[-0.01902,-0.06128,-0.03691,0.17291,-0.14512,0.00292,-0.09262,-0.03095,0.15847,0.14153,-0.03254,0.16922,-0.14342,0.03213,0.24909,-0.071,0.02953,0.1292,-0.13628,0.06269,-0.08371,-0.17711,0.08047,-0.00763,0.19083,-0.0435,-0.08044,-0.00174,-0.02687,0.00315,-0.02761,-0.09527,-0.1413,0.04758,-0.24414,-0.07917,0.00512,0.06858,-0.02773,0.04308,-0.12083,-0.22234,0.11855,0.01002,-0.13473,0.22023,0.02645,-0.01796,0.17897,0.1137,0.11825,-0.11934,-0.09423,-0.15266,-0.10505,-0.32563,0.17677,-0.1728,0.11931,-0.1249,-0.01457,-0.11645,0.05777,-0.26182],[-0.1169,0.11469,0.05438,-0.11277,-0.10182,-0.28064,-0.13415,-0.09803,0.19963,0.0687,-0.01958,-0.06556,-0.12177,0.04471,-0.04149,-0.04296,0.26114,-0.18059,-0.06288,0.00941,-0.10866,-0.18211,0.18078,-0.06448,-0.11626,0.00172,-0.19743,-0.04583,-0.05055,-0.00166,-0.04542,-0.18476,-0.13055,0.14819,0.17176,-0.0182,0.10317,-0.09514,-0.14274,-0.21056,-0.18323,0.12264,0.00142,0.21983,0.12961,0.0364,-0.07307,0.03175,0.13129,0.00433,0.02825,-0.14097,0.13662,-0.09345,-0.1467,-0.26127,0.12339,-0.13407,0.04733,-0.03522,-0.06444,-0.25027,-0.05132,-0.20131],[0.3178,0.15734,0.03135,0.39995,0.10162,-0.14722,-0.17937,0.05111,-0.09738,-0.00069,-0.09912,0.20488,-0.04005,0.22019,0.1281,-0.11039,0.02991,0.26173,0.16458,0.09497,0.06329,0.01934,0.13268,0.00422,-0.01516,0.00428,-0.11608,-0.00882,0.25093,-0.06348,-0.08428,-0.03261,0.03474,-0.0425,-0.00026,0.02221,0.20695,0.01245,-0.05081,-0.32645,0.08666,-0.15288,0.098,-0.02427,-0.07764,-0.14532,-0.10674,-0.09837,0.06732,0.06053,0.02781,0.1522,0.185,-0.00505,0.25856,-0.12627,0.0331,0.02713,-0.33877,0.213,-0.17871,0.09147,0.20633,0.03932],[0.01705,-0.26543,-0.05075,0.2921,-0.10039,-0.08478,-0.28064,0.19692,-0.0129,-0.03999,0.13545,-0.07594,-0.04187,0.31081,-0.1424,0.02084,0.00543,0.07717,-0.00949,-0.16022,-0.01087,-0.03007,-0.11945,-0.16334,-0.17432,-0.04637,0.0928,0.00355,0.21329,0.08195,-0.07289,-0.0653,0.02573,-0.01683,-0.04291,0.21289,0.03597,-0.04072,-0.20193,-0.21051,0.03231,-0.12287,-0.02364,0.13063,-0.05613,-0.14136,0.1354,-0.08533,0.04675,-0.06902,0.22514,-0.16033,-0.06132,0.10369,0.22547,-0.08724,0.16402,0.00128,-0.17486,-0.07251,0.10268,0.07659,-0.13572,0.09944],[0.04807,-0.15636,0.17215,-0.05159,-0.00763,-0.16983,-0.016,0.00688,0.02695,-0.1415,-0.04846,0.01613,0.25982,0.13025,0.09958,0.03464,-0.04311,0.05531,0.17265,-0.15447,-0.1017,0.0404,-0.21805,-0.15475,-0.14659,0.03105,0.19802,0.28647,0.11638,-0.13464,0.10993,0.08277,-0.02416,0.17782,-0.0652,-0.10661,-0.01477,0.08871,-0.13988,0.0242,-0.19451,0.03364,-0.08762,-0.04256,-0.32942,-0.22897,-0.17839,-0.03131,-0.14239,-0.00638,0.02933,-0.04438,-0.08376,0.20767,0.09421,-0.14166,0.02804,-0.04481,0.03282,0.05306,-0.12135,0.03994,-0.4767,0.11647],[-0.27994,0.31848,0.05295,-0.16313,0.18446,-0.1179,0.03905,-0.04187,0.07543,-0.04713,0.00871,-0.07744,0.02495,-0.02559,-0.04737,0.13384,0.1197,-0.07802,-0.05726,-0.01026,-0.0526,-0.05656,-0.01863,0.04606,0.00939,0.13606,0.10216,0.06157,0.06642,-0.05491,-0.12046,-0.0155,0.11895,0.11913,-0.10817,-0.18722,0.14843,0.2224,-0.01139,0.0624,-0.032,0.20317,-0.0285,0.10722,0.05685,0.13269,0.08597,0.04388,-0.19048,0.19508,-0.00459,0.0717,-0.02497,-0.14823,-0.01407,-0.12178,0.00762,-0.13126,0.20217,-0.09624,-0.09166,-0.10424,-0.1027,-0.08237],[0.0834,0.05142,0.00517,0.03674,-0.2385,0.15641,-0.12589,0.04156,-0.21143,0.18377,0.08875,-0.14533,0.0256,0.25168,-0.0859,-0.26301,0.02032,0.28831,-0.21529,0.15411,-0.00688,-0.02709,0.00174,-0.04516,0.06238,-0.10667,-0.00564,-0.10622,-0.03657,0.42721,-0.08604,0.07183,0.0683,-0.19082,-0.2583,-0.2211,0.05707,-0.18893,-0.2143,-0.26072,0.09215,-0.041,0.08528,-0.10674,0.08145,0.05434,0.29548,-0.18842,-0.03352,-0.22367,-0.24629,0.14738,0.13299,-0.15711,-0.08249,0.09537,-0.0159,0.09242,-0.08684,0.17044,0.22289,-0.20914,0.15949,-0.00469],[-0.1915,0.08799,-0.09252,-0.28058,-0.03175,0.1605,-0.26552,-0.15328,-0.18236,-0.14425,-0.05541,-0.14228,-0.00403,-0.04911,0.11258,0.23675,-0.09691,-0.13238,-0.08999,-0.0157,-0.00022,0.07065,-0.03706,0.28275,-0.03434,-0.04711,-0.15477,0.05914,0.07238,0.03649,-0.29594,-0.04364,-0.00482,-0.07174,0.02153,0.09132,-0.26262,0.11185,-0.17096,0.05285,0.0058,-0.00172,0.09142,0.0792,-0.05388,0.01074,0.04647,0.10219,0.07646,0.13606,-0.03128,-0.18884,-0.14691,-0.22118,-0.31571,-0.1293,-0.11024,-0.04232,0.04122,0.07326,0.0882,-0.11567,0.02416,-0.14443],[-0.17537,-0.10261,0.06632,0.01544,-0.01768,0.10062,-0.02274,0.11259,-0.02491,0.0387,0.20394,-0.02729,-0.00408,0.15944,0.08811,0.13798,-0.02051,-0.10727,0.07275,-0.015,-0.14635,-0.09611,0.0223,0.19203,0.19172,0.10499,-0.02119,0.0695,-0.17194,0.01205,-0.00593,0.15789,0.00361,-0.00955,0.01918,-0.22285,-0.02004,0.04821,0.15209,-0.10061,0.01752,-0.235,0.02585,0.11389,-0.05414,-0.09353,0.30783,0.05426,0.21255,-0.09747,-0.1324,-0.22356,0.08939,-0.22968,-0.33195,-0.0806,0.0244,-0.0484,-0.18191,-0.0507,0.09512,-0.21856,0.09531,0.03898],[0.15751,0.06511,0.15821,-0.0322,0.18917,0.00573,0.22385,0.04407,-0.05355,0.02294,-0.22537,0.16527,-0.05339,0.12898,0.29908,0.0113,0.00082,-0.05765,0.02617,0.30516,0.15139,0.01903,0.14583,-0.19903,0.03183,-0.39402,0.12759,-0.26434,0.05724,-0.11519,0.29851,-0.16521,-0.05352,0.1882,0.07678,0.05022,-0.04434,0.03965,-0.06654,-0.00788,-0.03695,0.10445,0.02834,0.11345,-0.18393,0.22622,0.03887,0.25307,0.06444,0.01205,0.03224,-0.1685,-0.03831,0.00197,-0.0053,-0.21831,0.065,-0.1543,0.0139,0.11639,-0.1235,-0.02032,-0.01528,0.01565],[-0.0695,-0.21504,-0.28707,0.36547,0.02029,0.19909,-0.04603,0.15306,-0.12465,-0.19,0.12119,-0.04041,0.07455,0.15227,0.04502,-0.05559,-0.33105,-0.07048,0.10603,0.20117,-0.00322,0.19871,-0.28746,0.07037,-0.13604,-0.05051,0.01038,0.06851,-0.12748,0.07395,-0.17828,0.01228,-0.05695,0.05861,-0.3185,-0.18178,-0.13077,0.07044,-0.35141,0.1543,0.00093,-0.02978,-0.33614,-0.09555,0.09443,0.07684,-0.06264,0.07882,0.01262,0.0073,-0.0838,0.23046,-0.13534,0.13985,0.28065,0.10817,-0.04081,0.15867,-0.03471,-0.06968,0.15985,0.15461,-0.20709,0.29379],[0.07878,0.00255,0.08011,-0.0849,0.00856,-0.0798,-0.02652,0.03774,-0.04306,-0.04541,0.09905,0.02482,0.12389,0.14675,-0.31502,0.0644,0.27851,-0.03752,0.36732,-0.28207,-0.08801,0.12024,-0.09621,-0.17791,0.03661,0.15091,0.1115,-0.04691,-0.2132,0.11875,0.16636,-0.05074,0.14543,-0.12067,-0.13365,-0.2129,-0.01094,-0.1645,0.11966,0.05871,0.04527,0.11777,0.1324,0.05867,-0.09243,-0.02791,-0.00552,-0.01138,-0.08508,-0.0469,0.09632,-0.10046,0.02828,0.039,-0.02862,0.31485,0.12412,-0.0509,-0.09314,0.00696,0.16567,-0.05547,0.04231,-0.21588],[0.00572,0.15454,0.03416,-0.14318,0.17897,-0.11764,-0.00656,0.07397,-0.04443,0.34912,-0.03213,-0.21479,0.11725,-0.02505,0.05345,0.10917,0.10939,-0.07453,0.07449,0.08131,0.12671,-0.07321,-0.02519,0.15276,0.08294,-0.07051,0.00732,0.08684,-0.13232,-0.1707,0.09761,0.05231,-0.12783,-0.07564,-0.02366,-0.01963,0.13725,0.04374,-0.06814,0.11605,0.02852,0.01889,0.02972,0.08291,0.10117,0.06831,-0.11567,0.03573,0.12576,0.16412,0.00239,0.32234,0.25437,-0.1769,0.09275,-0.147,0.00378,0.12818,0.1666,-0.14864,0.01064,-0.30639,-0.1669,-0.0308],[-0.04342,0.20415,0.00531,0.23496,-0.05686,0.03377,-0.04685,0.24491,0.08493,-0.16409,-0.02539,0.15926,-0.18198,0.32729,0.01161,-0.19623,-0.33383,0.09395,0.20888,0.02888,0.05963,0.07047,-0.04786,-0.02506,-0.09407,0.02064,-0.09301,-0.02795,-0.02224,0.05404,-0.06255,-0.2187,-0.08721,0.03344,-0.03931,-0.07988,0.1082,-0.25767,0.03497,-0.12271,-0.05721,-0.05725,0.00198,-0.07244,0.08368,-0.31226,0.04902,0.05729,0.08262,-0.1289,-0.0296,0.25698,-0.05068,0.07508,0.07205,0.11527,-0.1171,0.28659,0.01021,0.30975,0.07552,-0.06117,-0.23839,-0.01594],[-0.09906,-0.28015,-0.14579,0.08527,-0.06691,0.33582,0.21295,0.04873,0.03413,-0.06224,-0.03955,0.0327,0.01787,0.25739,0.05729,-0.1447,-0.19269,0.10864,0.17694,-0.15748,-0.06728,0.31707,-0.0306,-0.15267,0.28531,0.01982,-0.08297,0.01782,-0.1285,-0.2075,0.16168,-0.04567,0.15015,-0.11067,-0.08666,-0.16967,-0.08224,0.18665,0.02917,0.04783,0.05982,-0.11412,-0.05608,-0.07907,0.07334,-0.07504,0.05737,0.28267,0.01342,-0.2122,0.1651,-0.05271,-0.08731,0.28046,0.15119,-0.06218,0.00964,0.25308,-0.18849,0.00229,0.36104,-0.02226,-0.2647,0.03448],[-0.07568,-0.04962,0.25516,-0.06061,0.02281,0.07987,-0.01498,-0.25116,0.09174,0.07708,0.02241,0.11711,0.10611,0.14381,-0.19915,-0.08304,-0.12206,-0.13303,0.03572,-0.18623,0.0344,-0.21947,-0.1,-0.10709,-0.05893,-0.01451,-0.33154,0.16147,-0.14812,0.30997,-0.19092,0.10296,-0.00013,-0.07218,-0.16779,-0.02328,0.15138,0.11618,0.15072,0.20398,-0.00839,0.02318,0.12547,0.08407,0.18469,0.08,0.04568,0.17857,0.01884,-0.08988,-0.00654,-0.03021,0.0962,0.04429,-0.13796,-0.00751,-0.00602,-0.06654,0.16469,-0.10398,0.12668,-0.04878,0.11462,-0.11667],[-0.01567,-0.10499,0.06512,-0.11753,0.00399,-0.08467,-0.01442,0.22757,-0.08343,-0.108,-0.21102,-0.25254,0.02626,-0.07815,0.05916,0.00525,-0.06872,0.27502,0.1803,-0.04448,0.13808,0.25909,0.0489,-0.12399,-0.00843,0.08974,0.08525,0.12269,0.07854,-0.22368,-0.06112,-0.09322,0.10163,0.03105,0.03871,-0.07394,-0.09387,0.10771,-0.01434,-0.04676,-0.17377,-0.18568,-0.17709,0.00078,-0.13671,0.09657,0.12067,-0.0947,-0.11279,-0.08953,0.01892,0.05334,0.00774,0.18265,0.01888,0.0276,0.04484,0.05342,0.09892,-0.10691,-0.11115,0.06947,-0.0792,0.15779],[0.03171,0.10335,-0.09324,-0.10158,-0.04145,-0.04842,-0.00036,0.19056,0.29927,-0.0695,-0.02848,-0.0748,-0.13692,-0.08423,-0.1508,0.07921,-0.13685,-0.08028,-0.12666,0.13173,0.08379,0.04416,0.0025,-0.11198,-0.1357,-0.00538,-0.11119,-0.06584,0.0331,0.28051,-0.12086,0.04408,-0.12685,-0.17408,0.00839,-0.00031,-0.07335,0.10087,0.00952,-0.20095,0.0177,0.09172,-0.15412,-0.08113,-0.04988,0.16441,-0.08339,0.19198,-0.07949,0.16339,0.03714,-0.08965,0.04733,0.05444,0.02421,-0.05951,0.14858,0.09859,0.02304,0.04512,-0.2172,-0.07135,0.18055,-0.17087],[0.01207,0.00858,0.25211,-0.05936,-0.23649,-0.03817,0.00525,0.06998,0.05181,0.1737,0.02942,-0.20513,-0.12612,0.01715,-0.04993,0.04652,0.01123,0.14681,-0.12173,-0.09901,-0.20928,-0.09609,0.09741,0.0693,-0.0813,-0.17546,-0.06985,0.07242,0.09085,0.13611,0.03377,0.00183,-0.14889,0.01613,0.21691,0.03169,0.09846,0.00321,0.07692,-0.10002,-0.20682,-0.10642,-0.20609,0.11327,0.29118,-0.01275,-0.13749,0.10961,0.23018,0.05451,0.15798,-0.04848,0.03217,-0.23284,0.00672,-0.14736,-0.08695,-0.19013,0.09598,0.31747,-0.13061,-0.0075,0.03397,-0.00133],[0.16449,-0.03397,-0.11499,-0.14884,0.06301,-0.01686,-0.163,-0.14051,-0.11539,-0.07343,-0.1386,0.073,-0.05841,-0.07679,-0.16525,0.07992,-0.00026,0.08105,0.12637,-0.0677,0.03512,0.15778,0.02801,-0.06255,-0.05641,0.07762,-0.02162,0.16655,0.10404,0.04097,-0.029,0.02466,0.15773,-0.20613,-0.04068,-0.08472,-0.1714,0.07419,-0.11928,-0.1268,0.17889,-0.24609,0.11324,0.16086,-0.23313,-0.03697,0.06926,0.06129,-0.10695,-0.04092,0.0947,-0.05638,0.07764,0.19269,-0.06724,0.0735,-0.05294,-0.12139,0.0073,-0.08468,-0.10494,-0.07991,-0.22338,0.10807],[0.00063,0.05936,0.00744,-0.01609,-0.00941,0.07656,0.01622,0.08994,0.05868,-0.06926,0.03152,0.24062,0.10487,-0.17641,-0.11089,-0.12254,0.12579,-0.03447,-0.21412,-0.02623,0.17052,0.00411,0.00951,-0.07032,0.08115,-0.16516,-0.06642,0.09832,-0.13852,-0.062,0.03693,0.07895,0.16314,-0.05755,0.1838,0.05477,-0.27074,0.19963,0.19739,0.23868,-0.16616,0.06765,0.282,0.32137,0.02525,0.15212,-0.19963,0.02516,-0.11577,0.23394,0.03583,-0.02921,-0.00207,-0.05941,0.10597,-0.02802,0.05278,-0.20341,0.08323,-0.21585,-0.18355,-0.09899,0.22313,-0.14738],[0.29048,0.0792,0.04104,0.01735,0.11747,0.24999,0.17671,0.08547,0.01634,0.09603,0.03177,-0.05096,0.16937,-0.02868,0.0954,0.27127,0.13603,0.01345,0.01077,0.08696,-0.06088,0.2454,-0.05516,-0.06672,0.02787,0.44535,0.00478,-0.08186,-0.02645,0.09416,0.25255,-0.10549,0.00534,0.01398,-0.05687,0.15529,0.12459,0.1792,-0.12837,-0.14543,0.02127,-0.12626,-0.02057,0.10617,-0.29737,0.06774,0.29131,-0.23092,0.0384,-0.19031,-0.13504,0.04879,0.22209,0.23833,0.0533,0.16211,-0.34521,-0.03272,-0.07512,0.04895,-0.05366,-0.10726,-0.20711,0.15487],[0.07452,-0.04577,0.11094,0.06302,-0.12147,-0.04947,-0.28789,0.02406,0.01403,0.04785,0.32638,-0.0758,-0.17859,0.29145,0.1493,-0.14135,-0.03721,-0.00187,0.04646,-0.02212,0.09461,0.11976,0.21354,-0.09537,0.05399,0.18095,0.04737,0.20501,0.07014,0.06476,0.27462,-0.08926,0.18151,-0.04353,-0.01076,0.08607,0.17821,0.14245,-0.16193,-0.20396,0.05243,-0.02265,-0.08339,0.2448,0.01703,-0.22349,-0.07328,0.03061,0.03895,0.07994,0.08116,0.15774,0.15931,0.11259,0.04301,0.22663,0.17018,-0.04547,-0.02128,0.15135,0.25071,0.0824,0.09269,-0.18994],[-0.03715,-0.02847,-0.18416,0.1068,-0.01865,-0.13901,0.11878,-0.04704,-0.05528,0.10814,-0.06618,0.13904,0.09081,-0.12975,-0.20494,0.02987,-0.06565,-0.09149,-0.13422,-0.09535,0.00294,0.0518,-0.20045,-0.1523,-0.10322,-0.07162,0.22949,0.01084,0.32119,0.10115,0.11968,0.20324,0.23192,-0.18216,-0.11486,-0.02381,0.04717,-0.00049,-0.192,0.04117,0.12259,-0.14875,-0.16634,0.06171,-0.03242,0.10035,0.15606,0.01977,-0.19713,-0.14348,-0.20392,0.00055,-0.12761,0.22489,0.11281,0.04243,-0.18924,0.21925,-0.15721,-0.27729,-0.05817,-0.19824,0.03319,-0.04677],[-0.01769,0.19681,-0.08783,-0.08079,0.25217,-0.01611,-0.05173,-0.03503,-0.08751,-0.08256,0.16368,0.18508,-0.18614,0.23507,0.2343,-0.00519,-0.24947,-0.21569,-0.08046,0.03117,0.18184,-0.20908,-0.04195,0.02822,0.11194,-0.10382,-0.07772,0.01817,-0.09993,0.09075,0.0577,-0.13777,0.06537,-0.03968,-0.01256,0.13941,0.18111,-0.08929,0.04604,0.04083,0.18598,0.06466,0.05387,0.22784,-0.01056,0.17392,-0.12585,0.10032,0.13874,-0.06069,-0.02898,-0.03587,0.14708,-0.24737,-0.12271,0.11069,-0.24924,0.20662,0.0429,0.29692,-0.08512,0.12166,0.01578,-0.0115],[-0.14098,-0.03888,-0.18389,-0.45835,-0.00898,0.29018,-0.05062,-0.07239,0.07038,-0.08683,-0.09566,-0.14752,-0.15316,0.11983,-0.06245,0.02962,-0.04547,-0.16749,0.01914,0.14747,-0.18182,0.0276,0.13878,0.04827,0.01068,-0.04425,0.00864,-0.14555,-0.09568,0.02151,-0.13083,-0.02679,-0.0662,-0.11317,0.04193,0.20323,-0.12869,-0.05081,0.01073,-0.12539,-0.10521,-0.13478,-0.19284,0.06646,0.12891,0.26964,-0.06029,0.18673,-0.03045,-0.03841,0.0901,-0.07557,0.07842,-0.00513,-0.16422,-0.19084,-0.11937,-0.09466,0.15522,0.00279,0.13799,0.07926,0.04237,-0.16706],[-0.29521,0.16396,0.20382,-0.00512,0.23601,0.02637,-0.04687,0.12997,-0.03686,-0.05253,0.00596,-0.21641,-0.07289,0.04808,0.14302,0.0799,-0.16002,0.05281,-0.01499,0.25029,0.06852,-0.07091,0.12618,-0.0587,-0.11766,-0.01055,0.01216,0.15214,-0.04603,0.06275,0.09285,-0.037,0.15838,-0.18788,-0.21667,-0.03736,0.20896,-0.05623,-0.16909,0.06783,-0.15513,0.1306,0.09379,0.13521,0.13229,0.09715,0.10331,0.00323,0.03766,-0.03667,0.01734,-0.06648,-0.101,0.1489,-0.20048,0.15608,0.31739,-0.12119,0.12678,0.03584,0.26456,-0.04957,0.24444,-0.08573],[-0.13084,0.20742,-0.1067,-0.07421,-0.03387,-0.01639,0.11049,-0.08588,-0.13445,0.09074,-0.06758,0.11178,0.13265,0.30211,0.2187,-0.12354,-0.14108,-0.05292,-0.01769,-0.0164,-0.05145,-0.14388,0.12481,-0.24167,-0.09788,0.17515,0.08828,-0.01062,0.12331,-0.00651,0.23083,0.1084,-0.05837,0.1474,-0.08843,-0.01653,0.21196,-0.17495,0.11839,-0.07237,0.17354,-5e-05,-0.10727,-0.21471,-0.16461,-0.0377,-0.09985,-0.09758,-0.0058,-0.13857,-0.07068,0.09947,0.20477,0.07965,0.128,0.09965,-0.02646,0.06765,-0.29448,0.02423,0.00043,0.17485,-0.23972,-0.09419],[0.3451,0.10393,0.15434,-0.0234,-0.21129,0.0543,0.03749,0.08474,0.15766,0.07743,-0.18239,-0.09581,-0.2317,0.11375,-0.07463,-0.18055,0.02791,-0.06676,0.15334,0.04833,0.28609,-0.06391,0.12301,-0.14386,-0.14244,0.29803,-0.09365,0.21519,-0.02919,0.06248,-0.04419,-0.09834,-0.05201,0.00057,-0.0012,0.12013,-0.0847,-0.03355,-0.09918,-0.05843,-0.07574,-0.18437,-0.07944,0.00241,0.28149,0.11977,-0.00672,-0.23302,-0.23157,0.07074,0.11696,0.27633,0.0328,0.08614,0.00682,0.19594,0.10809,-0.05436,-0.04277,-0.16024,0.03618,-0.18537,-0.02612,0.19359],[0.21749,-0.16584,-0.02764,-0.09322,0.13978,0.03855,-0.09468,-0.12012,0.0957,-0.22716,0.00282,0.02894,-0.01692,-0.03233,-0.01682,0.06402,-0.11033,0.00936,0.15505,-0.38839,0.15847,0.21263,0.22455,-0.36342,-0.06973,-0.02259,0.20209,0.07579,0.17251,0.10533,-0.14232,0.08945,-0.22469,0.11161,0.01614,-0.05238,0.01567,0.10086,-0.14663,0.02523,0.19923,0.04591,-0.01699,-0.00324,-0.19097,0.15717,-0.02881,-0.08249,-0.06098,-0.062,-0.08125,-0.05614,0.12764,0.21117,0.06824,-0.1195,-0.04218,0.25874,0.02423,-0.0717,-0.1044,0.05245,-0.14072,0.20104],[0.1399,0.06525,-0.02459,0.13606,0.06317,-0.03907,0.18872,0.00515,-0.07086,-0.20889,0.00558,-0.21685,0.13299,-0.11242,-0.14166,-0.26617,0.01884,0.00112,0.09679,-0.15522,0.15976,0.10576,0.13791,-0.13797,0.1387,-0.07573,0.02563,-0.01753,0.20925,0.09447,0.28237,0.06395,0.10515,0.00368,0.01804,-0.13324,-0.1244,-0.08313,-0.05609,-0.07371,0.31975,-0.03731,-0.05974,0.13864,0.10421,-0.09661,0.14369,0.07544,-0.12949,-0.0589,0.10479,0.1503,-0.10206,0.18688,0.07223,-0.07076,-0.047,0.20884,0.01083,-0.2503,-0.10083,0.17247,-0.06001,0.07669],[-0.19826,0.22401,0.03906,-0.16495,0.01009,0.39472,-0.01058,0.04136,-0.13936,-0.06413,-0.25456,0.14249,-0.12953,-0.06454,0.06806,0.04977,0.26583,-0.07404,0.08958,-0.03625,-0.15401,-0.12103,0.04029,0.13805,0.1267,0.05308,-0.10531,0.02985,0.09137,-0.1702,-0.12142,-0.18646,-0.01459,-0.02438,-0.07521,0.09481,-0.06182,0.16216,0.00996,0.06244,-0.23094,0.09759,-0.00686,-0.20859,-0.21387,-0.0625,0.04042,-0.20506,0.1687,0.04888,0.04147,0.03802,-0.11477,-0.06124,-0.07528,-0.14291,0.13009,-0.00548,0.13272,0.15064,0.02358,0.13534,-0.0745,-0.21791],[0.22991,-0.02594,0.08306,0.20338,0.01738,0.11117,-0.15955,-0.13878,-0.19185,-0.09118,-0.33336,-0.0784,-0.19908,0.17694,0.02272,0.02678,-0.26781,0.01356,0.18643,-0.10617,0.07982,-0.09815,-0.01784,0.27126,0.07108,0.23248,0.00571,-0.18992,-0.17581,0.12329,0.0271,0.17293,-0.03737,0.1094,-0.12474,0.02331,-0.23424,-0.03605,-0.19569,-0.05706,0.14573,0.02346,0.04802,-0.1179,-0.35158,-0.19805,0.06691,-0.20713,0.05262,-0.2791,-0.06458,0.17255,0.13169,-0.07556,0.1171,-0.01639,-0.05874,0.19037,-0.01747,-0.24084,-0.02515,0.20642,-0.04698,0.27083],[0.14871,0.10424,-0.26762,0.02083,0.15263,0.13979,0.05508,-0.08959,-0.13421,-0.07293,-0.01094,-0.16148,-0.07001,0.08488,0.11309,0.03096,0.02337,0.246,-0.1101,-0.08034,-0.00117,-0.00733,-0.14238,0.03116,0.08713,0.03629,0.25881,-0.17715,0.03575,-0.13875,0.25491,0.01052,0.13881,-0.01267,0.09803,0.05252,0.056,-0.01299,0.07952,0.02761,0.15581,-0.19378,0.14147,0.15451,0.10162,-0.0682,0.06262,-0.20389,-0.10093,0.19662,-0.02177,-0.01429,0.07982,0.04147,-0.06418,0.1139,-0.10676,0.02741,0.10089,-0.01761,0.17877,0.02446,-0.16196,-0.03404],[-0.00863,-0.14674,0.02236,0.13827,-0.00975,-0.05808,0.40197,0.1061,0.18606,0.07371,0.00538,0.24382,0.13276,-0.36387,0.12994,0.09114,-0.08519,0.11533,0.06199,-0.11952,-0.24863,0.06277,0.04431,0.10417,0.05714,-0.01179,-0.06395,-0.04177,0.07125,-0.31109,-0.02288,0.0333,-0.00746,0.03918,0.35134,0.00494,-0.12804,-0.00988,-0.15364,0.18602,0.05195,0.20111,0.10192,0.20302,-0.02388,0.13835,-0.13948,0.16134,-0.19899,0.20105,0.12554,0.12314,-0.07144,-0.10004,-0.03409,-0.05236,-0.0955,-0.16503,0.03034,-0.32906,-0.17706,-0.04975,0.19834,0.17153],[0.22659,0.07897,-0.05709,-0.13684,-0.05691,-0.07762,-0.04376,0.11497,-0.02124,-0.11831,-0.23251,0.15605,0.1957,-0.01,0.01854,-0.01872,0.14005,0.13884,0.01379,-0.14634,-0.12214,0.02095,0.12626,-0.29073,0.04349,0.1294,-0.02804,-0.20655,-0.01491,-0.08483,0.10113,0.10527,0.03564,0.11727,-0.10324,-0.02397,0.11037,-0.04236,-0.06806,-0.17783,0.00334,0.04933,-0.09246,0.19992,0.06335,-0.01388,0.14072,0.06709,0.12429,-0.14057,-0.10024,0.0431,-0.07295,0.30592,-0.16748,0.05921,-0.23009,-0.12731,0.14538,-0.07879,-0.006,0.03024,-0.17143,0.12573],[0.11967,-0.02562,-0.08663,0.11375,0.09343,-0.02401,0.01221,-0.15629,0.06457,-0.0973,0.11341,0.03515,0.19535,0.07854,-0.15776,0.26706,0.16533,0.07551,-0.05035,-0.17433,0.22539,0.0077,0.00934,0.02195,0.02283,0.13704,0.03133,0.07773,0.17209,-0.00142,0.08998,-0.01446,0.09343,0.20424,0.00124,-0.0667,0.00405,-0.16971,-0.06469,0.13801,0.25191,-0.24712,0.05726,0.02302,-0.19526,-0.00868,0.09395,0.01753,-0.16754,-0.10141,0.03862,-0.13825,0.22012,0.08593,0.00602,0.20916,0.12588,-0.01417,-0.15991,0.07107,0.36722,0.01162,-0.10411,-0.07844],[0.19215,-0.15762,0.05429,0.15371,0.02955,-0.03614,0.27764,-0.01522,0.25364,0.05126,0.19232,-0.08908,0.12708,0.05433,-0.06259,-0.18142,0.02846,0.35886,0.26599,-0.07142,0.29394,0.05083,-0.09322,-0.07443,0.06326,0.14821,0.02815,0.03201,0.11344,0.03305,-0.06924,0.00963,0.06175,-0.00199,0.00996,-0.00498,-0.0569,0.17951,-0.13536,-0.02733,0.01098,-0.11475,0.007,0.0026,0.0395,0.18377,0.05144,-0.12561,-0.36131,0.0035,-0.13295,-0.05552,-0.04077,0.10353,0.00844,0.00275,-0.05044,0.00094,-0.08765,-0.12649,-0.11193,0.10572,-0.03581,0.04912],[-0.09794,0.204,-0.03241,0.03215,-0.15741,0.14858,0.09042,-0.12773,0.05377,-0.11225,-0.05662,-0.12677,-0.02427,0.11388,-0.04501,-0.05262,0.06485,-0.25333,0.0081,-0.0204,-0.0145,-0.36193,-0.0963,0.05561,0.09269,-0.20345,-0.19937,-0.03389,-0.09291,0.01056,-0.11972,0.00621,-0.18211,-0.01059,0.13658,-0.03775,-0.07382,0.22001,-0.08148,-0.10476,-0.25537,-0.09746,-0.01234,-0.06426,0.26299,-0.06364,-0.1909,0.02607,0.17694,-0.01543,0.03712,-0.00335,-0.07894,-0.04511,-0.14715,-0.34969,-0.04879,0.0915,0.02909,-0.12073,-0.0913,0.1894,-0.09419,-0.03544],[-0.04827,0.15674,-0.03774,0.13863,-0.06255,0.1259,0.08841,0.20712,0.19901,-0.10882,0.22646,-0.212,-0.15094,0.08333,-0.13646,0.00883,-0.00519,0.0439,-0.0723,0.23022,-0.00868,0.04495,0.14143,0.25643,0.01116,0.00271,-0.05662,0.0083,-0.1481,0.14807,-0.12107,-0.18284,-0.05519,-0.09709,0.22231,0.06326,0.33843,-0.17826,-0.08646,0.1467,-0.08566,-0.10738,0.05124,-0.13281,-0.07252,0.10149,-0.04507,0.09073,0.19066,0.11175,0.10491,0.23562,-0.05335,-0.16161,0.1712,0.0581,0.01959,-0.25156,0.09225,0.17085,-0.01636,0.08548,0.20659,-0.16313],[0.17094,-0.00795,-0.19932,0.01085,-0.07327,-0.13486,0.01922,-0.26279,0.00171,0.17677,0.01805,-0.0908,0.13384,0.08204,0.01329,0.12317,-0.06386,0.21518,-0.08189,-0.06627,-0.1431,0.10171,-0.22211,-0.04823,0.36295,-0.16337,0.08497,0.03137,0.06725,0.02739,-0.1211,0.26271,0.15084,0.10427,-0.01929,0.11387,-0.20996,0.12259,-0.08824,0.00067,0.13882,0.07916,0.088,0.1002,-0.17718,0.08346,0.08161,-0.04572,-0.21872,-0.0344,0.04311,0.12163,0.04252,0.21981,-0.04534,-0.10705,-0.31156,0.05083,0.06552,-0.05664,-0.02327,-0.31323,-0.04981,0.00933],[0.19754,-0.17218,-0.14035,-0.01216,-0.18787,-0.11319,0.26406,0.06653,0.04765,0.22352,0.04396,0.079,-0.05922,0.09886,-0.13235,0.20271,0.1664,-0.04231,0.26774,0.0257,0.06967,-0.10056,-0.01173,0.1282,0.09041,-0.00415,0.06446,0.14175,-0.05552,-0.04327,-0.19949,0.11303,-0.04515,0.15957,0.19438,0.09127,-0.27836,-0.13925,0.06362,-0.03628,-0.07399,0.01364,-0.01681,0.17993,0.04693,0.26278,-0.15835,0.29636,-0.10216,0.09747,0.30299,0.09897,-0.23774,0.07515,0.17283,0.02649,0.05352,-0.17331,0.06156,-0.10705,-0.26481,-0.11415,-0.04962,-0.12082],[-0.13022,-0.08567,0.21797,0.10741,0.3254,-0.07171,0.00643,0.00025,-0.04077,-0.06757,0.10482,0.14979,0.21156,-0.1499,0.20398,-0.05523,-0.136,0.03943,-0.05263,0.26159,-0.24303,0.09899,-0.09612,-0.00759,-0.27135,0.01454,-0.09642,0.14333,-0.00208,-0.09377,-0.15808,-0.15842,-0.24803,0.03374,-0.11749,-0.04307,0.01208,-0.00228,0.03254,0.11202,-0.00648,-0.13768,0.00011,-0.20862,-0.03809,0.08893,-0.08075,0.12036,0.21476,0.20747,-0.27185,-0.12634,-0.05545,0.08803,-0.09956,0.18141,0.11577,-0.14036,-0.05376,-0.04595,-0.21546,-0.00938,0.24309,0.00138],[0.07427,-0.05794,-0.04811,-0.11178,-0.08074,0.28749,-0.0912,0.14174,0.03641,-0.02262,0.14661,0.0854,-0.02317,-0.01346,0.06666,0.09589,0.16944,-0.12297,-0.12532,-0.0624,-0.04309,-0.02027,0.20794,-0.059,-0.27671,-0.14518,-0.09238,-0.09608,0.23628,-0.30315,-0.17719,-0.10487,-0.01654,0.05141,0.01246,0.09554,-0.00506,-0.1921,0.03693,-0.09998,-0.33171,-0.11832,-0.04262,-0.05191,0.01997,0.07265,0.15279,0.12652,0.07866,0.1986,0.16807,-0.14304,-0.00018,-0.15858,-0.0587,0.1571,0.07123,-0.06558,-0.1445,0.12965,0.03524,0.14438,0.18857,0.06306],[0.07117,0.00399,0.2171,-0.3368,0.17634,-0.13642,-0.2453,-0.03616,0.00302,0.1266,-0.13067,-0.14025,-0.02414,-0.02241,-0.0604,0.02673,-0.1217,-0.06659,-0.02608,-0.27118,0.00905,-0.02855,-0.09941,0.354,0.0582,-0.09618,-0.1177,-0.01062,0.07874,0.05928,0.04354,-0.14236,-0.10435,0.11097,0.06559,0.08936,-0.07257,0.16576,-0.00682,-0.04462,-0.22573,0.12138,0.03458,0.12728,-0.03807,-0.09408,-0.0486,-0.04352,-0.09422,0.23554,0.12756,-0.13604,-0.0353,-0.22378,-0.27657,0.00545,0.14155,-0.03607,-0.06821,0.01452,-0.06035,0.00865,-0.1136,-0.0357],[-0.07462,-0.12532,-0.19651,-0.22268,-0.07247,0.04052,0.09555,-0.204,0.14055,0.1109,-0.09604,-0.06748,0.18848,-0.06882,0.1328,-0.00913,-0.03054,-0.2645,-0.05333,-0.102,-0.13572,0.03223,-0.03462,0.35798,-0.02365,0.01495,0.02541,-0.18297,-0.31758,-0.23418,-0.04279,0.12017,-0.12761,-0.03107,0.06231,0.3738,-0.08238,-0.02651,0.05102,0.04,-0.33312,0.0221,-0.04816,-0.01353,-0.11949,0.17453,0.07408,0.13924,0.12275,-0.05599,-0.06951,-0.12086,0.04379,0.0656,0.06754,-0.00987,0.04729,0.05967,-0.03902,-0.02682,0.01574,0.2118,0.13486,-0.04826],[-0.03062,-0.01853,-0.40193,0.2091,0.04383,-0.12078,0.08696,0.03627,-0.15505,0.14395,0.04892,-0.1617,0.31421,0.00773,-0.11373,0.09609,0.11637,0.11045,-0.32603,-0.06109,0.05365,0.02758,0.05829,-0.08795,-0.0248,-0.1393,0.16669,-0.03092,0.06887,0.09302,0.16153,0.06084,-0.01607,-0.08905,-0.18985,-0.07658,0.23563,-0.05229,-0.0222,-0.02764,0.16491,0.10631,-0.02912,0.12186,0.40582,-0.36949,0.03267,-0.06045,-0.03167,-0.03763,-0.09364,0.46046,-0.01625,0.08876,0.03199,0.14543,0.02331,0.07551,0.113,-0.06217,0.15338,-0.21895,0.06546,0.15166],[0.02137,0.01635,-0.03554,-0.38243,-0.03067,-0.045,-0.09137,0.04505,0.01885,0.09565,-0.09771,0.06636,0.00363,-0.09775,-0.08978,0.18438,-0.01445,-0.21042,0.16359,-0.09198,-0.09887,0.01705,-0.16988,-0.00358,-0.21402,-0.29774,-0.01786,-0.05165,0.15596,0.2373,0.02786,0.05448,-0.19391,-0.08371,0.07604,0.15809,0.02453,0.03007,0.04582,-0.02602,-0.21407,-0.05533,0.03472,0.05447,1e-05,0.14483,0.19501,0.06578,-0.06112,0.06072,-0.21069,0.03156,-0.03162,0.00027,-0.16839,-0.1513,-0.20721,-0.06878,0.43719,0.07499,0.21036,-0.0985,0.03546,0.07915],[0.19833,-0.24321,-0.0939,0.2378,0.18398,-0.07848,-0.04973,-0.13588,-0.04496,-0.09061,0.23156,0.17455,0.06814,0.19659,0.07159,0.02454,0.08008,0.17197,0.03861,-0.12598,-0.00995,0.03883,0.13158,-0.18222,-0.03063,0.20047,-0.02834,0.12028,-0.11305,-0.07358,-0.013,0.19333,0.07423,-0.14431,-0.16524,0.20754,0.09033,-0.08117,0.05689,-0.10152,0.10296,-0.12881,-0.02063,-0.15242,0.03909,0.07389,0.00092,0.02046,-0.09769,-0.09203,0.00662,0.15561,0.01857,-0.0081,-0.06405,0.03849,-0.14502,0.15152,-0.27244,-0.00851,0.00161,-0.05091,-0.0622,0.16073]],"b2":[-0.03921,0.02995,0.03567,0.0026,-0.00576,-0.01686,0.00406,0.01459,-0.01495,0.01359,0.01456,-0.02939,0.02541,0.01884,0.00882,0.02194,-0.02009,0.03444,0.04002,-0.02796,-0.03828,0.0802,-0.01724,0.05739,-0.01401,0.00781,-0.0263,-0.01366,0.03634,0.02144,0.02146,-0.01558,0.00012,0.01307,-0.03882,-0.04173,0.01187,0.05261,-0.00177,0.02352,-0.01917,0.05475,0.03869,-0.0076,-0.01686,-0.00365,0.00832,0.01632,-0.03493,-0.09801,-0.03741,-0.00221,0.00513,0.00641,0.04893,-0.05522,-0.05074,0.03292,0.05079,-0.01958,9e-05,-0.01563,-0.03259,0.00938],"Wp":[[-0.028,0.04082,-0.00364,-0.00481,-0.03203,-0.02,-0.00587,0.01675,-0.0167,0.00138,-0.01988,-0.01743,0.02744,0.00316,-0.00288,0.00406,-0.03477,0.06265,0.02912,-0.01104,-0.03688,0.01516,-0.0047,0.00545,0.00861,0.04562,-0.0483,-0.01588,0.04929,0.01971,-0.01295,-0.00977,-0.02568,0.02422,-0.02331,-0.03964,-0.01655,0.02118,-0.02386,0.01416,0.00443,-0.00024,0.02859,0.00297,-0.02375,-0.02376,0.01923,0.01871,-0.01511,-0.02619,-0.03204,-0.01069,-0.01543,0.02084,0.02818,-0.02008,-0.00934,0.02576,0.0246,-0.02865,0.01847,-0.01503,-0.03614,0.02473],[0.04629,-0.03013,-0.01029,0.03386,0.00648,-0.02128,-0.01415,-0.01703,0.01563,-0.02878,0.01925,-0.04195,-0.03662,-0.01337,0.02952,-0.01055,0.03982,-0.02462,-0.03747,-5e-05,0.04748,-0.00768,-0.00578,-0.00584,0.01334,-0.04086,0.03679,0.0485,-0.05573,-0.00897,-0.01368,0.0363,0.00609,-0.04511,0.02315,0.04021,0.02828,-0.01908,0.01508,-0.01714,-0.04458,-0.00054,-0.00242,0.0079,0.03173,0.03397,-0.04121,0.02729,0.03029,0.03125,0.03116,-0.00717,0.04138,-0.03078,-0.04589,0.04044,0.02023,-0.02601,-0.05828,0.01574,0.01163,-0.01408,-0.04456,-0.02843],[0.00741,-0.01354,-0.0288,-0.04782,0.04548,-0.0396,-0.05015,-0.01317,0.05677,-0.03926,0.03154,0.00938,-0.0427,-0.05357,0.02832,-0.05619,-0.02377,-0.00932,0.00336,0.03395,0.00518,-0.04393,-0.042,-0.04649,-0.0161,-0.00577,0.01634,0.02396,-0.03143,0.01846,-0.0435,0.03665,-0.02787,-0.02779,0.0279,0.03325,0.05368,-0.02151,0.0436,-0.03754,-0.04097,-0.05122,0.00923,0.03076,0.03703,0.04284,-0.04449,0.01514,0.05458,0.05888,0.0255,0.05028,0.04331,-0.05037,-0.01902,0.02162,0.04415,-0.04178,-0.01797,-0.02368,-0.02147,0.0365,-0.00477,0.01809],[-0.03839,0.00611,0.06656,0.05846,-0.04609,0.07056,0.04757,0.03128,-0.06364,0.08727,-0.04387,-0.03784,0.06922,0.07113,-0.06643,0.0703,0.02232,-0.02877,-0.01737,-0.07345,-0.01907,0.05868,0.05596,0.05642,0.04037,0.0199,-0.00151,-0.01509,0.03971,-0.01365,0.0704,-0.07256,0.05289,0.06113,-0.06585,-0.06298,-0.07197,0.01994,-0.06204,0.06368,0.06102,0.08889,-0.0163,-0.04782,-0.07058,-0.06231,0.0264,-0.05354,-0.0795,-0.07354,-0.05146,-0.03737,-0.07878,0.06038,0.05379,-0.0634,-0.06003,0.05534,0.04336,0.03004,0.00031,-0.03819,0.01773,-0.03178],[0.05943,-0.02599,-0.03867,-0.02772,0.0479,-0.01949,0.00491,-0.03642,0.02734,-0.05413,0.03787,0.0628,-0.0612,-0.03764,0.05368,-0.03496,0.02886,-0.02587,-0.01613,0.07194,0.05465,-0.04513,-0.01483,-0.0313,-0.03864,-0.04587,0.02689,-0.00638,-0.06086,-0.02887,-0.01994,0.04932,-0.01065,-0.05001,0.06854,0.05718,0.05381,-0.02352,0.05599,-0.06498,-0.02371,-0.06395,-0.02092,0.01721,0.06712,0.05055,0.01418,0.02465,0.04192,0.04067,0.06521,0.00366,0.05033,-0.02994,-0.06794,0.06617,0.03207,-0.04784,-0.05124,0.02489,-0.00305,0.02392,0.04856,0.00252]],"bp":[0.01829,-0.03485,-0.01988,0.06598,-0.06443],"Wv":[[-0.02799,-0.04667,0.03245,0.06788,-0.0696,0.09843,0.07217,0.02995,-0.0607,0.0432,-0.06919,-0.02992,0.0182,0.03671,-0.04864,0.06538,0.03553,-0.10091,-0.09786,-0.04693,0.01974,-0.09817,0.08231,-0.00404,0.0943,-0.06671,-0.02024,0.04735,-0.07351,-0.06852,0.0104,-0.03184,0.04954,0.03615,-0.02318,0.1012,-0.06295,-0.05395,-0.03573,0.03933,0.08126,-0.00495,-0.07824,-0.04317,-0.03383,-0.02777,0.04045,-0.08896,-0.03507,0.07605,-0.0213,-0.09541,-0.04294,0.07103,0.01084,-0.01135,0.05546,0.03175,0.01793,0.0595,0.08698,-0.06963,0.08889,-0.06075]],"bv":[0.00182]}
