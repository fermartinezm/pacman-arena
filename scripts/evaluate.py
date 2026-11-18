#!/usr/bin/env python3
"""
Quick headless evaluation of a team: plays many games with fixed seeds,
without graphics, optionally in parallel, and reports results and timing.

It does not replace the official commands (runner.py / capture.py); it only
uses the contest framework to play games faster and reproducibly, so changes
to the agents can be measured and compared.

Run from the repository root:

    python scripts/evaluate.py
        my_team.py (blue) vs baseline_team (red), 10 seeds on defaultCapture
    python scripts/evaluate.py -n 30 -l defaultCapture,RANDOM --swap
        30 seeds per layout, playing both as red and as blue
    python scripts/evaluate.py -r baseline -b my_idea
        any two teams: a strategy folder name in agents/, a team file,
        or a team in the contest folder (baseline_team)
    python scripts/evaluate.py --league
        every strategy in agents/ and baseline_team play each other,
        both colours, and a standings table is printed
    python scripts/evaluate.py --league baseline,my_idea,my_team.py -n 20
        a league between the listed teams only
    python scripts/evaluate.py --digest
        also print a fingerprint of every game's moves; two versions of the
        code that print the same fingerprint played exactly the same games

Every game is seeded with its seed number, so the same seed, layout and
teams always give the same game. Layout RANDOM uses the seed as the maze
seed too (RANDOM<seed>).
"""

import argparse
import contextlib
import hashlib
import io
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTEST_SRC = os.path.join(ROOT, 'pacman-contest', 'src')
CONTEST_DIR = os.path.join(CONTEST_SRC, 'contest')
AGENTS_DIR = os.path.join(ROOT, 'agents')

# Use the contest package from this repository even when it is not installed.
if CONTEST_SRC not in sys.path:
    sys.path.insert(0, CONTEST_SRC)

MOVE_WARNING_TIME = 1.0  # seconds per move before the contest gives a warning
STARTUP_TIME = 15.0  # seconds allowed for register_initial_state


def resolve_team(team):
    """
    Return an absolute path for a team: a strategy in agents/ (e.g. baseline),
    a folder holding a my_team.py, a file path, or a team in the contest
    folder (e.g. baseline_team).
    """
    for base in (AGENTS_DIR, os.getcwd(), ROOT):
        candidate = os.path.join(base, team, 'my_team.py')
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    path = team if team.endswith('.py') else team + '.py'
    for base in (os.getcwd(), ROOT, CONTEST_DIR):
        candidate = os.path.join(base, path)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    sys.exit(f'Team not found: {team}')


def strategies():
    """Names of the strategy folders in agents/ (those with a my_team.py), sorted."""
    if not os.path.isdir(AGENTS_DIR):
        return []
    return sorted(name for name in os.listdir(AGENTS_DIR)
                  if not name.startswith(('.', '_'))
                  and os.path.isfile(os.path.join(AGENTS_DIR, name, 'my_team.py')))


def team_name(path):
    """Short name of a team for reports: the strategy folder name, or the path from the repository root."""
    if os.path.dirname(os.path.dirname(path)) == AGENTS_DIR and os.path.basename(path) == 'my_team.py':
        return os.path.basename(os.path.dirname(path))
    if os.path.dirname(path) == CONTEST_DIR:
        return os.path.splitext(os.path.basename(path))[0]
    return os.path.relpath(path, ROOT)


def make_layout(name, seed):
    import contest.capture as capture
    import contest.layout as layout
    if name == 'RANDOM':
        layout_name, layout_text = capture.random_layout(seed=seed)
        return layout.Layout(layout_name=layout_name, layout_text=layout_text.split('\n'))
    if name.startswith('RANDOM'):
        layout_name, layout_text = capture.random_layout(seed=int(name[6:]))
        return layout.Layout(layout_name=layout_name, layout_text=layout_text.split('\n'))
    path = name if name.endswith('.lay') else name + '.lay'
    for candidate in (path, os.path.join(CONTEST_DIR, 'layouts', path)):
        if os.path.isfile(candidate):
            with open(candidate) as f:
                return layout.Layout(layout_name=os.path.basename(candidate),
                                     layout_text=[line.strip() for line in f])
    raise ValueError(f'Layout not found: {name}')


def timed(times, func):
    """Wrap an agent method so each call's duration is appended to `times`."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            times.append(time.perf_counter() - start)
    return wrapper


def play_game(job):
    """Play one game and return its result as a dict (runs in a worker process)."""
    red, blue, layout_name, seed, length = job
    import contest.capture as capture
    import contest.text_display as text_display

    random.seed(seed)
    game_layout = make_layout(layout_name, seed)
    # Silence the framework's own prints (team loading, who starts, ...).
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        red_agents = capture.load_agents(True, red, {})
        blue_agents = capture.load_agents(False, blue, {})
    if None in red_agents or None in blue_agents:
        return {'seed': seed, 'layout': layout_name, 'error': 'team failed to load:\n' + err.getvalue()}
    agents = [red_agents[0], blue_agents[0], red_agents[1], blue_agents[1]]

    init_times = [[] for _ in agents]
    move_times = [[] for _ in agents]
    for i, agent in enumerate(agents):
        agent.register_initial_state = timed(init_times[i], agent.register_initial_state)
        agent.get_action = timed(move_times[i], agent.get_action)

    rules = capture.CaptureRules(quiet=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        game = rules.new_game(game_layout, agents, text_display.NullGraphics(), length,
                              mute_agents=True, catch_exceptions=True)
        game.run()

    return {
        'seed': seed,
        'layout': game_layout.layout_name,
        'score': game.state.data.score,
        'moves': len(game.move_history),
        'crashed': game.agent_crashed,
        'timeout': game.agent_timeout,
        'stderr': err.getvalue().strip(),
        'digest': hashlib.sha1(repr(game.move_history).encode()).hexdigest()[:12],
        'init_times': [max(t, default=0.0) for t in init_times],
        'move_times': move_times,
    }


def parse_seeds(text):
    """'10' -> seeds 1..10, '5-8' -> 5..8, '1,4,9' -> those seeds."""
    if '-' in text:
        first, last = text.split('-')
        return list(range(int(first), int(last) + 1))
    if ',' in text:
        return [int(s) for s in text.split(',')]
    return list(range(1, int(text) + 1))


def run_jobs(jobs, n_jobs):
    """Play all jobs, in parallel when asked, and return their results in order."""
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            return list(pool.map(play_game, jobs))
    return [play_game(job) for job in jobs]


def league(teams, args):
    """Play every pair of teams with both colours and print a standings table."""
    if len(set(teams)) < 2:
        sys.exit('A league needs at least two different teams.')
    jobs = [(r, b, layout, seed, args.time)
            for r in teams for b in teams if r != b
            for layout in args.layouts.split(',')
            for seed in parse_seeds(args.seeds)]
    start = time.perf_counter()
    results = run_jobs(jobs, args.jobs)
    elapsed = time.perf_counter() - start

    # stats[team] = [wins, ties, losses, score sum, crashes, slowest move]
    stats = {team: [0, 0, 0, 0, 0, 0.0] for team in teams}
    pairs = {}  # (team, opponent) -> [wins, ties, losses]
    for (r, b, *_), res in zip(jobs, results):
        if 'error' in res:
            print(f'{team_name(r)} vs {team_name(b)}, seed {res["seed"]} on {res["layout"]}: {res["error"]}')
            continue
        for side, team, other in ((0, r, b), (1, b, r)):
            score = res['score'] if side == 0 else -res['score']
            outcome = 0 if score > 0 else 1 if score == 0 else 2
            stats[team][outcome] += 1
            stats[team][3] += score
            moves = res['move_times'][side] + res['move_times'][side + 2]
            stats[team][5] = max(stats[team][5], max(moves, default=0))
            pairs.setdefault((team, other), [0, 0, 0])[outcome] += 1
        if res['crashed']:
            # The framework only tells that some agent crashed or timed out;
            # blame the side that lost the game by forfeit.
            stats[r if res['score'] < 0 else b][4] += 1

    names = {team: team_name(team) for team in teams}
    width = max(12, *(len(n) for n in names.values()))
    ranking = sorted(teams, key=lambda t: (-(stats[t][0] + stats[t][1] / 2), -stats[t][3]))
    print(f'{"team":<{width}} {"games":>5} {"wins":>5} {"ties":>5} {"losses":>6} {"win%":>5} '
          f'{"avg score":>9} {"crashes":>7} {"max move":>9}')
    for team in ranking:
        w, t, l, total, crashes, slowest = stats[team]
        played = w + t + l
        if not played:
            continue
        print(f'{names[team]:<{width}} {played:>5} {w:>5} {t:>5} {l:>6} {100 * w / played:>4.0f}% '
              f'{total / played:>+9.2f} {crashes:>7} {slowest * 1000:>7.1f}ms')

    print('\nHead to head (wins-ties-losses of the row team):')
    print(' ' * width + ''.join(f' {names[t]:>{width}}' for t in ranking))
    for team in ranking:
        cells = ['-' if team == other else '%d-%d-%d' % tuple(pairs.get((team, other), (0, 0, 0)))
                 for other in ranking]
        print(f'{names[team]:<{width}}' + ''.join(f' {c:>{width}}' for c in cells))
    print(f'\n{len(jobs)} games in {elapsed:.1f}s')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-r', '--red', default='baseline_team', help='red team (default: baseline_team)')
    parser.add_argument('-b', '--blue', default='my_team.py', help='blue team (default: my_team.py)')
    parser.add_argument('-l', '--layouts', default='defaultCapture',
                        help='comma-separated layouts, RANDOM = random maze per seed (default: defaultCapture)')
    parser.add_argument('-n', '--seeds', default='10',
                        help="games per layout: '10' = seeds 1..10, '5-8' or '1,4,9' (default: 10)")
    parser.add_argument('-i', '--time', type=int, default=1200, help='game length in moves (default: 1200)')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 1,
                        help='parallel processes (default: all CPUs)')
    parser.add_argument('--swap', action='store_true', help='also play every game with the colours swapped')
    parser.add_argument('--digest', action='store_true', help='print a fingerprint of each game and of all games')
    parser.add_argument('--league', nargs='?', const='', metavar='TEAMS',
                        help='play every pair of the comma-separated teams, both colours (default: '
                             'baseline_team and every strategy in agents/); ignores -r and -b')
    args = parser.parse_args()

    if args.league is not None:
        names = args.league.split(',') if args.league else ['baseline_team'] + strategies()
        league([resolve_team(name) for name in names], args)
        return

    red, blue = resolve_team(args.red), resolve_team(args.blue)
    pairings = [(red, blue)] + ([(blue, red)] if args.swap else [])
    jobs = [(r, b, layout, seed, args.time)
            for r, b in pairings
            for layout in args.layouts.split(',')
            for seed in parse_seeds(args.seeds)]

    start = time.perf_counter()
    results = run_jobs(jobs, args.jobs)
    elapsed = time.perf_counter() - start

    # Report from the point of view of the team under test (blue by default).
    team = blue
    wins = ties = losses = 0
    team_scores, team_moves, team_inits, all_digests = [], [], [], []
    print(f'{"side":<5} {"layout":<22} {"seed":>5} {"score":>6} {"result":<7} {"max move":>9}  notes')
    for (r, b, *_), res in zip(jobs, results):
        if 'error' in res:
            print(f'seed {res["seed"]} on {res["layout"]}: {res["error"]}')
            continue
        side = 0 if r == team else 1  # agent indices of the team: side, side + 2
        score = res['score'] if side == 0 else -res['score']
        wins, ties, losses = wins + (score > 0), ties + (score == 0), losses + (score < 0)
        team_scores.append(score)
        moves = res['move_times'][side] + res['move_times'][side + 2]
        team_moves.extend(moves)
        team_inits.extend(res['init_times'][side::2])
        all_digests.append(res['digest'])
        notes = []
        if res['crashed']:
            notes.append('TIMEOUT' if res['timeout'] else 'CRASH')
        slow = sum(t > MOVE_WARNING_TIME for t in moves)
        if slow:
            notes.append(f'{slow} moves over {MOVE_WARNING_TIME:.0f}s')
        if args.digest:
            notes.append(res['digest'])
        result = 'win' if score > 0 else 'tie' if score == 0 else 'loss'
        print(f'{("red", "blue")[side]:<5} {res["layout"]:<22} {res["seed"]:>5} {score:>6} {result:<7} '
              f'{max(moves, default=0) * 1000:>7.1f}ms  {" ".join(notes)}')
        if res['crashed'] and res['stderr']:
            print('      ' + res['stderr'].replace('\n', '\n      '))

    played = wins + ties + losses
    if not played:
        return
    team_moves.sort()
    print(f'\nTeam {team_name(team)}: {wins} wins, {ties} ties, {losses} losses '
          f'of {played} games ({100 * wins / played:.0f}% wins), average score {sum(team_scores) / played:+.2f}')
    print(f'Move time: mean {1000 * sum(team_moves) / len(team_moves):.2f}ms, '
          f'p99 {1000 * team_moves[int(0.99 * (len(team_moves) - 1))]:.2f}ms, '
          f'max {1000 * team_moves[-1]:.2f}ms (limit {MOVE_WARNING_TIME:.0f}s); '
          f'max startup {max(team_inits):.2f}s (limit {STARTUP_TIME:.0f}s)')
    if args.digest:
        print('All games digest:', hashlib.sha1(' '.join(all_digests).encode()).hexdigest()[:12])
    print(f'{len(jobs)} games in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
