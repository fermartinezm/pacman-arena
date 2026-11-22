"""Play games and print one JSON line per game with what happened to TEAM (deaths, food lost, capsule, ...).
   python tools/loss_stats.py TEAM OPP LAYOUT SEEDS(a-b)   (plays both colours)"""
import sys, os, random, io, json, contextlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import evaluate
import contest.capture as capture, contest.text_display as td
from concurrent.futures import ProcessPoolExecutor


def play(job):
    team, opp, lay, seed, team_red = job
    red, blue = (team, opp) if team_red else (opp, team)
    random.seed(seed)
    L = evaluate.make_layout(lay, seed)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ra = capture.load_agents(True, red, {}); ba = capture.load_agents(False, blue, {})
        agents = [ra[0], ba[0], ra[1], ba[1]]
        game = capture.CaptureRules(quiet=True).new_game(L, agents, td.NullGraphics(), 1200, True, True)
        ours = [0, 2] if team_red else [1, 3]
        theirs = [1, 3] if team_red else [0, 2]
        S = dict(deaths=0, lost=0, kills=0, killed_carry=0, first_death=None, our_cap_lost=None, their_cap_eaten=None,
                 max_their_carry=0, max_our_carry=0, our_pac_time=0, their_pac_time=0, both_in_time=0,
                 no_home_guard_time=0, score_t={})
        prev = [None] * 4
        step = [0]
        orig = game.state

        def hook(state):
            st = state
            t = step[0]
            for i in range(4):
                a = st.data.agent_states[i]
                pos = a.get_position()
                p0 = prev[i]
                if p0 is not None and pos == st.get_initial_agent_position(i) and abs(p0[0] - pos[0]) + abs(p0[1] - pos[1]) > 1:
                    if i in ours:
                        S['deaths'] += 1; S['lost'] += p0c[i]
                        S['first_death'] = S['first_death'] if S['first_death'] is not None else t // 4
                    else:
                        S['kills'] += 1; S['killed_carry'] += p0c[i]
                prev[i] = pos
                p0c[i] = a.num_carrying
            for i in ours:
                S['max_our_carry'] = max(S['max_our_carry'], st.data.agent_states[i].num_carrying)
            for i in theirs:
                S['max_their_carry'] = max(S['max_their_carry'], st.data.agent_states[i].num_carrying)
            S['our_pac_time'] += sum(st.data.agent_states[i].is_pacman for i in ours)
            tp = sum(st.data.agent_states[i].is_pacman for i in theirs)
            S['their_pac_time'] += tp
            S['both_in_time'] += tp == 2
            if tp and all(st.data.agent_states[i].is_pacman or st.data.agent_states[i].scared_timer > 0 for i in ours):
                S['no_home_guard_time'] += 1
            caps_red, caps_blue = st.get_red_capsules(), st.get_blue_capsules()
            our_caps = caps_red if team_red else caps_blue
            their_caps = caps_blue if team_red else caps_red
            if S['our_cap_lost'] is None and len(our_caps) < n_caps[0]:
                S['our_cap_lost'] = t // 4
            if S['their_cap_eaten'] is None and len(their_caps) < n_caps[1]:
                S['their_cap_eaten'] = t // 4
            if t % 100 == 0:
                sc = st.data.score if team_red else -st.data.score
                S['score_t'][t // 4] = sc
            step[0] += 1

        p0c = [0] * 4
        st0 = game.state
        n_caps = (len(st0.get_red_capsules() if team_red else st0.get_blue_capsules()),
                  len(st0.get_blue_capsules() if team_red else st0.get_red_capsules()))
        # run the game, calling hook after every move
        import contest.game as gm
        orig_gen = type(game.state).generate_successor
        game.run_hook = hook
        # simplest: replay move_history afterwards
        game.run()
        st = st0
        for (i, a) in game.move_history:
            st = st.generate_successor(i, a)
            hook(st)
    score = game.state.data.score if team_red else -game.state.data.score
    S.update(score=score, moves=len(game.move_history), layout=lay, seed=seed, colour='red' if team_red else 'blue',
             opp=os.path.basename(opp), banked=sum(game.state.data.agent_states[i].num_returned for i in ours),
             conceded=sum(game.state.data.agent_states[i].num_returned for i in theirs))
    return S


if __name__ == '__main__':
    team, opp, lay, seeds = sys.argv[1:5]
    a, b = map(int, seeds.split('-'))
    jobs = [(team, opp, lay, s, c) for s in range(a, b + 1) for c in (True, False)]
    with ProcessPoolExecutor(int(os.environ.get('JOBS', os.cpu_count()))) as ex:
        for r in ex.map(play, jobs):
            print('STAT ' + json.dumps(r), flush=True)
