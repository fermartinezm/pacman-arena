"""Play one game and print what happens (positions, food carried/banked, deaths, scares)."""
import sys, os, random, io, contextlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import evaluate
import contest.capture as capture, contest.text_display as td
red, blue, lay, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
every = int(sys.argv[5]) if len(sys.argv) > 5 else 50
random.seed(seed)
L = evaluate.make_layout(lay, seed)
with contextlib.redirect_stdout(io.StringIO()):
    ra = capture.load_agents(True, red, {}); ba = capture.load_agents(False, blue, {})
agents = [ra[0], ba[0], ra[1], ba[1]]
game = capture.CaptureRules(quiet=True).new_game(L, agents, td.NullGraphics(), 1200, True, False)
st = game.state
starts = [st.get_initial_agent_position(i) for i in range(4)]
for a in agents:
    a.register_initial_state(st.deep_copy())
eaten = [0]*4; deaths = [0]*4; pac_time = [0]*4; prev = [None]*4
idx = 0; step = 0
err_sum = [0, 0]
print('red =', red, ' blue =', blue, ' agents 0,2 red / 1,3 blue')
while not st.is_over() and step < 1200:
    obs = agents[idx].observation_function(st.deep_copy())
    act = agents[idx].get_action(obs)
    if hasattr(agents[idx], 'real_obs'):
        for o in agents[idx].get_opponents(st):
            est, true = obs.get_agent_position(o), st.get_agent_position(o)
            if est is not None and st.data.agent_states[o].is_pacman:
                err_sum[0] += abs(est[0]-true[0]) + abs(est[1]-true[1]); err_sum[1] += 1
                if step % every < 4 and idx == min(agents[idx].get_team(st)):
                    print(f'   t={step} track enemy{o} est {est} true {true} carrying {st.data.agent_states[o].num_carrying}')
    old = st
    st = st.generate_successor(idx, act)
    for j in range(4):
        a0, a1 = old.data.agent_states[j], st.data.agent_states[j]
        if a1.num_carrying > a0.num_carrying:
            eaten[j] += a1.num_carrying - a0.num_carrying
        p = st.get_agent_position(j)
        if p == starts[j] and prev[j] is not None and abs(prev[j][0]-p[0]) + abs(prev[j][1]-p[1]) > 1:
            deaths[j] += 1
            print(f't={step} agent{j} died at {prev[j]} carrying {a0.num_carrying} score={st.data.score}')
        if a1.scared_timer > a0.scared_timer:
            print(f't={step} agent{j} scared (capsule eaten) score={st.data.score}')
        prev[j] = p
        if a1.is_pacman:
            pac_time[j] += 1
    if step % every == 0:
        s = ' | '.join(f"a{j}{'P' if st.data.agent_states[j].is_pacman else 'G'}{st.get_agent_position(j)}c{st.data.agent_states[j].num_carrying}r{st.data.agent_states[j].num_returned}s{st.data.agent_states[j].scared_timer}" for j in range(4))
        print(f't={step} score={st.data.score} {s}')
    idx = (idx + 1) % 4; step += 1
print('FINAL score', st.data.score, 'invader tracking error (manhattan avg)', err_sum[0] / max(1, err_sum[1]))
for j in range(4):
    print(f'agent{j}: eaten {eaten[j]} returned {st.data.agent_states[j].num_returned} deaths {deaths[j]} pacman-moves {pac_time[j]//4}')
