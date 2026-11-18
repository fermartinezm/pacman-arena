Pacman CTF heuristic for Windows x64
===================================

Requires 64-bit CPython 3.12 and an existing Pacman CTF contest
installation that runs under that Python version.

1. Extract the ZIP next to capture.py in the engine's contest directory.
   Keep heuristic_team.py and the _pac_heuristic_native folder together.
2. Open PowerShell in that directory and run:

   py -3.12 -m contest.capture -r heuristic_team -b baseline_team -q -f -l RANDOM23

   This plays the heuristic on red on a fixed map and seed. Use
   -r baseline_team -b heuristic_team to test blue. Replace baseline_team
   with your agent's filename without .py, and change RANDOM23 to test maps.
   If your contest installation uses a virtual environment, activate it
   and use its python instead of py -3.12.

The only Python file is the contest loader. The heuristic and its six
helpers are compiled into native .pyd modules with renamed internal
methods and classes. A determined analyst can still reverse engineer
the binaries.
