# Methods


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_984c3fe2c4c5", "created_at": "2026-07-17T08:43:02+00:00", "title": "Independent construction"}
-->
The positive support is the full Cartesian product of two transition choices for every state-action row. Bellman operators follow equations (7)–(8). Every fixed global kernel is solved by (I-gamma P)^-1 r; this is independent of Bellman iteration.


---
<!-- trackio-cell
{"type": "code", "id": "cell_023a99298d49", "created_at": "2026-07-17T08:43:38+00:00", "title": "Unit and control tests", "command": ["python", "-m", "pytest", "-q", "repro/tests"], "exit_code": 0, "duration_s": 4.85}
-->
````bash
$ python -m pytest -q repro/tests
````

exit 0 · 4.8s


````output
............                                                             [100%]
12 passed in 4.11s

````
