# Branch and history audit

## Final branch policy

This repository has one published branch:

~~~text
main
~~~

The original repository already used the clean main name and had no
experiment, release, repair, or orx branches. No branch rename was needed.
main is the reproducible publication snapshot.

## Repository rename

| State | Repository |
| --- | --- |
| Original | MachineLearning-Nerd/icml26-repro-oUv02QKUxG-epistemic-mdp |
| Final | [MachineLearning-Nerd/icml26-dynamic-programming-epistemic-mdp](https://github.com/MachineLearning-Nerd/icml26-dynamic-programming-epistemic-mdp) |

The original main history had four commits:

| Original commit | Message | Rewritten commit |
| --- | --- | --- |
| 87df20d356ce112ad74de98dee50f96419e2a067 | Reproduce epistemic MDP dynamic-programming claims | 894592199b59beb898899e05c375b56aba778c06 |
| 06d465ec202a17da50d807a73e65994d71ca336b | Add final epistemic MDP logbook summary | d40ea284201db377899eee16d0141a670d9912b1 |
| 8cf4d28110c72441f84adc7dacb055ca7edc0c7c | Record epistemic MDP publication queue status | a717757285ad28319f61f71b9f69ada0bfaf2dda |
| 43b05137b7128c8a10882e0c91849f7f615375ef | Add exact Proposition 4.2 convergence audit | 962e5ca1f1caf9bed67f44120d4f6f55c70ae4bf |

All final reachable commits must use:

~~~text
MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>
~~~

The final publication tip and GitHub refs are checked by
[verify_final.py](verify_final.py), so the state is not inferred from the
branch name alone.

## Recovery record

Before rewriting identity history, a complete local bundle was created and
verified on 2026-08-15:

~~~text
/tmp/icml113-recovery.lWml6H/before-identity.bundle
~~~

The bundle contained the original main, origin/main, origin/HEAD, and HEAD
refs and passed git bundle verify. It remains outside the repository because
it contains the superseded author identities.
