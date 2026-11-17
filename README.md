# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arifsaputra03/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arifsaputra03/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Twitch concurrent viewers in the **Grand Theft Auto VI** category
for every UTC hour of release day, 19 November 2026 — from the hour-1 print
alone, plus the launch curves of past big games embedded in each case file.

Bounty: https://make.fun/bounty/gta6-twitch-ccu-forecast-01

## The model

`main.py` (Python 3.13, standard library only, no network) fits everything
fresh from the comparables inside the case it is given — nothing is
hard-coded from the practice cases. It works in log space.

1. **Amplitude.** Across past launches the day's peak log-ratio,
   log(peak / hour-1), falls as the hour-1 level rises: a launch already
   huge at 00:00 UTC has little room left to climb, while a small hour-1
   print (only New Zealand / Australia unlocked so far) can rise many-fold
   once Europe and the Americas wake up. The relation is convex-decreasing,
   so the amplitude is predicted by a quadratic in log(hour-1), fit over
   all usable comparables and floored at 0.

2. **Shape.** For each forecast hour H = 2..24, the log-ratio
   log(y_H / y1) is regressed on the amplitude over the big comparables
   only (hour-1 viewers above 20,000 — the population a GTA VI-sized
   launch belongs to):

   `log(y_H / y1) = alpha_H + beta_H * A`

   The intercept carries the night dip (hours 2–4 fall below hour 1 even
   when the amplitude is 0, because the first unlock wave fades before
   Europe wakes); the slope carries the unlock-wave rise (near 0 at night,
   about 1.0 at the evening peak, hours 19–22). The median curve dips,
   rises through the US unlocks, peaks in the evening and decays overnight.

3. **Spread.** The per-hour spread is the leave-one-out residual standard
   deviation of that same regression, computed inside the program (each
   held-out launch is predicted from the rest, with the amplitude refit
   without it), times 0.9. It is smallest at the peak hours (the peak
   tracks the amplitude almost exactly) and widest around midday (the
   rise's timing varies with unlock hours). Quantiles are lognormal:
   `q_tau = exp(median + z_tau * sigma_H)`.

## Practice results

Mean pinball loss on the log scale (lower is better), four practice cases:

| case | this model | house starter |
| --- | --- | --- |
| p00 Marvel's Wolverine | 0.186 | 0.744 |
| p01 Red Dead Redemption 2 | 0.232 | 0.287 |
| p02 God of War Ragnarök | 0.180 | 0.215 |
| p03 Marvel's Spider-Man 2 | 0.191 | 0.303 |
| **overall** | **0.197** | **0.387** |

Typical miss 33–53% (starter: 42–189%); 90% coverage 0.87–1.00
(starter: 0.52–0.91). The biggest win is on the small-hour-1 launches,
where persistence misses the multi-fold evening rise entirely.

The model was chosen by leave-one-out cross-validation over the
comparables (not by tuning to the four practice cases): the quadratic
amplitude and the per-hour shape both beat their simpler alternatives
there, and the bias correction that helped inside the comparables was
dropped because it did not transfer to the console-only practice cases.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` with
exactly 23 rows (hours 2..24), five non-decreasing quantiles each.
Deterministic (two runs are byte-identical) and fast (~0.1 s per case).

## License

MIT — see [LICENSE](LICENSE). SPDX-License-Identifier: MIT
