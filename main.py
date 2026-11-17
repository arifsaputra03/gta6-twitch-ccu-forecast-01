"""gta6-twitch-ccu-forecast-01: Twitch concurrent-viewer forecast for GTA VI release day.

Model (fit fresh from the comparables embedded in each case file -- nothing is
hard-coded from the practice cases):

  The forecast is built in log space.  Let y1 be hour 1's print (the case's
  `known` row) and let A be the day's peak log-ratio, log(peak / y1).

  1. Amplitude.  Across past big launches the peak log-ratio falls as the
     hour-1 level rises -- a launch that is already huge at 00:00 UTC has
     little room left to climb, while a small hour-1 print (only New
     Zealand / Australia unlocked so far) can rise many-fold once Europe
     and the Americas wake up.  The relation is convex-decreasing, so A is
     predicted by a quadratic in log(y1), fit by least squares over all
     usable comparables and floored at 0 (a peak cannot sit below hour 1).

  2. Shape.  For each forecast hour H = 2..24, the log-ratio
     log(y_H / y1) is regressed on A over the big comparables only
     (hour-1 viewers above 20,000 -- the population a GTA VI-sized launch
     belongs to; small launches have far noisier shapes):

         log(y_H / y1) = alpha_H + beta_H * A

     The intercept carries the night dip (hours 2-4 fall below hour 1 even
     when A is 0, because the first unlock wave has faded before Europe
     wakes), the slope carries the unlock-wave rise (near 0 at night, about
     1.0 at the evening peak, hours 19-22), so the median curve dips, rises
     through the US unlocks, peaks in the evening and decays overnight.

  3. Spread.  The per-hour spread sigma_H is the leave-one-out residual
     standard deviation of that same regression, computed inside the
     program over the shape pool (each held-out launch is predicted from
     the rest, with the amplitude refit without it), times 0.9.  It is
     smallest at the peak hours (the peak tracks A almost exactly) and
     widest around midday (the rise's timing varies with unlock hours).
     Quantiles are lognormal: q_tau = exp(median + z_tau * sigma_H).

  Everything is deterministic, standard library only, no network, and
  runs in well under a second per case.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim.
It reads every <id>.in in the input directory and writes <id>.out with
exactly 23 rows (hours 2..24), five non-decreasing quantiles each, finite,
above 0 and at most 1e9, at most 65,536 bytes.  Outside the container,
MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Only comparables this big shape the hourly curve: a GTA VI-sized launch
# belongs with the big launches, whose shapes are far less noisy.
SHAPE_MIN_Y1 = 20000
# Smallest shape pool we trust; below it, fall back to every usable comparable.
SHAPE_MIN_N = 8
# Spread calibration: leave-one-out residual sd, shrunk slightly (the final
# fit uses all points, so it is a touch sharper than any leave-one-out fit).
SIGMA_MULT = 0.9
SIGMA_FLOOR = 0.15
SIGMA_CAP = 1.6
# Above this hour-1 level the quadratic amplitude is forced to 0: beyond the
# largest launch ever seen the peak is the hour-1 print itself.
HUGE_LOG_Y1 = 15.0


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def pstdev(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def solve(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination with pivoting."""
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) < 1e-12:
            raise ValueError("singular matrix")
        rows[col], rows[piv] = rows[piv], rows[col]
        for r in range(col + 1, n):
            f = rows[r][col] / rows[col][col]
            for c in range(col, n + 1):
                rows[r][c] -= f * rows[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (rows[i][n] - sum(rows[i][j] * out[j] for j in range(i + 1, n))) / rows[i][i]
    return out


def ols(design, target):
    """Least-squares coefficients for target ~ design rows (with intercept inside)."""
    p = len(design[0])
    xtx = [[sum(row[i] * row[j] for row in design) for j in range(p)] for i in range(p)]
    xty = [sum(row[i] * t for row, t in zip(design, target)) for i in range(p)]
    return solve(xtx, xty)


def round2(x):
    """x rounded to 2 decimals, half up (as JavaScript's Math.round(x*100)/100)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def case_hours(hourly, day_start):
    """A comparable's hourly points as {case hour 1..24: viewers}."""
    start = datetime.fromisoformat(day_start.replace("Z", "+00:00"))
    out = {}
    for point in hourly:
        when = datetime.fromisoformat(point["from"].replace("Z", "+00:00"))
        h = (when - start).total_seconds() / 3600.0
        if 0 <= h < 24:
            out[int(h) + 1] = point["viewers"]
    return out


def peak_log_ratio(y1, hours):
    peak = max(v for h, v in hours.items() if 1 <= h <= 24 and v)
    return math.log(peak / y1)


def usable_comparables(case):
    """Comparables with a usable full release day.  The target game's own
    pre-release history is not a launch curve, so it is excluded."""
    out = []
    for comp in case.get("comparables") or []:
        if "Grand Theft Auto VI" in comp.get("name", ""):
            continue
        try:
            hours = case_hours(comp.get("hourly") or [], comp.get("day_start", ""))
        except (ValueError, TypeError, KeyError):
            continue
        if 1 in hours and hours[1] and hours[1] > 0 and len(hours) >= 22:
            out.append({"name": comp.get("name", ""), "y1": hours[1], "hours": hours})
    return out


def fit_amplitude(comps):
    """Quadratic least squares: peak log-ratio ~ 1 + log y1 + (log y1)^2."""
    design = []
    target = []
    for comp in comps:
        x = math.log(max(comp["y1"], 1))
        design.append([1.0, x, x * x])
        target.append(peak_log_ratio(comp["y1"], comp["hours"]))
    try:
        return ols(design, target)
    except ValueError:
        # Singular design (too few or identical levels): fall back to a line,
        # and to a flat amplitude if even that is singular.
        try:
            design = [[1.0, math.log(max(comp["y1"], 1))] for comp in comps]
            return ols(design, target) + [0.0]
        except ValueError:
            return [0.0, 0.0, 0.0]


def predict_amplitude(y1, beta):
    x = math.log(max(y1, 1))
    if x > HUGE_LOG_Y1:
        return 0.0
    value = beta[0] + beta[1] * x + beta[2] * x * x
    return max(0.0, value)


def fit_shape(pool):
    """Per hour H: log(y_H / y1) ~ alpha_H + beta_H * A, over the pool.
    Returns {H: (alpha, beta, in-sample residual sd)}."""
    out = {}
    for hour in range(2, 25):
        xs = []
        ys = []
        for comp in pool:
            hours = comp["hours"]
            if hour in hours and hours[hour] and hours[hour] > 0:
                xs.append(peak_log_ratio(comp["y1"], hours))
                ys.append(math.log(hours[hour] / comp["y1"]))
        if len(xs) < 5:
            out[hour] = (0.0, 0.0, 0.6)
            continue
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 1e-9 else 0.0
        intercept = my - slope * mx
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        out[hour] = (intercept, slope, pstdev(resid))
    return out


def loo_spreads(pool, all_comps):
    """Leave-one-out residual sd per hour for the shape regression.  Each
    held-out launch is predicted from the rest of the pool, with the
    amplitude refit on the comparables without it."""
    resid = {hour: [] for hour in range(2, 25)}
    for target in pool:
        rest = [c for c in pool if c is not target]
        shape = fit_shape(rest)
        others = [c for c in all_comps if c is not target]
        beta = fit_amplitude(others) if len(others) >= 4 else fit_amplitude(all_comps)
        amp = predict_amplitude(target["y1"], beta)
        for hour in range(2, 25):
            hours = target["hours"]
            if hour in hours and hours[hour] and hours[hour] > 0:
                alpha, slope, _ = shape[hour]
                resid[hour].append(
                    math.log(hours[hour]) - (math.log(max(target["y1"], 1)) + alpha + slope * amp)
                )
    spreads = {}
    shape_full = fit_shape(pool)
    for hour in range(2, 25):
        if len(resid[hour]) >= 3:
            spreads[hour] = pstdev(resid[hour])
        else:
            spreads[hour] = shape_full[hour][2]
    return spreads


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    y1 = max(1, first["viewers"])
    log_y1 = math.log(y1)

    comps = usable_comparables(case)
    if len(comps) < 4:
        # Too little to fit anything: fall back to the house persistence baseline.
        rows = []
        for row in case["forecast"]:
            h = row["hour"]
            sig = 0.15 * math.sqrt(h - 1)
            rows.append({"hour": h, "q": [round2(y1 * math.exp(z * sig)) for z in Z]})
        return {"hours": rows}

    beta = fit_amplitude(comps)
    amp = predict_amplitude(y1, beta)

    pool = [c for c in comps if c["y1"] > SHAPE_MIN_Y1]
    if len(pool) < SHAPE_MIN_N:
        pool = comps
    shape = fit_shape(pool)
    spreads = loo_spreads(pool, comps)

    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        alpha, slope, _ = shape[h]
        med = log_y1 + alpha + slope * amp
        sig = min(SIGMA_CAP, max(SIGMA_FLOOR, SIGMA_MULT * spreads[h]))
        qs = []
        prev = 0.0
        for z in Z:
            q = math.exp(med + z * sig)
            q = max(1.0, min(1e9, q))
            q = round2(q)
            if q < prev:
                q = prev
            prev = q
            qs.append(q)
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # One bad case must not sink the rest: fall back to the
            # persistence forecast the rules score an invalid case as
            # (every quantile equal to hour 1's print).
            first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
            if first is None:
                raise
            y1 = round2(max(1.0, float(first["viewers"])))
            forecast = {
                "hours": [
                    {"hour": row["hour"], "q": [y1, y1, y1, y1, y1]}
                    for row in case.get("forecast", [])
                ]
            }
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
