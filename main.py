"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers, hours 2..24.

Model (fitted at run time from the case file's own comparables, stdlib only):
  Past game launches fall into two regimes. Type 1 ("rolling midnight"): the
  game is already live in early timezones by 00:00 UTC, hour 1 is large, and
  the day's curve decays from the midnight spike, dips in the UTC morning,
  then rises into the US-evening peak (hours ~17-21). Type 2 ("later
  unlock"): hour 1 is still the quiet pre-launch baseline and the big spike
  arrives only when the game unlocks later in the day, so later hours can be
  hundreds of times hour 1. Which regime a case is in is largely told by
  hour 1 itself: its size, and its size against the category's pre-release
  baseline. A two-feature logistic (log10 hour-1 viewers, log10 hour-1 /
  baseline) gives P(type 2). Within each regime, for every hour h the
  log-ratio log(viewers_h / viewers_1) is regressed on centered log10
  hour-1 viewers (bigger launches decay more by evening), with a robust
  residual spread; both are smoothed across neighbouring hours. The
  forecast quantiles are the quantiles of the resulting two-component
  normal mixture in log space, so a genuinely ambiguous case (hour 1 in
  the overlap zone) honestly reports a wide, bimodal spread instead of
  pretending to know. The case notes are also read: a "rolling midnight"
  unlock means the launch is already under way, so P(type 2) is capped.

  Backtest (leave-one-out over the kit's 37 full-day comparables):
  mean pinball loss 0.39 overall (type 1: 0.20, type 2: 0.74) against the
  house persistence baseline's 0.93 (type 1: 0.28, type 2: 2.26). The
  house baseline scores ~0.387 on the practice cases; this model should
  score ~0.20-0.25 there.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.
  - reads every <id>.in in the input directory (one case at a time);
  - writes <id>.out to the output directory: {"hours": [{"hour": 2,
    "q": [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24
    each once, each q five finite numbers above 0 and at most 1e9,
    non-decreasing; at most 65,536 bytes.
Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def phi(x: float) -> float:
    """The standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def smooth(vals: list, win: int) -> list:
    """Centered moving average (window win, odd); edges use what is there."""
    n = len(vals)
    half = win // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


def fit_logistic(X, ys, lam=1.0):
    """Ridge-regularised logistic regression by IRLS. X rows start with 1.0.

    The ridge keeps the coefficients finite on (nearly) separable data, so
    the type-2 probability stays honest in the overlap zone instead of
    snapping to 0/1.
    """
    p = len(X[0])
    a = [0.0] * p
    for _ in range(100):
        g = [0.0] * p
        H = [[0.0] * p for _ in range(p)]
        for x, y in zip(X, ys):
            z = max(-30.0, min(30.0, sum(ai * xi for ai, xi in zip(a, x))))
            pr = 1.0 / (1.0 + math.exp(-z))
            r = y - pr
            w = pr * (1.0 - pr) + 1e-9
            for i in range(p):
                g[i] += r * x[i]
                for j in range(p):
                    H[i][j] += w * x[i] * x[j]
        for i in range(1, p):
            g[i] -= lam * a[i]
            H[i][i] += lam
        H[0][0] += 1e-6
        # Solve H da = g by Gaussian elimination.
        M = [row[:] + [g[i]] for i, row in enumerate(H)]
        for col in range(p):
            piv = max(range(col, p), key=lambda r: abs(M[r][col]))
            M[col], M[piv] = M[piv], M[col]
            if abs(M[col][col]) < 1e-12:
                continue
            for r in range(p):
                if r != col and M[r][col] != 0.0:
                    f = M[r][col] / M[col][col]
                    for cc in range(col, p + 1):
                        M[r][cc] -= f * M[col][cc]
        da = [M[i][p] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(p)]
        for i in range(p):
            a[i] += da[i]
        if max(abs(d) for d in da) < 1e-8:
            break
    return a


def day_series(comp: dict) -> dict:
    """A comparable's release-day viewers, keyed by 1-based hour (1 = 00:00-01:00 UTC)."""
    ds = comp["day_start"][:10]
    out = {}
    for h in comp.get("hourly", []):
        if h["from"][:10] == ds:
            out[int(h["from"][11:13]) + 1] = h["viewers"]
    return out


def comp_base(comp: dict) -> float:
    """A comparable's pre-release baseline: median viewers before its release day."""
    ds = comp["day_start"][:10]
    pre = [h["viewers"] for h in comp.get("hourly", []) if h["from"][:10] < ds]
    return float(statistics.median(pre)) if pre else 1.0


def target_base(case: dict) -> float:
    """The target category's pre-checkpoint baseline: median of its history points."""
    vals = [h["viewers"] for h in case.get("history", []) or []]
    return float(statistics.median(vals)) if vals else 1.0


def rolling_midnight(notes) -> bool:
    """True when the case notes say the launch unlocks at midnight local, rolling.

    Then the game is already live in early timezones by 00:00 UTC, so the
    later-unlock regime is (nearly) ruled out however small hour 1 looks.
    """
    if not isinstance(notes, str):
        return False
    n = notes.lower()
    return "rolling" in n and ("midnight" in n or "local" in n)


# Fallback constants, fitted on the kit's comparables (2026-11-17): used
# only if a case carries too few usable comparables to fit at run time.
# Per-hour (hours 2..24) median log-ratio and spread, type 1 then type 2.
FALLBACK_M1 = [-0.35, -0.33, -0.30, -0.30, -0.34, -0.38, -0.40, -0.37, -0.31,
               -0.20, -0.10, -0.01, 0.08, 0.17, 0.23, 0.29, 0.32, 0.34, 0.37,
               0.32, 0.25, 0.16, 0.16]
FALLBACK_S1 = [0.30, 0.28, 0.34, 0.39, 0.46, 0.50, 0.49, 0.51, 0.50, 0.48,
               0.48, 0.49, 0.53, 0.55, 0.55, 0.58, 0.61, 0.59, 0.57, 0.54,
               0.56, 0.56, 0.56]
FALLBACK_M2 = [-0.30, -0.27, -0.19, -0.09, 0.16, 0.46, 0.78, 1.07, 1.48, 1.89,
               2.39, 2.95, 3.51, 3.92, 4.27, 4.49, 4.60, 4.66, 4.68, 4.61,
               4.52, 4.47, 4.41]
FALLBACK_S2 = [0.57, 0.64, 0.67, 0.78, 0.98, 1.25, 1.51, 1.79, 2.04, 2.17,
               2.17, 2.23, 2.28, 2.31, 2.34, 2.39, 2.36, 2.35, 2.36, 2.39,
               2.43, 2.47, 2.51]
FALLBACK_COEF = [10.287, -1.805, -1.134]  # w2 = sigmoid(c0 + c1*log10(h1) + c2*log10(h1/base))
FALLBACK_X1MEAN1 = 5.12
FALLBACK_X1MEAN2 = 3.54


class Forecaster:
    """Two-regime mixture forecaster, fitted from one case file's comparables."""

    def __init__(self, case: dict):
        series = []
        for comp in case.get("comparables", []) or []:
            s = day_series(comp)
            if 1 in s and len(s) >= 20 and s[1] > 0:
                series.append((s, comp_base(comp)))
        self.coef = FALLBACK_COEF
        self.m = {1: [(m, 0.0) for m in FALLBACK_M1], 2: [(m, 0.0) for m in FALLBACK_M2]}
        self.s = {1: FALLBACK_S1, 2: FALLBACK_S2}
        self.x1mean = {1: FALLBACK_X1MEAN1, 2: FALLBACK_X1MEAN2}
        if len(series) >= 8 and len({1 if max(s.values()) / s[1] > 4 else 0 for s, _ in series}) == 2:
            self._fit(series)
        self.rolling = rolling_midnight(case.get("notes"))

    def _fit(self, series):
        X = []
        ys = []
        t1 = []
        t2 = []
        for s, base in series:
            t = 2 if max(s.values()) / s[1] > 4 else 1
            x1 = math.log10(s[1])
            x2 = math.log10(s[1] / max(1.0, base))
            X.append([1.0, x1, x2])
            ys.append(1 if t == 2 else 0)
            (t2 if t == 2 else t1).append((s, base))
        if len(t1) < 4 or len(t2) < 3:
            return
        self.coef = fit_logistic(X, ys, lam=1.0)
        for t, sel in ((1, t1), (2, t2)):
            xm = statistics.mean([math.log10(s[1]) for s, _ in sel])
            self.x1mean[t] = xm
            inters = []
            slopes = []
            sds = []
            for h in range(2, 25):
                pts = [(math.log10(s[1]) - xm, math.log(s[h] / s[1]))
                       for s, _ in sel if h in s and s[h] > 0]
                if len(pts) >= 4:
                    n = len(pts)
                    mx = statistics.mean([p[0] for p in pts])
                    my = statistics.mean([p[1] for p in pts])
                    vx = sum((p[0] - mx) ** 2 for p in pts)
                    cov = sum((p[0] - mx) * (p[1] - my) for p in pts)
                    # Shrink the slope: with ~24 launches a raw slope overfits.
                    slope = (cov / vx) * (n / (n + 8.0)) if vx > 1e-9 else 0.0
                    inter = my - slope * mx
                    resid = [p[1] - (inter + slope * p[0]) for p in pts]
                    sd = max(0.15, statistics.pstdev(resid) if len(resid) > 1 else 0.5)
                    inters.append(inter)
                    slopes.append(slope)
                    sds.append(sd)
                else:
                    inters.append(inters[-1] if inters else 0.0)
                    slopes.append(slopes[-1] if slopes else 0.0)
                    sds.append(sds[-1] if sds else 0.5)
            win = 3 if t == 1 else 5
            sm = smooth(inters, win)
            self.m[t] = [(sm[i], slopes[i]) for i in range(len(sm))]
            self.s[t] = smooth(sds, win)

    def w2(self, h1: float, base: float) -> float:
        """P(later-unlock regime) from hour 1's size and its baseline ratio."""
        x1 = math.log10(max(1.0, h1))
        x2 = math.log10(max(1.0, h1) / max(1.0, base))
        z = self.coef[0] + self.coef[1] * x1 + self.coef[2] * x2
        z = max(-30.0, min(30.0, z))
        w = 1.0 / (1.0 + math.exp(-z))
        if self.rolling:
            w = min(w, 0.05)
        return w

    def quantiles(self, h1: float, base: float, h: int) -> list:
        """The five forecast quantiles (in log-ratio units) for hour h."""
        i = h - 2
        w = self.w2(h1, base)
        x1c = math.log10(max(1.0, h1))
        m1 = self.m[1][i][0] + self.m[1][i][1] * (x1c - self.x1mean[1])
        m2 = self.m[2][i][0] + self.m[2][i][1] * (x1c - self.x1mean[2])
        s1 = self.s[1][i]
        s2 = self.s[2][i]

        def F(q: float) -> float:
            return (1.0 - w) * phi((q - m1) / s1) + w * phi((q - m2) / s2)

        out = []
        for tau in LEVELS:
            lo, hi = -8.0, 14.0
            for _ in range(60):
                mid = (lo + hi) / 2.0
                if F(mid) < tau:
                    lo = mid
                else:
                    hi = mid
            out.append((lo + hi) / 2.0)
        return out

    def forecast(self, case: dict) -> dict:
        if case["levels"] != LEVELS:
            raise ValueError(f"levels {case['levels']}, not {LEVELS}")
        first = next((k for k in case["known"] if k["hour"] == 1), None)
        if first is None:
            raise ValueError("no hour-1 print")
        p1 = max(1.0, float(first["viewers"]))
        base = target_base(case)
        hours = []
        for row in case["forecast"]:
            h = row["hour"]
            qs = []
            prev = 0.0
            for q in self.quantiles(p1, base, h):
                v = p1 * math.exp(q)
                v = min(1e9, max(0.01, v))
                v = round2(v)
                if v < prev:
                    v = prev
                prev = v
                qs.append(v)
            hours.append({"hour": h, "q": qs})
        return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = Forecaster(case).forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
