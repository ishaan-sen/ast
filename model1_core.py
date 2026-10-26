import math

import numpy as np

PARAM_NAMES = ["rho", "tau", "q", "mu_F", "k", "h", "sigma"]

PARAM_LABELS = {
    "rho": "rho: season-to-season carryover",
    "tau": "tau: offseason change (points)",
    "q": "q: game-to-game drift (points)",
    "mu_F": "mu_F: Florida long-run level (points)",
    "k": "k: weight on opponent's prior SP+",
    "h": "h: home edge (points)",
    "sigma": "sigma: game noise (points)",
}


PRIORS = {
    "rho": ("beta", 10.0, 2.0),
    "tau": ("normal+", 8.7, 3.0),
    "q": ("halfnormal", 3.0),
    "mu_F": ("normal", 10.0, 10.0),
    "k": ("normal", 0.78, 0.10),
    "h": ("normal", 2.8, 1.0),
    "sigma": ("halfnormal", 20.0),
}


TRANSFORM = {
    "rho": "logit",
    "tau": "log",
    "q": "log",
    "sigma": "log",
    "mu_F": "none",
    "k": "none",
    "h": "none",
}

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def norm_cdf(z):

    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def log_prior_one(value, spec):
    kind = spec[0]
    if kind == "normal":
        m, s = spec[1], spec[2]
        return -0.5 * ((value - m) / s) ** 2 - math.log(s) - LOG_SQRT_2PI
    if kind == "halfnormal":
        if value <= 0:
            return -math.inf
        s = spec[1]
        return math.log(2.0) - 0.5 * (value / s) ** 2 - math.log(s) - LOG_SQRT_2PI
    if kind == "normal+":
        if value <= 0:
            return -math.inf
        m, s = spec[1], spec[2]
        mass = 1.0 - 0.5 * (1.0 + math.erf((-m / s) / math.sqrt(2.0)))
        return (
            -0.5 * ((value - m) / s) ** 2 - math.log(s) - LOG_SQRT_2PI - math.log(mass)
        )
    if kind == "beta":
        if not 0 < value < 1:
            return -math.inf
        a, b = spec[1], spec[2]
        log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        return (a - 1) * math.log(value) + (b - 1) * math.log(1 - value) - log_beta
    raise ValueError(f"unknown prior type {kind}")


def log_prior(theta: dict) -> float:
    return sum(log_prior_one(theta[p], PRIORS[p]) for p in PARAM_NAMES)


def prior_pdf(name, grid):

    return np.exp([log_prior_one(v, PRIORS[name]) for v in grid])


def state_transition_f(x, theta, new_season):

    if new_season:
        return theta["mu_F"] + theta["rho"] * (x - theta["mu_F"]), theta["rho"]
    return x, 1.0


def process_noise_Q(theta, new_season):

    return theta["tau"] ** 2 if new_season else theta["q"] ** 2


def observation_g(x, theta, opp_sp, site):

    return x - theta["k"] * opp_sp + theta["h"] * site, 1.0


def observation_noise_R(theta):

    return theta["sigma"] ** 2


def kalman_filter(theta, data, collect=False):

    x, P = data["start_strength"], 0.0
    loglik = 0.0
    n = len(data["margin"])
    if collect:
        out = {
            k: np.empty(n)
            for k in ("pred_mean", "pred_var", "x_prior", "P_prior", "x_post", "P_post")
        }

    for t in range(n):
        new = data["new_season"][t]

        fx, Fx = state_transition_f(x, theta, new)
        x_prior, P_prior = fx, Fx * Fx * P + process_noise_Q(theta, new)

        gx, Gx = observation_g(x_prior, theta, data["opp_sp"][t], data["site"][t])
        S = Gx * Gx * P_prior + observation_noise_R(theta)
        resid = data["margin"][t] - gx
        if data["fit"][t]:
            loglik += -0.5 * (math.log(2 * math.pi * S) + resid * resid / S)

        K = P_prior * Gx / S
        x, P = x_prior + K * resid, (1 - K * Gx) * P_prior
        if collect:
            out["pred_mean"][t], out["pred_var"][t] = gx, S
            out["x_prior"][t], out["P_prior"][t] = x_prior, P_prior
            out["x_post"][t], out["P_post"][t] = x, P

    if collect:
        out["x_last"], out["P_last"] = x, P
        return loglik, out
    return loglik


def to_theta(u):
    theta = {}
    for i, p in enumerate(PARAM_NAMES):
        tr = TRANSFORM[p]
        theta[p] = (
            1 / (1 + math.exp(-u[i]))
            if tr == "logit"
            else math.exp(u[i])
            if tr == "log"
            else u[i]
        )
    return theta


def to_unconstrained(theta):
    u = []
    for p in PARAM_NAMES:
        tr, v = TRANSFORM[p], theta[p]
        u.append(
            math.log(v / (1 - v))
            if tr == "logit"
            else math.log(v)
            if tr == "log"
            else v
        )
    return np.array(u)


def log_jacobian(u, theta):
    total = 0.0
    for i, p in enumerate(PARAM_NAMES):
        if TRANSFORM[p] == "logit":
            total += math.log(theta[p] * (1 - theta[p]))
        elif TRANSFORM[p] == "log":
            total += u[i]
    return total


def log_posterior(u, data):
    try:
        theta = to_theta(u)
    except OverflowError:
        return -math.inf
    lp = log_prior(theta)
    if not math.isfinite(lp):
        return -math.inf
    return kalman_filter(theta, data) + lp + log_jacobian(u, theta)


def prior_center():

    start = {}
    for p, spec in PRIORS.items():
        kind = spec[0]
        start[p] = (
            spec[1] / (spec[1] + spec[2])
            if kind == "beta"
            else spec[1] * 0.8
            if kind == "halfnormal"
            else spec[1]
        )
    return start


INIT_STEP = {
    "rho": 0.3,
    "tau": 0.3,
    "q": 0.5,
    "mu_F": 3.0,
    "k": 0.05,
    "h": 0.7,
    "sigma": 0.05,
}


def _random_start(base, data, rng):

    while True:
        u = base + rng.normal(0, 0.1, len(base))
        lp = log_posterior(u, data)
        if math.isfinite(lp):
            return u, lp


def run_mcmc(
    data, n_chains=4, n_burn=3000, n_keep=3000, thin=3, seed=1, progress=print
):

    rng = np.random.default_rng(seed)
    d = len(PARAM_NAMES)
    base = to_unconstrained(prior_center())
    n_save = n_keep // thin
    draws = np.empty((n_chains, n_save, d))
    accept_rates = []

    for c in range(n_chains):
        u, lp = _random_start(base, data, rng)
        cov = np.diag([INIT_STEP[p] ** 2 for p in PARAM_NAMES])
        scale = 1.0
        history, recent, accepted, kept = [], [], 0, 0
        for it in range(n_burn + n_keep):
            prop = rng.multivariate_normal(u, scale * cov)
            lp_prop = log_posterior(prop, data)
            ok = math.log(rng.random()) < lp_prop - lp
            if ok:
                u, lp = prop, lp_prop
            if it < n_burn:
                history.append(u.copy())
                recent.append(ok)
                if len(recent) == 100:
                    scale *= math.exp(np.mean(recent) - 0.234)
                    recent = []
                if it >= 500 and (it + 1) % 250 == 0:
                    emp = np.cov(np.array(history[len(history) // 2 :]).T)
                    cov = (2.38**2 / d) * emp + np.eye(d) * 1e-8
                    scale = 1.0
            else:
                accepted += ok
                if (it - n_burn) % thin == 0 and kept < n_save:
                    th = to_theta(u)
                    draws[c, kept] = [th[p] for p in PARAM_NAMES]
                    kept += 1
        accept_rates.append(accepted / n_keep)
        progress(
            f"    chain {c + 1}/{n_chains} done (acceptance {accepted / n_keep:.0%})"
        )
    return draws, accept_rates


def split_rhat(chains):

    n = chains.shape[1] // 2
    seqs = np.concatenate([chains[:, :n], chains[:, n : 2 * n]])
    W = seqs.var(axis=1, ddof=1).mean()
    B = n * seqs.mean(axis=1).var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def draws_to_thetas(draws, n_max, seed=1):

    flat = draws.reshape(-1, draws.shape[-1])
    idx = np.linspace(0, len(flat) - 1, min(n_max, len(flat))).astype(int)
    return [dict(zip(PARAM_NAMES, flat[i])) for i in idx]


def filter_all_draws(thetas, data):

    S, n = len(thetas), len(data["margin"])
    res = {k: np.empty((S, n)) for k in ("pred_mean", "pred_var", "x_post", "P_post")}
    res["x_last"], res["P_last"] = np.empty(S), np.empty(S)
    for s, th in enumerate(thetas):
        _, out = kalman_filter(th, data, collect=True)
        for k in ("pred_mean", "pred_var", "x_post", "P_post"):
            res[k][s] = out[k]
        res["x_last"][s], res["P_last"][s] = out["x_last"], out["P_last"]
    return res


def forecast_upcoming(thetas, filtered, upcoming):

    S, m = len(thetas), len(upcoming["opp_sp"])
    mean, var = np.empty((S, m)), np.empty((S, m))
    for s, th in enumerate(thetas):
        x, P = filtered["x_last"][s], filtered["P_last"][s]
        for j in range(m):
            new = upcoming["new_season"][j]
            fx, Fx = state_transition_f(x, th, new)
            x, P = fx, Fx * Fx * P + process_noise_Q(th, new)
            gx, Gx = observation_g(x, th, upcoming["opp_sp"][j], upcoming["site"][j])
            mean[s, j], var[s, j] = gx, Gx * Gx * P + observation_noise_R(th)
    p_win = norm_cdf(mean / np.sqrt(var))
    return {"mean": mean, "var": var, "p_win_draws": p_win, "p_win": p_win.mean(axis=0)}
