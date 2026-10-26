FIT_LAST_SEASON = 2024

N_CHAINS = 4
N_BURN = 3000
N_KEEP = 6000
THIN = 3
N_PRED_DRAWS = 1000
SEED = 2026

import math
import sys
from datetime import date
from pathlib import Path

try:
    import numpy as np
    import pandas as pd

    import model1_core as core
    import model1_plots as plots
except ImportError as e:
    print(
        f"Missing package: {e.name}\nInstall with:  pip install numpy pandas matplotlib"
    )
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"


def find_data_dir() -> Path:
    candidates = [Path(a) for a in sys.argv[1:]] + [
        SCRIPT_DIR.parent / "Data" / "florida_data",
        SCRIPT_DIR / "florida_data",
    ]
    for p in candidates:
        if (p / "model_games.csv").exists():
            return p
    while True:
        p = Path(
            input("Paste the path to the florida_data folder: ").strip().strip('"')
        )
        if (p / "model_games.csv").exists():
            return p
        print("model_games.csv not found there (run prep_florida_model_data.py first).")


def load_data(data_dir: Path):
    games = (
        pd.read_csv(data_dir / "model_games.csv")
        .sort_values("game_idx")
        .reset_index(drop=True)
    )
    upcoming = pd.read_csv(data_dir / "model_predict.csv")
    seasons = pd.read_csv(data_dir / "model_seasons.csv")
    games["row"] = np.arange(len(games))
    upcoming["date"] = pd.to_datetime(upcoming["date"])
    upcoming = upcoming.sort_values("date").reset_index(drop=True)

    stale = upcoming[upcoming["date"].dt.date < date.today()]
    if not stale.empty:
        print("  NOTE: these games are in the past but have no result in the data:")
        for _, r in stale.iterrows():
            print(f"        {r['date']:%Y-%m-%d} {r['opponent']}")
        print(
            "        Re-run florida_cfb_data.py and prep_florida_model_data.py to include them."
        )
    return games, upcoming, seasons


def build_model_inputs(games, seasons, upcoming):
    first = int(games["season"].min())
    start = float(seasons.loc[seasons["season"] == first, "prev_sp"].iat[0])
    data = {
        "margin": games["margin"].astype(float).tolist(),
        "opp_sp": games["opp_prior_sp"].astype(float).tolist(),
        "site": games["site"].astype(float).tolist(),
        "new_season": (games["season"] != games["season"].shift()).tolist(),
        "fit": (games["season"] <= FIT_LAST_SEASON).tolist(),
        "start_strength": start,
    }
    prev = [int(games["season"].iat[-1])] + upcoming["season"].astype(int).tolist()[:-1]
    up = {
        "opp_sp": upcoming["opp_prior_sp"].astype(float).tolist(),
        "site": upcoming["site"].astype(float).tolist(),
        "new_season": [int(s) != p for s, p in zip(upcoming["season"], prev)],
    }
    return data, up


def fit_parameters(data):
    draws, acc = core.run_mcmc(
        data, n_chains=N_CHAINS, n_burn=N_BURN, n_keep=N_KEEP, thin=THIN, seed=SEED
    )
    rhats = {p: core.split_rhat(draws[:, :, i]) for i, p in enumerate(core.PARAM_NAMES)}
    return draws, acc, rhats


def summarize_parameters(draws, rhats):
    flat = draws.reshape(-1, draws.shape[-1])
    rows = []
    for i, p in enumerate(core.PARAM_NAMES):
        x = flat[:, i]
        spec = core.PRIORS[p]
        rows.append(
            {
                "parameter": p,
                "meaning": core.PARAM_LABELS[p].split(": ", 1)[1],
                "prior": f"{spec[0]}({', '.join(f'{a:g}' for a in spec[1:])})",
                "mean": x.mean(),
                "sd": x.std(),
                "q05": np.percentile(x, 5),
                "median": np.median(x),
                "q95": np.percentile(x, 95),
                "rhat": rhats[p],
            }
        )
    return pd.DataFrame(rows)


def predict_played_games(games, filtered):
    m, v = filtered["pred_mean"], filtered["pred_var"]
    p_win = core.norm_cdf(m / np.sqrt(v)).mean(axis=0)
    pred_mean = m.mean(axis=0)
    pred_sd = np.sqrt(v.mean(axis=0) + m.var(axis=0))
    t = games[
        [
            "game_idx",
            "season",
            "date",
            "opponent",
            "location",
            "site",
            "margin",
            "win",
            "opp_prior_sp",
            "vegas_fla_margin",
        ]
    ].copy()
    t["pred_margin"] = pred_mean
    t["pred_sd"] = pred_sd
    t["pred_lo80"] = pred_mean - 1.2816 * pred_sd
    t["pred_hi80"] = pred_mean + 1.2816 * pred_sd
    t["p_win"] = p_win
    t["strength_after"] = filtered["x_post"].mean(axis=0)
    t["split"] = np.where(t["season"] <= FIT_LAST_SEASON, "fit", "test")
    return t


def scores(p, w):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    w = np.asarray(w, float)
    return {
        "games": len(w),
        "log_loss": float(np.mean(-(w * np.log(p) + (1 - w) * np.log(1 - p)))),
        "brier": float(np.mean((p - w) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == (w == 1))),
    }


def vegas_sigma(table):

    d = table[(table["split"] == "fit") & table["vegas_fla_margin"].notna()]
    return float((d["margin"] - d["vegas_fla_margin"]).std()) if len(d) > 10 else 15.4


def evaluate(table, sigma_v):
    rows = []
    for split in ("fit", "test"):
        d = table[(table["split"] == split) & table["win"].notna()]
        r = scores(d["p_win"], d["win"])
        r.update(
            split=split,
            model="Model 1",
            margin_mae=float((d["margin"] - d["pred_margin"]).abs().mean()),
        )
        rows.append(r)
        lines = d[d["vegas_fla_margin"].notna()]
        if len(lines):
            pv = core.norm_cdf(lines["vegas_fla_margin"] / sigma_v)
            if len(lines) < len(d):
                r1 = scores(lines["p_win"], lines["win"])
                r1.update(
                    split=split,
                    model="Model 1, games with a line",
                    margin_mae=float(
                        (lines["margin"] - lines["pred_margin"]).abs().mean()
                    ),
                )
                rows.append(r1)
            r2 = scores(pv, lines["win"])
            r2.update(
                split=split,
                model="Vegas, same games",
                margin_mae=float(
                    (lines["margin"] - lines["vegas_fla_margin"]).abs().mean()
                ),
            )
            rows.append(r2)
    cols = ["split", "model", "games", "log_loss", "brier", "accuracy", "margin_mae"]
    return pd.DataFrame(rows)[cols]


def game_label(r):
    where = {"home": "vs", "away": "at", "neutral": "vs"}[r["location"]]
    tail = " (neutral)" if r["location"] == "neutral" else ""
    return f"{r['date']:%b %d}  {where} {r['opponent']}{tail}"


def predict_upcoming(upcoming, fc, sigma_v):
    m, v = fc["mean"], fc["var"]
    sd = np.sqrt(v.mean(axis=0) + m.var(axis=0))
    t = upcoming[
        [
            "date",
            "season",
            "opponent",
            "location",
            "site",
            "opp_prior_sp",
            "vegas_fla_margin",
        ]
    ].copy()
    t["label"] = t.apply(game_label, axis=1)
    t["p_win"] = fc["p_win"]
    t["p_win_lo"] = np.percentile(fc["p_win_draws"], 5, axis=0)
    t["p_win_hi"] = np.percentile(fc["p_win_draws"], 95, axis=0)
    t["expected_margin"] = m.mean(axis=0)
    t["margin_lo80"] = t["expected_margin"] - 1.2816 * sd
    t["margin_hi80"] = t["expected_margin"] + 1.2816 * sd
    t["vegas_p_win"] = core.norm_cdf(t["vegas_fla_margin"].fillna(np.nan) / sigma_v)
    t.loc[t["vegas_fla_margin"].isna(), "vegas_p_win"] = np.nan
    return t


def write_outputs(params, table, metrics, up, acc, sigma_v, data_dir):
    OUTPUT_DIR.mkdir(exist_ok=True)
    params.to_csv(OUTPUT_DIR / "parameter_summary.csv", index=False)
    table.to_csv(OUTPUT_DIR / "game_predictions.csv", index=False)
    up.drop(columns=["label"]).to_csv(
        OUTPUT_DIR / "upcoming_predictions.csv", index=False
    )

    with pd.option_context(
        "display.width",
        160,
        "display.max_columns",
        20,
        "display.float_format",
        "{:.3f}".format,
    ):
        text = [
            f"MODEL 1 RESULTS   run {date.today().isoformat()}",
            f"data: {data_dir}",
            (
                f"games: {len(table)} completed ({int((table['split'] == 'fit').sum())} fit through {FIT_LAST_SEASON}, "
                f"{int((table['split'] == 'test').sum())} test), {len(up)} upcoming"
            ),
            (
                f"MCMC: {N_CHAINS} chains x {N_KEEP} kept (thin {THIN}) after {N_BURN} burn-in; "
                f"acceptance {', '.join(f'{a:.0%}' for a in acc)}"
            ),
            "",
            "PARAMETERS (posterior)",
            params.to_string(index=False),
            "",
            f"ACCURACY (Vegas probabilities use margin ~ Normal(Vegas line, {sigma_v:.1f}))",
            "log loss: lower is better (coin flip = 0.693); brier: lower is better (coin flip = 0.25)",
            metrics.to_string(index=False),
            "",
            "UPCOMING GAMES",
            up[
                [
                    "label",
                    "p_win",
                    "p_win_lo",
                    "p_win_hi",
                    "expected_margin",
                    "margin_lo80",
                    "margin_hi80",
                    "vegas_p_win",
                ]
            ].to_string(index=False),
        ]
    (OUTPUT_DIR / "results.txt").write_text("\n".join(text), encoding="utf-8")


def make_figures(draws, rhats, games, filtered, seasons, table, up, fc):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plots.setup_style()
    figs = [
        (
            "1_parameter_posteriors.png",
            lambda p: plots.plot_parameter_posteriors(draws, p),
        ),
        (
            "2_florida_strength.png",
            lambda p: plots.plot_strength(games, filtered, seasons, FIT_LAST_SEASON, p),
        ),
        (
            "3_upcoming_win_probability.png",
            lambda p: plots.plot_upcoming_win_prob(up, p),
        ),
        (
            "4_upcoming_margin_distributions.png",
            lambda p: plots.plot_margin_distributions(up, fc, p),
        ),
        (
            "5_calibration.png",
            lambda p: plots.plot_calibration(table, FIT_LAST_SEASON, p),
        ),
        (
            "6_predicted_vs_actual.png",
            lambda p: plots.plot_predicted_vs_actual(table, p),
        ),
        ("7_mcmc_traces.png", lambda p: plots.plot_traces(draws, rhats, p)),
    ]
    for name, fn in figs:
        fn(FIG_DIR / name)
    return [name for name, _ in figs]


def main():
    print("Model 1: Florida win probability from a latent-strength model\n")
    data_dir = find_data_dir()
    print(f"[1/7] Loading data from {data_dir}")
    games, upcoming, seasons = load_data(data_dir)

    print(
        f"[2/7] Building model inputs: {len(games)} completed games, {len(upcoming)} upcoming"
    )
    data, up_inputs = build_model_inputs(games, seasons, upcoming)

    print(f"[3/7] Learning parameters with MCMC (games through {FIT_LAST_SEASON})")
    draws, acc, rhats = fit_parameters(data)
    params = summarize_parameters(draws, rhats)
    worst = max(rhats.values())
    print(
        f"      worst R-hat {worst:.3f}"
        + ("  (OK)" if worst < 1.05 else "  (chains disagree: raise N_KEEP)")
    )

    print("[4/7] Predicting every completed game one step ahead")
    thetas = core.draws_to_thetas(draws, N_PRED_DRAWS)
    filtered = core.filter_all_draws(thetas, data)
    table = predict_played_games(games, filtered)

    print("[5/7] Evaluating against the test seasons and Vegas")
    sigma_v = vegas_sigma(table)
    metrics = evaluate(table, sigma_v)

    print("[6/7] Predicting upcoming games")
    fc = core.forecast_upcoming(thetas, filtered, up_inputs)
    up = predict_upcoming(upcoming, fc, sigma_v)

    print("[7/7] Saving results and figures")
    write_outputs(params, table, metrics, up, acc, sigma_v, data_dir)
    names = make_figures(draws, rhats, games, filtered, seasons, table, up, fc)

    print("\nParameters (posterior mean, 90% interval):")
    for _, r in params.iterrows():
        print(
            f"  {r['parameter']:<6} {r['mean']:7.2f}   ({r['q05']:.2f} to {r['q95']:.2f})"
        )
    test = metrics[metrics["split"] == "test"]
    print("\nTest seasons:")
    print(test.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nUpcoming games:")
    for _, r in up.iterrows():
        vegas = (
            f"   Vegas {r['vegas_p_win']:.0%}"
            if not math.isnan(r["vegas_p_win"])
            else ""
        )
        print(
            f"  {r['label']:<32} P(win) {r['p_win']:.0%}   expected margin {r['expected_margin']:+.1f}{vegas}"
        )
    print(
        f"\nSaved to {OUTPUT_DIR}:  results.txt, 3 CSV files, figures/ ({len(names)} PNGs)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"\nSomething went wrong: {e}")
    if sys.stdin and sys.stdin.isatty():
        try:
            input("\nPress Enter to close.")
        except EOFError:
            pass
