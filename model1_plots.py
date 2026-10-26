import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model1_core as core

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3de"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
LOSS_GRAY = "#c9c7bf"
CHAIN_COLORS = [BLUE, ORANGE, AQUA, YELLOW]


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": INK,
            "font.size": 9.5,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.dpi": 110,
        }
    )


def _save(fig, path, title, subtitle=None, bottom=0.0):

    h = fig.get_figheight()
    top = 1 - (0.85 if subtitle else 0.55) / h
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.text(0.012, 1 - 0.12 / h, title, ha="left", va="top", fontsize=13, color=INK)
    if subtitle:
        fig.text(
            0.012, 1 - 0.45 / h, subtitle, ha="left", va="top", fontsize=9, color=INK_2
        )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_posteriors(draws, path):

    flat = draws.reshape(-1, draws.shape[-1])
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for i, (ax, name) in enumerate(zip(axes.flat, core.PARAM_NAMES)):
        x = flat[:, i]
        lo, hi = np.percentile(x, [0.5, 99.5])
        pad = (hi - lo) * 0.6
        grid = np.linspace(lo - pad, hi + pad, 300)
        if name in ("rho",):
            grid = grid[(grid > 0) & (grid < 1)]
        if name in ("tau", "q", "sigma"):
            grid = grid[grid > 0]
        ax.hist(
            x,
            bins=45,
            density=True,
            color=BLUE,
            alpha=0.35,
            edgecolor="none",
            label="posterior",
        )
        ax.plot(grid, core.prior_pdf(name, grid), color=ORANGE, label="prior")
        ax.set_title(core.PARAM_LABELS[name], fontsize=9.5, loc="left")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.text(
            0.98,
            0.95,
            f"mean {x.mean():.2f}\n90%: {np.percentile(x, 5):.2f} to {np.percentile(x, 95):.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=INK_2,
        )
    last = axes.flat[-1]
    last.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    last.legend(handles, labels, loc="center", fontsize=10)
    _save(
        fig,
        path,
        "Model 1 parameters: what the data learned",
        "Blue: posterior from Florida's games. Orange: the prior pi(theta) before seeing them.",
    )


def plot_strength(games, filtered, seasons, fit_last_season, path, seed=1):

    rng = np.random.default_rng(seed)
    x_post, P_post = filtered["x_post"], filtered["P_post"]
    samples = rng.normal(x_post, np.sqrt(np.maximum(P_post, 1e-9)))
    mean = x_post.mean(axis=0)
    lo, hi = np.percentile(samples, [5, 95], axis=0)
    idx = np.arange(len(games))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(
        idx, lo, hi, color=BLUE, alpha=0.12, linewidth=0, label="90% interval"
    )
    ax.plot(idx, mean, color=BLUE, label="Model 1 strength (after each game)")

    last_game = games.groupby("season")["row"].max()
    bench = seasons.set_index("season")["bench_final_sp"].reindex(last_game.index)
    ax.scatter(
        last_game.values,
        bench.values,
        s=34,
        color=ORANGE,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=3,
        label="SP+ at season end (benchmark, not used)",
    )

    split_row = games.loc[games["season"] > fit_last_season, "row"].min()
    if not math.isnan(split_row):
        ax.axvline(split_row - 0.5, color=INK_2, linewidth=1)
        ax.text(
            split_row + 1,
            ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 1,
            "test seasons",
            color=INK_2,
            fontsize=8.5,
            va="top",
        )
    first_game = games.groupby("season")["row"].min()
    ticks = [first_game[s] for s in first_game.index if s % 4 == 0]
    ax.set_xticks(ticks, [str(s) for s in first_game.index if s % 4 == 0])
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_ylabel("points better than an average FBS team")
    ax.set_xlim(-2, len(games) + 2)
    ax.legend(loc="lower left", ncol=3)
    _save(
        fig,
        path,
        "Florida's latent strength, 1992 to now",
        "Kalman-filtered strength after each game, averaged over the parameter posterior. "
        "0 = an average FBS team.",
    )


def plot_upcoming_win_prob(up, path):

    n = len(up)
    fig, ax = plt.subplots(figsize=(9, 0.7 * n + 1.8))
    y = np.arange(n)[::-1]
    p = up["p_win"].values * 100
    lo, hi = up["p_win_lo"].values * 100, up["p_win_hi"].values * 100
    ax.barh(y, p, height=0.42, color=BLUE, label="Model 1 win probability")
    ax.errorbar(
        p,
        y,
        xerr=[p - lo, hi - p],
        fmt="none",
        ecolor=INK_2,
        elinewidth=1.2,
        capsize=3,
        label="90% range from parameter uncertainty",
    )
    vegas = up["vegas_p_win"].values * 100
    has = ~np.isnan(vegas)
    ax.scatter(
        vegas[has],
        y[has],
        marker="D",
        s=40,
        color=ORANGE,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=4,
        label="Vegas (from the point spread)",
    )
    for yi, pi, hii in zip(y, p, hi):
        ax.text(
            max(pi, hii) + 1.5, yi, f"{pi:.0f}%", va="center", fontsize=9, color=INK
        )
    ax.set_yticks(y, up["label"].values)
    ax.axvline(50, color=INK_2, linewidth=0.8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("probability Florida wins (%)")
    ax.grid(axis="y", visible=False)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
    )
    _save(
        fig,
        path,
        "Upcoming games: probability Florida wins",
        "Bars average over the parameter posterior; whiskers show how much they move across it.",
        bottom=0.75 / fig.get_figheight(),
    )


def plot_margin_distributions(up, fc, path, seed=1):

    rng = np.random.default_rng(seed)
    n = len(up)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.3 * cols, 3.2 * rows), squeeze=False
    )
    bins = np.arange(-70, 72, 2)
    for j in range(rows * cols):
        ax = axes.flat[j]
        if j >= n:
            ax.axis("off")
            continue
        m = fc["mean"][:, j]
        sd = np.sqrt(fc["var"][:, j])
        sims = rng.normal(np.repeat(m, 20), np.repeat(sd, 20))
        counts, edges = np.histogram(sims, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        win = centers > 0
        ax.bar(centers[win], counts[win], width=1.7, color=BLUE, label="Florida wins")
        ax.bar(
            centers[~win],
            counts[~win],
            width=1.7,
            color=LOSS_GRAY,
            label="Florida loses",
        )
        ax.axvline(0, color=INK, linewidth=1)
        v = up["vegas_fla_margin"].iat[j]
        if not np.isnan(v):
            ax.axvline(v, color=ORANGE, linewidth=2, label="Vegas expected margin")
        ax.axvline(
            m.mean(),
            color=INK_2,
            linewidth=1.2,
            linestyle=(0, (3, 2)),
            label="model expected margin",
        )
        ax.set_title(
            f"{up['label'].iat[j]}\nP(win) {up['p_win'].iat[j]:.0%}, "
            f"expected margin {m.mean():+.1f}",
            fontsize=9.5,
            loc="left",
        )
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Florida margin (points)")
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    _save(
        fig,
        path,
        "Predicted margin distributions for upcoming games",
        "Blue area = probability Florida wins. Includes game-to-game noise and parameter uncertainty.",
        bottom=0.5 / fig.get_figheight(),
    )


def plot_calibration(table, fit_last_season, path):

    edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, label="perfect calibration")
    first = int(table["season"].min())
    for name, color, sel in (
        (
            f"{first}-{fit_last_season} (parameters fit on these)",
            BLUE,
            table["split"] == "fit",
        ),
        (
            f"{fit_last_season + 1} onward (out of sample)",
            ORANGE,
            table["split"] == "test",
        ),
    ):
        d = table[sel & table["win"].notna()]
        if d.empty:
            continue
        b = np.clip(np.digitize(d["p_win"], edges) - 1, 0, len(edges) - 2)
        g = d.groupby(b).agg(p=("p_win", "mean"), w=("win", "mean"), n=("win", "size"))
        ax.plot(
            g["p"],
            g["w"],
            color=color,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=name,
        )
        for _, r in g.iterrows():
            ax.annotate(
                f"{int(r['n'])}",
                (r["p"], r["w"]),
                textcoords="offset points",
                xytext=(7, 4),
                fontsize=7.5,
                color=INK_2,
            )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("predicted probability Florida wins")
    ax.set_ylabel("share of those games Florida actually won")
    ax.legend(loc="upper left")
    _save(
        fig,
        path,
        "Calibration of win probabilities",
        "Games grouped in 20%-wide bands; small numbers = games in the band.",
    )


def plot_predicted_vs_actual(table, path):

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, color, sel, size in (
        ("fit seasons", BLUE, table["split"] == "fit", 18),
        ("test seasons", ORANGE, table["split"] == "test", 40),
    ):
        d = table[sel]
        ax.scatter(
            d["pred_margin"],
            d["margin"],
            s=size,
            color=color,
            alpha=0.75 if name == "fit seasons" else 1,
            edgecolor=SURFACE,
            linewidth=0.8,
            label=f"{name} (n={len(d)})",
        )
    lim = [-75, 80]
    ax.plot(lim, lim, color=INK_2, linewidth=1)
    ax.axhline(0, color=GRID, linewidth=1.2)
    ax.axvline(0, color=GRID, linewidth=1.2)
    ax.set_xlim(-45, 55)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted Florida margin before the game (points)")
    ax.set_ylabel("actual Florida margin (points)")
    ax.legend(loc="upper left")
    r = np.corrcoef(table["pred_margin"], table["margin"])[0, 1]
    _save(
        fig,
        path,
        "Predicted vs. actual margin",
        f"Correlation {r:.2f}. Points above the line = Florida did better than predicted.",
    )


def plot_traces(draws, rhats, path):

    fig, axes = plt.subplots(4, 2, figsize=(13, 9), sharex=True)
    for i, (ax, name) in enumerate(zip(axes.flat, core.PARAM_NAMES)):
        for c in range(draws.shape[0]):
            ax.plot(
                draws[c, :, i],
                color=CHAIN_COLORS[c % 4],
                linewidth=0.6,
                alpha=0.8,
                label=f"chain {c + 1}",
            )
        ax.set_title(
            f"{core.PARAM_LABELS[name]}   (R-hat {rhats[name]:.3f})",
            fontsize=9,
            loc="left",
        )
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", ncol=2, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("saved draw")
    _save(
        fig,
        path,
        "MCMC diagnostics: trace plots",
        "R-hat near 1.00 (under about 1.05) means the chains agree.",
    )
