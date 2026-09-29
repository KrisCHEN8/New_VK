import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
import conditional_dist as cd
from conditional_dist import REGIMES


# tunable parameters: conditional_dist.CONFIG plus the overrides/additions below
CONFIG = cd._merge_config({

    # model terms: "x" linear, "x^p" power, "a*b" interaction (all on centred covariates)
    "hum_col": "RH_sup",
    "mean_terms": ["RH_sup", "RH_sup^2", "T_out_3h", "solar_acc"],
    "scale_terms": ["RH_sup", "T_out_3h"],
    "ref_round": {"RH_sup": 0, "T_out_3h": 0, "solar_acc": -1},   # decimals for reference values

    # plots
    "hum_bin_width": 5.0,   # %RH, for the empirical comparison
    "min_bin_count": 50,
})


# Design matrices
def _base_cols(terms):
    """Covariates used by 'terms' (e.g. 'RH_sup^2' -> RH_sup, 'a*b' -> a, b)."""
    out = []
    for t in terms:
        for c in t.split("^")[0].split("*"):
            if c not in out:
                out.append(c)
    return out


def _design(d, terms, ref):
    """Design matrix with an intercept and one column per term, covariates centred at 'ref'."""
    X = {"const": np.ones(len(d))}
    for t in terms:
        if "*" in t:
            a, b = t.split("*")
            X[t] = (d[a] - ref[a]).to_numpy() * (d[b] - ref[b]).to_numpy()
        elif "^" in t:
            base, p = t.split("^")
            X[t] = (d[base] - ref[base]).to_numpy() ** int(p)
        else:
            X[t] = (d[t] - ref[t]).to_numpy()
    return pd.DataFrame(X, index=d.index)


def reference_values(d, cols, config=None):
    """Rounded medians used to centre the covariates (shared by all regimes)."""
    cfg = cd._merge_config(config, CONFIG)
    return {c: float(np.round(d[c].median(), cfg["ref_round"].get(c, 1))) for c in cols}


# Model
def _nll(theta, y, X, Z):
    """Negative log-likelihood (without the constant) and its gradient."""
    p = X.shape[1]
    beta, gamma = theta[:p], theta[p:]
    log_s = Z @ gamma
    s = np.exp(log_s)
    r = (y - X @ beta) / s
    nll = np.sum(log_s + 0.5 * r ** 2)
    grad = np.concatenate([-X.T @ (r / s), Z.T @ (1.0 - r ** 2)])
    return nll, grad


def fit_hetero_gaussian(y, X, Z, groups):
    """
    Maximum-likelihood fit of y ~ N(X @ beta, exp(Z @ gamma)^2).
    Columns are scaled internally for conditioning; the results are in the original units.
    Returns params (beta, gamma as Series), their cluster-robust covariance, and fit statistics.
    """
    y = np.asarray(y, dtype=float)
    sx = X.std().replace(0, 1.0).to_numpy()
    sz = Z.std().replace(0, 1.0).to_numpy()
    Xn, Zn = X.to_numpy() / sx, Z.to_numpy() / sz
    p, q = Xn.shape[1], Zn.shape[1]

    # start from OLS with a constant spread
    beta0 = np.linalg.lstsq(Xn, y, rcond=None)[0]
    gamma0 = np.zeros(q)
    gamma0[0] = np.log(np.std(y - Xn @ beta0))
    res = optimize.minimize(_nll, np.concatenate([beta0, gamma0]), args=(y, Xn, Zn),
                            jac=True, method="L-BFGS-B")
    if not res.success:
        print(f"  warning: optimiser did not converge ({res.message})")
    theta = res.x
    beta, gamma = theta[:p], theta[p:]

    # expected Fisher information (the beta/gamma cross block is zero) and per-sample scores
    s = np.exp(Zn @ gamma)
    r = (y - Xn @ beta) / s
    info = np.zeros((p + q, p + q))
    info[:p, :p] = (Xn / s[:, None] ** 2).T @ Xn
    info[p:, p:] = 2.0 * Zn.T @ Zn
    scores = np.hstack([Xn * (r / s)[:, None], Zn * (r ** 2 - 1.0)[:, None]])
    # cluster-robust (sandwich) covariance: sum scores within each day
    g = pd.DataFrame(scores).groupby(np.asarray(groups)).sum().to_numpy()
    bread = np.linalg.inv(info)
    cov = bread @ (g.T @ g) @ bread

    scale = np.concatenate([sx, sz])
    cov = cov / np.outer(scale, scale)
    theta = theta / scale
    names = [f"mu:{c}" for c in X.columns] + [f"logsig:{c}" for c in Z.columns]
    ll = -res.fun - 0.5 * len(y) * np.log(2 * np.pi)
    k = p + q
    return {
        "beta": pd.Series(theta[:p], index=X.columns),
        "gamma": pd.Series(theta[p:], index=Z.columns),
        "cov": pd.DataFrame(cov, index=names, columns=names),
        "loglik": ll, "aic": 2 * k - 2 * ll, "bic": k * np.log(len(y)) - 2 * ll,
        "n": len(y), "n_days": g.shape[0],
    }


def predict(model, d):
    """Mean and standard deviation of T for the rows of 'd'."""
    X = _design(d, model["mean_terms"], model["ref"])
    Z = _design(d, model["scale_terms"], model["ref"])
    mu = X.to_numpy() @ model["beta"].to_numpy()
    sigma = np.exp(Z.to_numpy() @ model["gamma"].to_numpy())
    return pd.Series(mu, index=d.index), pd.Series(sigma, index=d.index)


def _model_data(data, cols, config):
    """Eligible samples: target and covariates present, not in a setpoint recovery."""
    tgt = config["target_col"]
    ok = data[[tgt] + cols].notna().all(axis=1) & ~data["recovery"]
    return data.loc[ok]


def fit_models(data, config=None, mean_terms=None, scale_terms=None):
    """Fit one heteroscedastic Gaussian per regime. Terms default to the CONFIG ones."""
    cfg = cd._merge_config(config, CONFIG)
    mean_terms = cfg["mean_terms"] if mean_terms is None else mean_terms
    scale_terms = cfg["scale_terms"] if scale_terms is None else scale_terms
    cols = _base_cols(cfg["mean_terms"] + cfg["scale_terms"])   # same samples for every variant
    d_all = _model_data(data, cols, cfg)
    ref = reference_values(d_all, cols, cfg)

    models = {}
    for regime in REGIMES:
        d = d_all.loc[d_all["regime"] == regime]
        X = _design(d, mean_terms, ref)
        Z = _design(d, scale_terms, ref)
        days = d.index.normalize()
        m = fit_hetero_gaussian(d[cfg["target_col"]], X, Z, days)
        m.update(regime=regime, ref=ref, mean_terms=mean_terms, scale_terms=scale_terms)
        models[regime] = m
    return models


# Reporting
def _fmt_term(t, ref):
    def c(x):
        return f"({x} - {ref[x]:g})"
    if "*" in t:
        a, b = t.split("*")
        return f"{c(a)}*{c(b)}"
    if "^" in t:
        base, p = t.split("^")
        return f"{c(base)}^{p}"
    return c(t)


def expression(model, digits=4):
    """The fitted mean and log-sigma as readable formulas."""
    ref = model["ref"]
    out = []
    for lhs, params in [("E[T]    ", model["beta"]), ("log sigma", model["gamma"])]:
        s = f"{params['const']:.{digits}g}"
        for t, v in params.drop("const").items():
            s += f" {'-' if v < 0 else '+'} {abs(v):.{digits}g}*{_fmt_term(t, ref)}"
        out.append(f"{lhs} = {s}")
    return "\n".join(out)


def coef_table(model):
    est = pd.concat([model["beta"].add_prefix("mu:"), model["gamma"].add_prefix("logsig:")])
    se = np.sqrt(np.diag(model["cov"]))
    return pd.DataFrame({"estimate": est, "se_day_cluster": se, "z": est / se})


def print_summary(models, variants):
    for regime, m in models.items():
        print(f"\n=== {regime}: n = {m['n']} samples, {m['n_days']} days ===")
        print(expression(m))
        print(coef_table(m).to_string(float_format=lambda x: f"{x:.4g}"))
        print("\n  model comparison (lower AIC/BIC is better):")
        rows = {name: {k: v[regime][k] for k in ["loglik", "aic", "bic"]}
                for name, v in variants.items()}
        print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.1f}"))


# Plots
def _hum_bins(h, width):
    edges = np.arange(np.floor(h.min() / width) * width, h.max() + width, width)
    return pd.cut(h, edges, right=False), edges


def plot_fit(data, models, config=None):
    """
    Per regime:
      left   - temperature adjusted to the reference conditions of the other covariates
               vs humidity, with the fitted mean, +-1/2 sigma bands and binned means;
      middle - fitted sigma vs humidity against the binned std of the residuals;
      right  - standardised residuals against N(0, 1).
    """
    cfg = cd._merge_config(config, CONFIG)
    h_col, tgt = cfg["hum_col"], cfg["target_col"]
    cols = _base_cols(cfg["mean_terms"] + cfg["scale_terms"])
    fig, axes = plt.subplots(len(models), 3, figsize=(16, 4.5 * len(models)), squeeze=False)

    for row, (regime, m) in zip(axes, models.items()):
        d = _model_data(data, cols, cfg)
        d = d.loc[d["regime"] == regime]
        mu, sigma = predict(m, d)
        ref = m["ref"]

        # curve along humidity with the other covariates at their reference values
        grid = pd.DataFrame({c: ref[c] for c in cols},
                            index=range(200)).assign(**{h_col: np.linspace(d[h_col].min(),
                                                                           d[h_col].max(), 200)})
        mu_g, sig_g = predict(m, grid)
        # remove the effect of the other covariates so points and curve are comparable
        at_ref = d.assign(**{c: ref[c] for c in cols if c != h_col})
        mu_ref, sig_ref = predict(m, at_ref)
        z = (d[tgt] - mu) / sigma
        y_adj = mu_ref + z * sig_ref

        ax = row[0]
        ax.scatter(d[h_col], y_adj, s=3, alpha=0.15, color="grey", label="samples (adjusted)")
        for k, a in [(2, 0.15), (1, 0.3)]:
            ax.fill_between(grid[h_col], mu_g - k * sig_g, mu_g + k * sig_g, alpha=a,
                            color="tab:blue", lw=0, label=f"$\\mu \\pm {k}\\sigma$")
        ax.plot(grid[h_col], mu_g, color="tab:blue", lw=2, label="fitted $\\mu$")
        bins, edges = _hum_bins(d[h_col], cfg["hum_bin_width"])
        g = pd.DataFrame({"y": y_adj, "h": d[h_col]}).groupby(bins, observed=True)
        b = g.agg(n=("y", "size"), mean=("y", "mean"), std=("y", "std"), h=("h", "mean"))
        b = b.loc[b["n"] >= cfg["min_bin_count"]]
        ax.errorbar(b["h"], b["mean"], yerr=b["std"], fmt="o", color="tab:red", capsize=3,
                    label="binned mean $\\pm$ std")
        others = ", ".join(f"{c}={ref[c]:g}" for c in cols if c != h_col)
        ax.set(xlabel=f"{h_col} [%]", ylabel=f"{tgt} [°C]",
               title=f"{regime}: {tgt} vs supply-air RH\n(at {others})")
        ax.legend(fontsize=8)

        # the curve is at the reference conditions; the binned points compare the RMS of the
        # fitted sigma with the std of the residuals over the samples actually in each bin
        ax = row[1]
        ax.plot(grid[h_col], sig_g, color="tab:blue", lw=2,
                label=f"fitted $\\sigma$ (at {others})")
        g = pd.DataFrame({"r": d[tgt] - mu, "s2": sigma ** 2, "h": d[h_col]}).groupby(bins, observed=True)
        b = g.agg(n=("r", "size"), std=("r", "std"), s2=("s2", "mean"), h=("h", "mean"))
        b = b.loc[b["n"] >= cfg["min_bin_count"]]
        ax.plot(b["h"], np.sqrt(b["s2"]), "s", mfc="none", color="tab:blue",
                label="fitted $\\sigma$, RMS over bin samples")
        ax.plot(b["h"], b["std"], "o", color="tab:red", label="binned std of residuals")
        ax.set(xlabel=f"{h_col} [%]", ylabel="$\\sigma$ [°C]", title=f"{regime}: spread",
               ylim=(0, None))
        ax.legend(fontsize=8)

        ax = row[2]
        ax.hist(z, bins=80, density=True, color="grey", alpha=0.6)
        xs = np.linspace(-5, 5, 200)
        ax.plot(xs, stats.norm.pdf(xs), color="tab:blue", label="N(0, 1)")
        ax.set(xlabel="$(T - \\mu) / \\sigma$", title=f"{regime}: standardised residuals",
               xlim=(-5, 5))
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def run_analysis(dfs, config=None):
    cfg = cd._merge_config(config, CONFIG)
    data, _ = cd.combine_data(dfs, cfg)
    data = cd.preprocess(data, cfg)

    models = fit_models(data, cfg)
    h = cfg["hum_col"]
    no_hum = lambda terms: [t for t in terms if h not in _base_cols([t])]
    variants = {
        "full": models,
        "no humidity": fit_models(data, cfg, no_hum(cfg["mean_terms"]), no_hum(cfg["scale_terms"])),
        "constant sigma": fit_models(data, cfg, scale_terms=[]),
    }
    print_summary(models, variants)

    if cfg["make_plots"]:
        figure_dir = cd.Path("./figure")
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_fit(data, models, cfg).savefig(figure_dir / f"{cfg['target_col']}_hetero_gaussian.png",
                                            dpi=300, bbox_inches="tight")
        plt.show()
    return data, models


if __name__ == "__main__":
    import pickle

    with open("./07-D2-02_data/dfs.pkl", "rb") as f:
        dfs = pickle.load(f)

    data, models = run_analysis(dfs)
