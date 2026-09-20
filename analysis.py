from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polymul

from model import (
    Params,
    assimilation_linear,
    assimilation_nonlinear,
    diagnostics,
    emissions_from_tau,
    growth_rate,
    rhs_linear,
    rhs_nonlinear,
    solve_linear,
    tau_nonlinear_from_E,
    u_from_tau,
    x_nonlinear_from_E,
    F_nonlinear,
)
from dynamics import stable_path_to_E0, simulate_from_initial_state

BASE_B = 0.45
BASE_DELTA = 0.01
BASE_ETA = 1.5
BASE_PBAR = 1.0
BASE_M_L = 0.5
BASE_M_NL = 2.0

MULTIPLICITY_FILE = Path(__file__).with_name("multiplicity_region.csv.gz")


def base_params():
    return (
        Params(B=BASE_B, delta=BASE_DELTA, eta=BASE_ETA, Pbar=BASE_PBAR, m=BASE_M_L),
        Params(B=BASE_B, delta=BASE_DELTA, eta=BASE_ETA, Pbar=BASE_PBAR, m=BASE_M_NL),
    )


def _result_from_nonlinear_E(E_star: float, p: Params, root_number: int = 1) -> dict:
    tau_star = float(tau_nonlinear_from_E(E_star, p))
    x_star = float(x_nonlinear_from_E(E_star, p))
    u_star = float(u_from_tau(tau_star))
    g_star = float(growth_rate(tau_star, p))
    state = np.array([x_star, E_star, tau_star], dtype=float)
    diag = diagnostics(state, p, rhs_nonlinear)
    Z_star = float(emissions_from_tau(tau_star, p))
    A_star = float(assimilation_nonlinear(E_star, p))
    midpoint = p.Pbar / 2.0
    if E_star < midpoint - 1e-9:
        branch = "dolna"
    elif E_star > midpoint + 1e-9:
        branch = "górna"
    else:
        branch = "maksimum"
    return {
        "model": "nonlinear",
        "root": root_number,
        "branch": branch,
        "E": float(E_star),
        "pollution": float(p.Pbar - E_star),
        "tau": tau_star,
        "u": u_star,
        "x": x_star,
        "g": g_star,
        "Z": Z_star,
        "A": A_star,
        **diag,
    }


def solve_nonlinear_all(
    p: Params,
    imag_tol: float = 1e-7,
    residual_tol: float = 1e-7,
    merge_tol: float = 1e-7,
) -> list[dict]:
    """Find all economically admissible interior stationary points.

    Uses the polynomial representation from the multiplicity experiment in the thesis,
    so it is valid for arbitrary positive Pbar, not only the baseline Pbar=1 case.
    """
    if p.B <= 0 or p.delta <= 0 or p.eta <= 0 or p.Pbar <= 0 or p.m <= 0:
        return []

    s = p.m * p.Pbar
    D = np.array([p.B, s, -s], dtype=float)
    c = p.delta + (p.eta - 1.0) * p.B
    N = np.array([p.delta * p.B, c * s, -c * s], dtype=float)

    one_minus_2z = np.array([1.0, -2.0])
    z_poly = np.array([0.0, 1.0])
    term1 = polymul(polymul(one_minus_2z, D), z_poly)
    term1 *= -p.m * p.eta * p.Pbar * p.B**2

    D_squared = polymul(D, D)
    bracket = np.zeros(max(2, len(D_squared)))
    bracket[1] = p.Pbar * p.B**2
    bracket[: len(D_squared)] -= D_squared
    term2 = polymul(N, bracket)

    length = max(len(term1), len(term2))
    G = np.zeros(length)
    G[: len(term1)] += term1
    G[: len(term2)] += term2

    scale = np.max(np.abs(G))
    if not np.isfinite(scale) or scale == 0:
        return []
    G = G / scale
    while len(G) > 1 and abs(G[-1]) < 1e-13:
        G = G[:-1]

    roots = np.roots(G[::-1])
    candidates: list[float] = []
    for root in roots:
        if abs(root.imag) > imag_tol:
            continue
        z = float(root.real)
        if not (merge_tol < z < 1.0 - merge_tol):
            continue
        E = p.Pbar * z
        if not np.isfinite(E):
            continue
        try:
            F = float(F_nonlinear(E, p))
            tau = float(tau_nonlinear_from_E(E, p))
            x = float(x_nonlinear_from_E(E, p))
        except (FloatingPointError, ZeroDivisionError, ValueError):
            continue
        if not (np.isfinite(F) and np.isfinite(tau) and np.isfinite(x)):
            continue
        if abs(F) > residual_tol:
            continue
        if not (0.0 < E < p.Pbar and x > 0.0 and 0.0 < tau < 1.0):
            continue
        candidates.append(E)

    candidates.sort()
    unique: list[float] = []
    tolerance = merge_tol * max(1.0, p.Pbar)
    for E in candidates:
        if not unique or abs(E - unique[-1]) > tolerance:
            unique.append(E)

    return [_result_from_nonlinear_E(E, p, i + 1) for i, E in enumerate(unique)]


def valid_results(results: list[dict]) -> list[dict]:
    return [
        r for r in results
        if bool(r.get("admissible")) and bool(r.get("numerically_valid"))
    ]


def solve_pair(
    B=BASE_B,
    delta=BASE_DELTA,
    eta=BASE_ETA,
    Pbar=BASE_PBAR,
    m_l=BASE_M_L,
    m_nl=BASE_M_NL,
):
    p_l = Params(B=float(B), delta=float(delta), eta=float(eta), Pbar=float(Pbar), m=float(m_l))
    p_nl = Params(B=float(B), delta=float(delta), eta=float(eta), Pbar=float(Pbar), m=float(m_nl))
    try:
        L = solve_linear(p_l)
    except Exception:
        L = None
    L = L if L is not None and L.get("admissible") and L.get("numerically_valid") else None
    try:
        NL = valid_results(solve_nonlinear_all(p_nl))
    except Exception:
        NL = []
    return p_l, p_nl, L, NL


def result_table(L: dict | None, NL: list[dict]) -> pd.DataFrame:
    rows = []
    if L is not None:
        rows.append(_row(L, "Liniowy", 1))
    for i, r in enumerate(NL, start=1):
        rows.append(_row(r, "Nieliniowy", i))
    return pd.DataFrame(rows)


def _row(r: dict, model_label: str, point_number: int) -> dict:
    eig = np.asarray(r.get("eigenvalues", []), dtype=complex)
    eig_txt = ", ".join(
        f"{z.real:+.3f}{z.imag:+.3f}i" if abs(z.imag) > 1e-9 else f"{z.real:+.3f}"
        for z in eig
    )
    stability = {
        "saddle": "siodłowy",
        "unstable": "niestabilny",
        "nonhyperbolic": "niehiperboliczny",
        "other": "inna",
    }.get(r.get("stability"), str(r.get("stability")))
    return {
        "Model": model_label,
        "Punkt": point_number,
        "Gałąź": r.get("branch", "—"),
        "E*": r["E"],
        "P*": r["pollution"],
        "τ*": r["tau"],
        "u*": r["u"],
        "x*": r["x"],
        "g*": r["g"],
        "Z*": r["Z"],
        "A(E*)": r["A"],
        "Stabilność": stability,
        "R_max": r["residual_max"],
        "Wartości własne": eig_txt,
        "Dodatni wzrost": bool(r["positive_growth"]),
        "TVC": bool(r["transversality"]),
    }


def assimilation_dataframe(Pbar, m_l, m_nl, n=301):
    Pbar = float(Pbar)
    E = np.linspace(0.0, Pbar, n)
    p_l = Params(B=BASE_B, delta=BASE_DELTA, eta=BASE_ETA, Pbar=Pbar, m=float(m_l))
    p_nl = Params(B=BASE_B, delta=BASE_DELTA, eta=BASE_ETA, Pbar=Pbar, m=float(m_nl))
    return pd.DataFrame({
        "E": E,
        "Liniowa": [assimilation_linear(e, p_l) for e in E],
        "Nieliniowa": [assimilation_nonlinear(e, p_nl) for e in E],
    })


def sensitivity_grid(parameter: str, n=201) -> pd.DataFrame:
    if parameter == "eta":
        values = np.linspace(0.75, 2.25, n)
        base = BASE_ETA
    elif parameter == "delta":
        values = np.linspace(0.001, 0.02, n)
        base = BASE_DELTA
    elif parameter == "m":
        values = np.geomspace(0.05, 5.0, n)
        base = BASE_M_L
    else:
        raise ValueError(parameter)
    values[np.argmin(np.abs(values - base))] = base
    values = np.sort(values)

    rows = []
    for value in values:
        B, delta, eta, Pbar, m_l = BASE_B, BASE_DELTA, BASE_ETA, BASE_PBAR, BASE_M_L
        if parameter == "eta":
            eta = float(value)
        elif parameter == "delta":
            delta = float(value)
        else:
            m_l = float(value)
        m_nl = 4.0 * m_l
        _, _, L, NL = solve_pair(B, delta, eta, Pbar, m_l, m_nl)
        for model, result in (("Liniowy", L), ("Nieliniowy", NL[0] if len(NL) == 1 else None)):
            if result is None:
                rows.append({
                    "parameter": parameter,
                    "value": float(value),
                    "model": model,
                    "found": False,
                    "E": np.nan,
                    "u": np.nan,
                    "x": np.nan,
                    "g": np.nan,
                    "Z": np.nan,
                    "residual_max": np.nan,
                    "stability": None,
                    "positive_growth": False,
                    "TVC": False,
                    "n_nonlinear": len(NL),
                })
            else:
                rows.append({
                    "parameter": parameter,
                    "value": float(value),
                    "model": model,
                    "found": True,
                    "E": result["E"],
                    "u": result["u"],
                    "x": result["x"],
                    "g": result["g"],
                    "Z": result["Z"],
                    "residual_max": result["residual_max"],
                    "stability": result["stability"],
                    "positive_growth": result["positive_growth"],
                    "TVC": result["transversality"],
                    "n_nonlinear": len(NL),
                })
    return pd.DataFrame(rows)


def base_stable_paths(E0: float):
    p_l, p_nl = base_params()
    L = solve_linear(p_l)
    NL_all = valid_results(solve_nonlinear_all(p_nl))
    NL = NL_all[0] if NL_all else None
    out = {}
    if L is not None and L.get("stability") == "saddle":
        out["linear"] = stable_path_to_E0(L, p_l, rhs_linear, "linear", float(E0))
    else:
        out["linear"] = (None, {"success": False, "status": "no_saddle"})
    if NL is not None and NL.get("stability") == "saddle":
        out["nonlinear"] = stable_path_to_E0(NL, p_nl, rhs_nonlinear, "nonlinear", float(E0))
    else:
        out["nonlinear"] = (None, {"success": False, "status": "no_saddle"})
    return out


def stable_path_for_result(result: dict, p: Params, model: str, E0: float):
    if result is None or result.get("stability") != "saddle":
        return None, {"success": False, "status": "selected_point_not_saddle"}
    rhs = rhs_linear if model == "linear" else rhs_nonlinear
    return stable_path_to_E0(result, p, rhs, model, float(E0))


def free_trajectory(result: dict, p: Params, model: str, E0: float, x0: float, u0: float, t_max=25.0):
    rhs = rhs_linear if model == "linear" else rhs_nonlinear
    state_star = np.array([result["x"], result["E"], result["tau"]], dtype=float) if result is not None else None
    return simulate_from_initial_state(
        params=p,
        rhs=rhs,
        model_name=model,
        x0=float(x0),
        E0=float(E0),
        tau0=float(u0) ** 2,
        t_max=float(t_max),
        state_star=state_star,
        t_eval=np.linspace(0.0, float(t_max), 600),
        rtol=1e-9,
        atol=1e-11,
        max_step=0.05,
        bound_eps=1e-8,
    )


def multiplicity_region() -> pd.DataFrame:
    if MULTIPLICITY_FILE.exists():
        try:
            return pd.read_csv(MULTIPLICITY_FILE, compression="gzip")
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def multiplicity_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Pbar", "m_min", "m_max", "n_combinations"])
    return (
        df.groupby("Pbar")
        .agg(m_min=("mNL", "min"), m_max=("mNL", "max"), n_combinations=("mNL", "size"))
        .reset_index()
    )


def status_text(diag: dict) -> str:
    status = diag.get("status", "unknown")
    mapping = {
        "steady_state": "stan początkowy jest równy punktowi stacjonarnemu",
        "reached_target": "wyznaczono stabilną ścieżkę",
        "E0_outside_domain": "E₀ znajduje się poza dopuszczalną dziedziną",
        "selected_point_not_saddle": "wybrany punkt nie ma struktury siodłowej wymaganej przez tę procedurę",
        "no_saddle": "nie znaleziono odpowiedniego punktu siodłowego",
    }
    return mapping.get(status, str(status).replace("_", " "))
