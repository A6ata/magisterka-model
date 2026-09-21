from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    BASE_B,
    BASE_DELTA,
    BASE_ETA,
    BASE_M_L,
    BASE_M_NL,
    BASE_PBAR,
    assimilation_dataframe,
    base_params,
    base_stable_paths,
    free_trajectory,
    multiplicity_boundaries,
    multiplicity_region,
    result_table,
    sensitivity_grid,
    solve_nonlinear_all,
    solve_pair,
    stable_path_for_result,
)
from model import Params, rhs_linear, rhs_nonlinear, solve_linear


st.set_page_config(
    page_title="Nieliniowa asymilacja a zrównoważony wzrost",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1450px; padding-top: 3.0rem; padding-bottom: 2.5rem;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      .page-kicker {font-size: .86rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65; margin-bottom: .25rem;}
      .page-title {font-size: 2.15rem; font-weight: 750; line-height: 1.12; margin-bottom: .35rem;}
      .page-subtitle {font-size: 1.04rem; opacity: .72; margin-bottom: 1.35rem; max-width: 1050px;}
      .hero {padding: 1.7rem 1.8rem; border-radius: 1.05rem; background: linear-gradient(135deg, rgba(19,96,67,.12), rgba(19,96,67,.03)); border: 1px solid rgba(19,96,67,.22); margin-bottom: 1.35rem;}
      .hero-title {font-size: 2.25rem; font-weight: 780; line-height: 1.12; margin-bottom: .6rem;}
      .hero-meta {opacity: .70;}
      .card {border: 1px solid rgba(128,128,128,.22); border-radius: .9rem; padding: 1rem 1.05rem; height: 100%;}
      .card-title {font-weight: 700; margin-bottom: .35rem;}
      .takeaway {border-left: 4px solid #2f7d5a; background: rgba(47,125,90,.08); padding: .85rem 1rem; border-radius: .35rem; margin-top: 1rem;}
      .warnbox {border-left: 4px solid #d19a25; background: rgba(209,154,37,.09); padding: .85rem 1rem; border-radius: .35rem; margin-top: 1rem;}
      [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: .8rem 1rem; border-radius: .85rem; background: rgba(128,128,128,.025);}
      div[data-testid="stExpander"] details {border-radius: .8rem;}
      .small {font-size: .91rem; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)


PAGES = [
    "1. Problem badawczy i cel pracy",
    "2. Model bazowy",
    "3. Modyfikacja i wyniki analityczne",
    "4. Parametryzacja i scenariusz bazowy",
    "5. Analiza wrażliwości",
    "6. Dynamika przejściowa",
    "7. Wielopunktowość",
    "8. Wnioski i ograniczenia",
    "9. Symulacje",
]

if "nav" not in st.session_state:
    st.session_state.nav = PAGES[0]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def fmt(x, digits=4):
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"{float(x):.{digits}f}".replace(".", ",")
    except Exception:
        return "—"


def pct(new, old):
    if old == 0:
        return "—"
    return f"{(new / old - 1) * 100:+.1f}%".replace(".", ",")


def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="page-kicker">Obrona pracy magisterskiej</div>
        <div class="page-title">{title}</div>
        <div class="page-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, text: str):
    st.markdown(
        f'<div class="card"><div class="card-title">{title}</div><div>{text}</div></div>',
        unsafe_allow_html=True,
    )


def takeaway(text: str):
    st.markdown(f'<div class="takeaway"><b>Wniosek:</b> {text}</div>', unsafe_allow_html=True)


def warning_box(text: str):
    st.markdown(f'<div class="warnbox">{text}</div>', unsafe_allow_html=True)


def assimilation_plot(Pbar, m_l, m_nl, show_linear=True, show_nonlinear=True, mark_max=True):
    df = assimilation_dataframe(Pbar, m_l, m_nl)
    fig = go.Figure()
    if show_linear:
        fig.add_trace(go.Scatter(x=df["E"], y=df["Liniowa"], mode="lines", name="model liniowy"))
    if show_nonlinear:
        fig.add_trace(go.Scatter(x=df["E"], y=df["Nieliniowa"], mode="lines", name="model nieliniowy"))
    if show_nonlinear:
        fig.add_vline(x=Pbar / 2, line_dash="dot", line_width=1, annotation_text="P̄/2")
        if mark_max:
            fig.add_trace(go.Scatter(
                x=[Pbar / 2], y=[m_nl * Pbar / 4], mode="markers", name="maksimum NL", marker_size=10
            ))
    fig.update_layout(
        xaxis_title="Zasób środowiska E",
        yaxis_title="Strumień asymilacji A(E)",
        height=390,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def trajectory_plot(path_l, path_nl, variable, ylabel, steady_l=None, steady_nl=None):
    fig = go.Figure()
    if path_l is not None and not path_l.empty:
        fig.add_trace(go.Scatter(x=path_l["time"], y=path_l[variable], mode="lines", name="liniowy"))
    if path_nl is not None and not path_nl.empty:
        fig.add_trace(go.Scatter(x=path_nl["time"], y=path_nl[variable], mode="lines", name="nieliniowy"))
    if steady_l is not None:
        fig.add_hline(y=steady_l, line_dash="dash", line_width=1, opacity=.45)
    if steady_nl is not None:
        fig.add_hline(y=steady_nl, line_dash="dash", line_width=1, opacity=.45)
    fig.update_layout(
        xaxis_title="Czas",
        yaxis_title=ylabel,
        height=330,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def style_results(df: pd.DataFrame):
    if df.empty:
        return df
    visible = ["Model", "Punkt", "E*", "P*", "τ*", "u*", "x*", "g*", "Z*", "Stabilność"]
    out = df[visible].copy()
    return out.style.format({c: "{:.4f}" for c in ["E*", "P*", "τ*", "u*", "x*", "g*", "Z*"]})


@st.cache_data(show_spinner=False)
def cached_sensitivity(parameter: str):
    return sensitivity_grid(parameter)


@st.cache_data(show_spinner=False)
def cached_base_paths(E0: float):
    return base_stable_paths(float(E0))


@st.cache_data(show_spinner=False)
def cached_multiplicity():
    return multiplicity_region()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.radio("Przejdź do sekcji", PAGES, key="nav", label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Agata Kwiatkowska · UEP · 2026")

page = st.session_state.nav


# ---------------------------------------------------------------------
# 1. Problem and objective
# ---------------------------------------------------------------------
if page == PAGES[0]:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">Wpływ nieliniowej funkcji asymilacji zanieczyszczeń na dynamikę modelu zrównoważonego wzrostu gospodarczego</div>
          <div class="hero-meta">Agata Kwiatkowska · Informatyka i ekonometria · Uniwersytet Ekonomiczny w Poznaniu · 2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.05, 1], gap="large")
    with c1:
        st.subheader("Problem badawczy")
        st.write(
            "Rzeczywiste procesy regeneracji środowiska są złożone i mogą zależeć od jego aktualnego stanu. "
            "W modelach ekonomicznych często są reprezentowane za pomocą uproszczonych funkcji asymilacji. Przyęta postać funkcji może jednak wpływać na właściwości modelu."
        )
    with c2:
        st.subheader("Cel pracy")
        st.write(
            "Określenie konsekwencji zastąpienia liniowej funkcji asymilacji zanieczyszczeń funkcją nieliniową, uwzględniającą osłabienie zdolności asymilacyjnych przy silnej degradacji środowiska, w modelu optymalnego zrównoważonego wzrostu Cazzavillana i Musu (1998)."
        )

    st.markdown("### Zakres analizy")
    a, b, c, d = st.columns(4)
    with a: card("Punkty stacjonarne", "Istnienie, liczba i ekonomiczna dopuszczalność punktów stacjonarnych.")
    with b: card("Stabilność", "Lokalna klasyfikacja stabilności punktów stacjonarnych.")
    with c: card("Wzrost", "Długookresowe tempo wzrostu na zrównoważonej ścieżce wzrostu.")
    with d: card("Dynamika", r"Przebieg stabilnej ścieżki prowadzącej do punktu stacjonarnego przy różnych wartościach E0")


# ---------------------------------------------------------------------
# 2. Baseline model - deliberately slimmed
# ---------------------------------------------------------------------
elif page == PAGES[1]:
    page_header(
        "2. Model bazowy — Cazzavillan i Musu (1998)",
        "Optymalny podział zasobów między produkcję, konsumpcję i ograniczanie emisji, przy bezpośrednim wpływie środowiska na użyteczność.",
    )
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("#### Mechanizm gospodarczy")
        st.latex(r"Y=B(1-u)K")
        st.latex(r"Z=B\left(\frac{1}{u}-1\right)")
        st.latex(r"\dot K=B(1-u)K-C")
        st.latex(r"\dot E=A_L(E)-Z")
        st.latex(r"A_L(E)=m_L(\bar P-E)")
        st.caption("K, E — zmienne stanu; C, u — zmienne sterujące.")
    with c2:
        st.markdown("#### Problem planisty")
        st.latex(r"\max_{C,u}\int_0^\infty e^{-\delta t}U_\eta(C,E)\,dt")
        st.latex(r"U_\eta(C,E)=\frac{(CE)^{1-\eta}}{1-\eta},\quad \eta\neq 1")
        st.latex(r"U_1(C,E)=\ln(CE), \quad \eta = 1")
        st.caption("δ — stopa dyskontowa; η — parametr krzywizny funkcji użyteczności.")
    with c3:
        st.markdown("#### Redukcja modelu")
        st.latex(r"x=\frac{C}{K}")
        st.latex(r"\tau=\frac{\lambda}{vK}")
        st.latex(r"u=\sqrt{\tau}")

    a, b, c = st.columns(3)
    with a:
        card(
        "Istnienie i jednoznaczność punktu stacjonarnego",
        "Przy założeniu δ ≥ (1−η)B warunek "
        "P̄ > δ/(m<sub>L</sub>η + δ) jest konieczny i wystarczający "
        "dla istnienia dokładnie jednego dodatniego punktu stacjonarnego."
    )

    with b:
        card(
        "Wzrost",
        "Dodatnie długookresowe tempo wzrostu g<sup>*</sup> > 0 "
        "występuje wtedy i tylko wtedy, gdy "
        "B(1−√τ<sup>*</sup>) > δ."
    )

    with c:
        card(
        "Stabilność",
        "Punkt stacjonarny ma jedną wartość własną o ujemnej części rzeczywistej "
        "i dwie o dodatniej części rzeczywistej, co oznacza lokalny charakter siodłowy."
    )

    with st.expander("Pełny zredukowany układ modelu liniowego"):
        st.latex(r"\dot E=m_L(\bar P-E)-B(\tau^{-1/2}-1)")
        st.latex(r"\dot\tau=\tau\left[m_L-\frac{x}{E\tau}+x\right]")
        st.latex(r"\dot x=x\left[\frac{1-\eta}{\eta}\frac{\dot E}{E}+\frac{1-\eta}{\eta}B(1-\sqrt\tau)-\frac{\delta}{\eta}+x\right]")
    st.markdown("### Główne ograniczenia modelu bazowego")

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    with c1:
        card("Liniowa asymilacja", "")

    with c2:
        card("Uproszczona technologia środowiskowa", "")

    with c3:
        card("Brak niepewności", "")

    with c4:
        card("Brak amortyzacji kapitału", "")

    with c5:
        card("Centralny planista", "")

    with c6:
        card("Egzogeniczne preferencje i stopa dyskontowa", "")


# ---------------------------------------------------------------------
# 3. Modification + analytical results
# ---------------------------------------------------------------------
elif page == PAGES[2]:
    page_header(
        "3. Modyfikacja funkcji asymilacji i wyniki analityczne",
        "Zmiana dotyczy sposobu opisu naturalnej regeneracji środowiska.",
    )
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        st.plotly_chart(assimilation_plot(BASE_PBAR, BASE_M_L, BASE_M_NL, show_linear=False), use_container_width=True)
    with c2:
        st.markdown("#### Nieliniowa asymilacja")
        st.latex(r"A_{NL}(E)=m_{NL}E\left(1-\frac{E}{\bar P}\right)")
        st.latex(r"A_{NL}(0)=A_{NL}(\bar P)=0")
        st.latex(r"E^\dagger=\frac{\bar P}{2},\qquad A_{max}^{NL}=\frac{m_{NL}\bar P}{4}")
        st.write(
            "Strumień asymilacji jest niski zarówno przy niewielkiej ilości zanieczyszczeń, jak i przy silnej degradacji, a pomiędzy tymi przypadkami osiąga maksimum."
        )
        st.caption("Jest to założenie modelowe — nie uniwersalny opis procesów regeneracyjnych.")

    st.info(
    "Warunki pierwszego rzędu względem C i u oraz równanie dynamiki ceny cienia kapitału "
    "pozostają bez zmian. Zmieniają się równania dynamiki zasobu środowiska i jego ceny cienia, "
    "a tym samym zredukowany układ (x, E, τ)."
    )

    a, b, c = st.columns(3, gap="large")

    with a:
        card(
        "Punkty stacjonarne",
        "Warunek dE/dt = 0 może prowadzić do dwóch wartości zasobu środowiska: "
        "E₋(τ) < P̄/2 oraz E₊(τ) > P̄/2. Wynika to z nieliniowego kształtu funkcji asymilacji. "
        "Nie oznacza to jednak istnienia dwóch punktów stacjonarnych pełnego układu — "
        "bez dodatkowych założeń ich liczba nie jest z góry określona."
    )

    with b:
        card(
        "Długookresowy wzrost",
        "Wzór na tempo wzrostu pozostaje taki sam jak w modelu bazowym."
        "Dodatni wzrost występuje wtedy i tylko wtedy, gdy B(1−√τ*) > δ. "
    )

    with c:
        card(
        "Lokalna stabilność",
        "Dla η ≥ 1 każdy ekonomicznie dopuszczalny punkt na górnej gałęzi "
        "jest lokalnie siodłowy. Punkt E*=P̄/2 jest siodłowy dla każdego η > 0. "
        "W pozostałych przypadkach wynik zależy od parametrów."
    )

    with st.expander("Szczegóły analityczne"):
        st.markdown("**Zmodyfikowany układ:**")
        st.latex(r"\dot E=m_{NL}E\left(1-\frac{E}{\bar P}\right)-B(\tau^{-1/2}-1)")
        st.latex(r"\dot\tau=\tau\left[-m_{NL}\left(1-\frac{2E}{\bar P}\right)-\frac{x}{E\tau}+x\right]")
        st.latex(r"\dot x=x\left[\frac{1-\eta}{\eta}\frac{\dot E}{E}+\frac{1-\eta}{\eta}B(1-\sqrt\tau)-\frac{\delta}{\eta}+x\right]")
        st.markdown("**Konieczny warunek dodatniego wzrostu:**")
        st.latex(r"\frac{B m_{NL}\bar P}{4B+m_{NL}\bar P}>\delta")


# ---------------------------------------------------------------------
# 4. Parameterisation and baseline scenario
# ---------------------------------------------------------------------
elif page == PAGES[3]:
    page_header(
        "4. Parametryzacja i scenariusz bazowy",
        "Parametryzacja jest stylizowana: służy porównaniu dwóch sposobów modelowania asymilacji, a nie odwzorowaniu konkretnej gospodarki lub ekosystemu.",
    )

    c1, c2 = st.columns([.9, 1.25], gap="large")
    with c1:
        st.markdown("#### Wartości bazowe")
        st.markdown(
            r"""
            **B = 0,45** — produktywność kapitału produkcyjnego  
            **δ = 0,01** — stopa dyskontowa  
            **η = 1,5** — krzywizna funkcji użyteczności  
            **P̄ = 1** — maksymalny dopuszczalny poziom zanieczyszczeń  
            **mL = 0,5**, **mNL = 2** — parametry asymilacji
            """
        )
        st.markdown("#### Kryterium porównania")
        st.latex(r"A_{max}^{L}=A_{max}^{NL}\quad\Longrightarrow\quad m_{NL}=4m_L")
        st.caption("mL i mNL mają odmienną interpretację; zrównanie maksimów nie oznacza jednakowej asymilacji przy tym samym E.")
    with c2:
        st.plotly_chart(assimilation_plot(BASE_PBAR, BASE_M_L, BASE_M_NL), use_container_width=True)

    _, _, L, NL = solve_pair()
    N = NL[0]
    st.markdown("### Scenariusz bazowy")
    k1, k2, k3 = st.columns(3)
    k1.metric("Zasób środowiska E*", fmt(N["E"]), pct(N["E"], L["E"]))
    k2.metric("Udział kapitału u*", fmt(N["u"]), pct(N["u"], L["u"]))
    k3.metric("Tempo wzrostu g*", fmt(N["g"]), pct(N["g"], L["g"]))
    st.caption("Model nieliniowy - zmiana względem modelu liniowego.")

    takeaway(
        "Dla przyjętych wartości parametrów model nieliniowy ma wyższy E*, niższy u* i wyższy g*. Jednocześnie Z* jest wyższe, lecz w stanie stacjonarnym równoważy je wyższy strumień asymilacji."
    )

    with st.expander("Pokaż pełną tabelę wyników"):
        st.dataframe(style_results(result_table(L, NL)), use_container_width=True)
    with st.expander("Uzasadnienie wartości parametrów"):
        st.write(
            "η=1,5 i δ=0,01 mieszczą się w zakresach spotykanych w literaturze; B=0,45 przyjęto jako przybliżoną relację produkcji do kapitału produkcyjnego na podstawie wcześniejszych badań. "
            "Dla mL wykorzystano literaturę środowiskową jako punkt odniesienia skali asymilacji. Parametr mNL wyznaczono z warunku jednakowej maksymalnej zdolności asymilacyjnej obu modeli; P̄=1 jest założeniem parametryzacyjnym, które wpływa na warunki istnienia punktów stacjonarnych i wyklucza dolną gałąź w scenariuszu bazowym."
        )


# ---------------------------------------------------------------------
# 5. Sensitivity
# ---------------------------------------------------------------------
elif page == PAGES[4]:
    page_header(
        "5. Analiza wrażliwości",
        "Jednoczynnikowa analiza wrażliwości: czy wnioski ze scenariusza bazowego utrzymują się po zmianie η, δ lub m?",
    )

    option = st.radio(
        "Parametr",
        ["η", "δ", "mL"],
        horizontal=True,
    )
    parameter = {"η": "eta", "δ": "delta", "mL": "m"}[option]
    df = cached_sensitivity(parameter)

    meta = {
        "eta": ("η", "[0,75; 2,25]", BASE_ETA),
        "delta": ("δ", "[0,001; 0,02]", BASE_DELTA),
        "m": ("mL", "[0,05; 5]", BASE_M_L),
    }
    symbol, rng, base_val = meta[parameter]
    st.caption(f"Zakres: {rng} · wartość bazowa: {str(base_val).replace('.', ',')}" + (" · zawsze mNL = 4mL" if parameter == "m" else ""))

    def sens_fig(metric, ylabel):
        fig = go.Figure()
        for model in ["Liniowy", "Nieliniowy"]:
            sub = df[(df.model == model) & (df.found)].copy()
            fig.add_trace(go.Scatter(x=sub.value, y=sub[metric], mode="lines", name=model.lower()))
        fig.add_vline(x=base_val, line_dash="dot", line_width=1.2, annotation_text="baza")
        if parameter == "m":
            fig.update_xaxes(type="log")
        fig.update_layout(
            xaxis_title=symbol,
            yaxis_title=ylabel,
            height=375,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        return fig

    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(sens_fig("E", "E*"), use_container_width=True)
    with p2:
        st.plotly_chart(sens_fig("g", "g*"), use_container_width=True)

    st.markdown("#### Interpretacja")
    if parameter == "eta":
        st.write(
            "Dla najniższych wartości η procedura nie wyznacza ekonomicznie dopuszczalnego punktu; pierwsza wartość siatki z rozwiązaniem w obu modelach to około η=0,96. "
            "W zakresie z istniejącymi rozwiązaniami wzrost η zwiększa E*, u* i x*, a obniża g*."
        )
    elif parameter == "delta":
        st.write(
            "Wzrost δ zwiększa E*, u* i x*, a obniża g*. Zmiany są relatywnie niewielkie w badanym zakresie, więc wyniki są jedynie umiarkowanie wrażliwe na stopę dyskontową."
        )
    else:
        st.write(
            "Wzrost zdolności asymilacyjnej zmniejsza u* oraz zwiększa x* i g*. Zależność E* od mL nie jest ściśle monotoniczna przy niskich wartościach parametru, a następnie E* rośnie wraz z mL."
        )

    takeaway(
        "Dla wszystkich wyznaczonych ekonomicznie dopuszczalnych rozwiązań w analizowanych zakresach model nieliniowy miał wyższe E*, niższe u* i wyższe g* niż model liniowy. Skala różnic zależała od parametrów."
    )

    with st.expander("Pokaż dodatkowe zmienne: u*, x*, Z*"):
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(sens_fig("u", "u*"), use_container_width=True)
        with c2: st.plotly_chart(sens_fig("x", "x*"), use_container_width=True)
        with c3: st.plotly_chart(sens_fig("Z", "Z*"), use_container_width=True)


# ---------------------------------------------------------------------
# 6. Transitional dynamics
# ---------------------------------------------------------------------
elif page == PAGES[5]:
    page_header(
        "6. Dynamika przejściowa",
        "Dla zadanego E₀ zmienne skokowe x i τ są dobierane tak, aby gospodarka znalazła się na stabilnej ścieżce prowadzącej do punktu stacjonarnego.",
    )

    E0 = st.slider("Początkowy stan środowiska E₀", 0.05, 0.95, 0.50, 0.005)
    paths = cached_base_paths(E0)
    path_l, diag_l = paths["linear"]
    path_nl, diag_nl = paths["nonlinear"]
    p_l, p_nl = base_params()
    L = solve_linear(p_l)
    NL = solve_nonlinear_all(p_nl)[0]

    a, b = st.columns(2)
    with a:
        st.markdown("#### Model liniowy")
        if path_l is not None:
            st.metric("x₀", fmt(diag_l.get("x_0")), None)
            st.metric("u₀", fmt(diag_l.get("u_0")), None)
        else:
            st.warning("Dla tego E₀ procedura nie wyznaczyła stabilnej ścieżki modelu liniowego.")
    with b:
        st.markdown("#### Model nieliniowy")
        if path_nl is not None:
            st.metric("x₀", fmt(diag_nl.get("x_0")), None)
            st.metric("u₀", fmt(diag_nl.get("u_0")), None)
        else:
            st.warning("Dla tego E₀ zastosowana procedura nie wyznaczyła ekonomicznie dopuszczalnej stabilnej ścieżki modelu nieliniowego.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(trajectory_plot(path_l, path_nl, "E", "E(t)", L["E"], NL["E"]), use_container_width=True)
    with c2:
        st.plotly_chart(trajectory_plot(path_l, path_nl, "x", "x(t)", L["x"], NL["x"]), use_container_width=True)
    st.plotly_chart(trajectory_plot(path_l, path_nl, "u", "u(t)", L["u"], NL["u"]), use_container_width=True)


    warning_box(
        "Przy parametryzacji bazowej najniższą wartością zastosowanej siatki, dla której procedura wyznaczyła dopuszczalną stabilną ścieżkę modelu nieliniowego, było E₀=0,205 (u₀≈0,9885). Dla niższych E₀ trajektoria osiągała granicę u=1. Jest to wynik numeryczny właściwy dla przyjętej parametryzacji i procedury."
    )

    with st.expander("Odchylenie od stabilnej ścieżki"):
        st.write("W pracy dla E₀=0,50 osobno zaburzano x₀ i u₀ o ±1%. W obu modelach trajektorie opuszczały stabilną ścieżkę, co ilustruje siodłowy charakter równowagi. Interaktywną wersję tego eksperymentu zawiera sekcja 9.")


# ---------------------------------------------------------------------
# 7. Multiplicity
# ---------------------------------------------------------------------
elif page == PAGES[6]:
    page_header(
        "7. Wielopunktowość",
        "Eksperyment eksploracyjny: czy brak ogólnej gwarancji jednoznaczności ma wyłącznie charakter teoretyczny, czy model rzeczywiście może generować kilka ekonomicznie dopuszczalnych punktów stacjonarnych?",
    )

    region = cached_multiplicity()
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown("#### Numerycznie zidentyfikowany obszar")
        if not region.empty:
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=region["mNL"], y=region["Pbar"], mode="markers",
                marker=dict(size=5, opacity=.65), name="3 punkty"
            ))
            fig.add_trace(go.Scatter(x=[0.05], y=[400], mode="markers", marker=dict(size=12, symbol="x"), name="przykład"))
            fig.update_layout(
                xaxis_title="mNL", yaxis_title="P̄", height=470,
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            bounds = multiplicity_boundaries(region)
            st.caption(f"Zagęszczona siatka: mNL ∈ [0,04; 0,065], P̄ ∈ [200; 500]. Liczba kombinacji z >1 punktem: {len(region)}. Granice mają charakter przybliżony.")
        else:
            st.warning("Nie udało się wczytać preobliczonej mapy. Przykład ilustracyjny po prawej nadal jest liczony bezpośrednio z modelu.")
            st.caption("Plik multiplicity_region.csv.gz powinien znajdować się w tym samym katalogu co app.py.")

    with right:
        st.markdown("#### Przykład")
        st.latex(r"\bar P=400,\quad m_{NL}=0{,}05,\quad m_L=0{,}0125")
        _, _, L_ex, NL_ex = solve_pair(B=.45, delta=.01, eta=1.5, Pbar=400, m_l=.0125, m_nl=.05)
        df_ex = result_table(L_ex, NL_ex)
        st.dataframe(style_results(df_ex), use_container_width=True, height=260)
        st.write("Model liniowy ma **1** punkt stacjonarny. Model nieliniowy ma **3**: dwa siodłowe i jeden niestabilny.")
        st.caption("Przykład znajduje się poza podstawowym zakresem parametryzacji i nie stanowi ogólnej charakterystyki całej przestrzeni parametrów.")

    takeaway("Nieliniowość może wpływać nie tylko na liczbę i położenie punktów stacjonarnych, lecz także na ich lokalny charakter stabilności.")
    
# ---------------------------------------------------------------------
# 8. Conclusions
# ---------------------------------------------------------------------
elif page == PAGES[7]:
    page_header(
        "8. Wnioski i ograniczenia",
  
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Najważniejsze wnioski")
        st.markdown(
            """
            - Zmiana funkcji asymilacji zmienia warunki stacjonarności i nadaje warunkowi **dE/dt = 0 dwugałęziową strukturę**; bez dodatkowych założeń nie można zagwarantować jednoznaczności punktu stacjonarnego.
            - W scenariuszu bazowym oraz jednoczynnikowej analizie wrażliwości — tam, gdzie oba modele dawały ekonomicznie dopuszczalne rozwiązania — model nieliniowy miał **wyższe E∗**, **niższe u∗** i **wyższe g∗**.
            - Różnice między modelami dotyczą również dynamiki przejściowej; przy niskich wartościach E₀ w modelu nieliniowym nie zawsze udało się wyznaczyć ekonomicznie dopuszczalną stabilną ścieżkę.
            - Eksperyment eksploracyjny potwierdził, że model nieliniowy może dla wybranych kombinacji parametrów posiadać **kilka ekonomicznie dopuszczalnych punktów stacjonarnych**.
            """
        )
    with right:
        st.markdown("### Ograniczenia")
        st.markdown(
            """
            - Parametryzacja nie stanowi pełnej kalibracji empirycznej; wyniki nie opisują konkretnej gospodarki ani konkretnego ekosystemu.
            - Zastosowana nieliniowa funkcja asymilacji wciąż jest **uproszczeniem procesów regeneracji środowiska** i nie uwzględnia m.in. histerezy ani nagłych przejść między alternatywnymi stanami ekosystemu.
            - Przyjęcie **P̄ = 1 nie jest neutralną normalizacją** — wpływa na warunki istnienia punktów stacjonarnych i w analizie bazowej wyklucza ekonomicznie dopuszczalne rozwiązania na dolnej gałęzi.
            - Analiza wrażliwości była głównie **jednoczynnikowa**, a eksperyment dotyczący wielopunktowości obejmował tylko wybrany zakres parametrów mNL i P̄ przy ustalonych pozostałych parametrach; nie stanowi więc pełnej analizy całej przestrzeni parametrów.
            """
        )



# ---------------------------------------------------------------------
# 9. Simulator / commission questions
# ---------------------------------------------------------------------
elif page == PAGES[8]:
    page_header(
        "9. Symulacje",
    )

    st.markdown("### Parametry modelu")
    c1, c2, c3 = st.columns(3)
    with c1:
        eta = st.slider("η", 0.50, 3.00, BASE_ETA, 0.01)
        delta = st.slider("δ", 0.001, 0.05, BASE_DELTA, 0.001, format="%.3f")
    with c2:
        m_l = st.number_input("mL", min_value=0.001, max_value=20.0, value=BASE_M_L, step=0.05, format="%.3f")
        match_max = st.checkbox("Zachowaj jednakową maksymalną asymilację", value=True)
        if match_max:
            m_nl = 4.0 * m_l
            st.metric("mNL = 4mL", fmt(m_nl, 3))
        else:
            m_nl = st.number_input("mNL", min_value=0.001, max_value=80.0, value=BASE_M_NL, step=0.05, format="%.3f")
            st.warning("mNL ustawiono niezależnie od mL — główne kryterium porównawcze z pracy nie obowiązuje.")
    with c3:
        with st.expander("Pozostałe parametry", expanded=True):
            B = st.number_input("B", min_value=0.01, max_value=5.0, value=BASE_B, step=0.01, format="%.3f")
            Pbar = st.number_input("P̄", min_value=0.05, max_value=500.0, value=BASE_PBAR, step=0.05, format="%.3f")

    p_l, p_nl, L, NL = solve_pair(B, delta, eta, Pbar, m_l, m_nl)

    cplot, ctable = st.columns([1, 1.35], gap="large")
    with cplot:
        st.plotly_chart(assimilation_plot(Pbar, m_l, m_nl), use_container_width=True)
    with ctable:
        st.markdown("#### Punkty stacjonarne")
        table = result_table(L, NL)
        if table.empty:
            st.warning("Dla zadanych parametrów zastosowana procedura numeryczna nie wyznaczyła ekonomicznie dopuszczalnego punktu stacjonarnego.")
        else:
            st.dataframe(style_results(table), use_container_width=True, height=300)
            st.caption(f"Liniowy: {'1' if L is not None else '0'} punkt · Nieliniowy: {len(NL)} punkt(y).")

    selected_nl = None
    if NL:
        if len(NL) == 1:
            selected_nl = NL[0]
        else:
            labels = [f"Punkt {i+1}: E*={r['E']:.4f}, { {'saddle':'siodłowy','unstable':'niestabilny','other':'inna','nonhyperbolic':'niehiperboliczny'}.get(r['stability'], r['stability']) }" for i, r in enumerate(NL)]
            idx = st.selectbox("Punkt nieliniowy używany dalej w analizie dynamicznej", range(len(NL)), format_func=lambda i: labels[i])
            selected_nl = NL[idx]
            st.info("Model nieliniowy ma kilka ekonomicznie dopuszczalnych punktów. Dalsza dynamika odnosi się do wybranego punktu.")

    st.divider()
    st.markdown("### Dynamika")
    mode = st.radio("Tryb", ["Stabilna ścieżka", "Własne warunki początkowe"], horizontal=True)

    z0 = st.slider("Początkowa jakość środowiska E₀/P̄", 0.05, 0.95, 0.50, 0.01)
    E0 = float(z0 * Pbar)
    st.caption(f"Bieżąca wartość E₀ = {fmt(E0)} przy P̄ = {fmt(Pbar)}.")

    path_l = path_nl = None
    diag_l = diag_nl = {}

    if mode == "Stabilna ścieżka":
        st.caption("Aplikacja sama dobiera x₀ i u₀ odpowiadające stabilnemu kierunkowi wybranego punktu stacjonarnego.")
        try:
            if L is not None:
                path_l, diag_l = stable_path_for_result(L, p_l, "linear", E0)
            if selected_nl is not None:
                path_nl, diag_nl = stable_path_for_result(selected_nl, p_nl, "nonlinear", E0)
        except Exception as exc:
            st.warning(f"Procedura dynamiki nie zakończyła się poprawnie dla bieżącej parametryzacji: {exc}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("x₀ — liniowy", fmt(diag_l.get("x_0")))
        m2.metric("u₀ — liniowy", fmt(diag_l.get("u_0")))
        m3.metric("x₀ — nieliniowy", fmt(diag_nl.get("x_0")))
        m4.metric("u₀ — nieliniowy", fmt(diag_nl.get("u_0")))

        if L is not None and path_l is None:
            st.warning("Model liniowy: dla bieżących parametrów i E₀ procedura nie wyznaczyła stabilnej ścieżki.")
        if selected_nl is not None and path_nl is None:
            if selected_nl.get("stability") != "saddle":
                st.warning("Wybrany punkt nieliniowy nie jest punktem siodłowym, więc procedura stabilnej ścieżki używana w pracy nie ma tutaj zastosowania.")
            else:
                st.warning("Model nieliniowy: dla bieżących parametrów i E₀ procedura nie wyznaczyła ekonomicznie dopuszczalnej stabilnej ścieżki.")

    else:
        st.caption("To eksperyment poza stabilną rozmaitością. Zadane ręcznie x₀ i u₀ nie są automatycznie interpretowane jako ścieżka optymalna.")
        c1, c2, c3 = st.columns(3)
        with c1: x0 = st.number_input("x₀", min_value=0.0001, max_value=5.0, value=0.08, step=0.005, format="%.4f")
        with c2: u0 = st.slider("u₀", 0.01, 0.99, 0.55, 0.01)
        with c3: tmax = st.slider("Horyzont czasu", 2.0, 60.0, 20.0, 1.0)
        sim_signature = (round(B, 10), round(delta, 10), round(eta, 10), round(Pbar, 10), round(m_l, 10), round(m_nl, 10), round(E0, 10), round(x0, 10), round(u0, 10), round(tmax, 10), None if selected_nl is None else round(selected_nl["E"], 10))
        if st.button("Uruchom trajektorie", type="primary"):
            if L is not None:
                path_l, diag_l = free_trajectory(L, p_l, "linear", E0, x0, u0, tmax)
            if selected_nl is not None:
                path_nl, diag_nl = free_trajectory(selected_nl, p_nl, "nonlinear", E0, x0, u0, tmax)
            st.session_state["free_paths"] = (sim_signature, path_l, diag_l, path_nl, diag_nl)
        if "free_paths" in st.session_state and st.session_state["free_paths"][0] == sim_signature:
            _, path_l, diag_l, path_nl, diag_nl = st.session_state["free_paths"]

    if path_l is not None or path_nl is not None:
        steady_l_E = L["E"] if L is not None else None
        steady_nl_E = selected_nl["E"] if selected_nl is not None else None
        steady_l_x = L["x"] if L is not None else None
        steady_nl_x = selected_nl["x"] if selected_nl is not None else None
        steady_l_u = L["u"] if L is not None else None
        steady_nl_u = selected_nl["u"] if selected_nl is not None else None
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(trajectory_plot(path_l, path_nl, "E", "E(t)", steady_l_E, steady_nl_E), use_container_width=True)
        with c2: st.plotly_chart(trajectory_plot(path_l, path_nl, "x", "x(t)", steady_l_x, steady_nl_x), use_container_width=True)
        st.plotly_chart(trajectory_plot(path_l, path_nl, "u", "u(t)", steady_l_u, steady_nl_u), use_container_width=True)

    with st.expander("Szczegóły"):
        if not table.empty:
            diag_cols = ["Model", "Punkt", "R_max", "Dodatni wzrost", "TVC", "Stabilność", "Wartości własne"]
            st.dataframe(table[diag_cols], use_container_width=True)
        if diag_l:
            st.write("**Trajektoria liniowa:**", diag_l)
        if diag_nl:
            st.write("**Trajektoria nieliniowa:**", diag_nl)
