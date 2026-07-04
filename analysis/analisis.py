"""
Post-procesado de los resultados del barrido: gráficas, estimación del
punto crítico T_c(N) por el máximo del calor específico y su
extrapolación a N -> infinito, y estimación del exponente crítico beta
de la magnetización.
"""

import os

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from scipy import stats

from src.ising import simular_ising
from src.onsager import (
    energia_interna_onsager,
    magnetizacion_onsager,
    BETA_ONSAGER,
    TC_ONSAGER,
    NU_ONSAGER,
)

# Paleta y marcadores personalizados (no son los de ciclo por defecto de
# matplotlib: ni los colores "C0..C9" ni el marcador "o" repetido). Se usan
# en los plots que NO llevan el tratamiento "cada tamaño / todos los
# tamaños" con _PALETA_N (Tc_vs_1_N, residuos, colapso, autocorrelación).
_COLORS = ["#66c2a5", "#3288bd", "#8338EC", "#2A9D8F", "#BC4749", "#F4A261"]
_MARKERS = ["D", "^", "P", "s", "X", "v"]

# Paleta cualitativa (colores en HEX, elegidos por el usuario) para colorear
# por N en los plots con tratamiento "cada tamaño / todos los tamaños".
_PALETA_N = [
    "#66c2a5",
    "#3288bd",
    "#8338EC",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
]


def _serie(i):
    return _COLORS[i % len(_COLORS)], _MARKERS[i % len(_MARKERS)]


def _color_marker_por_N(N, N_values):
    """
    Color (paleta `_PALETA_N`) y marcador para el tamaño `N`, según su
    posición dentro de `N_values` (todos los tamaños presentes en los
    datos). Usar la posición dentro del conjunto COMPLETO —no solo los que
    se dibujan en esta llamada— asegura que un mismo N tenga siempre el
    mismo color tanto en su figura individual como en la de "todos los
    tamaños".
    """
    N_values = sorted(N_values)
    idx = N_values.index(N)
    color = _PALETA_N[idx % len(_PALETA_N)]
    marker = _MARKERS[idx % len(_MARKERS)]
    return color, marker


def config_ax(ax):
    ax.grid(which="major", color="#DDDDDD", linewidth=0.8, zorder=-1)
    ax.grid(which="minor", color="#DEDEDE", linestyle=":", linewidth=0.5, zorder=-1)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def setup_paper_plt(plt, latex=True, scaling: float = 1.2):
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": latex,
            "font.family": "serif",
            "text.latex.preamble": "\n".join(
                [
                    r"\usepackage[utf8]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{siunitx}",
                    r"\usepackage{physics}",
                ]
            ),
        }
    )
    BIGGER_SIZE = 11 * scaling
    BIGGEST_SIZE = 14 * scaling

    plt.rc("font", size=BIGGER_SIZE)
    plt.rc("axes", titlesize=BIGGEST_SIZE)
    plt.rc("axes", labelsize=BIGGEST_SIZE)
    plt.rc("xtick", labelsize=BIGGEST_SIZE)
    plt.rc("ytick", labelsize=BIGGEST_SIZE)
    plt.rc("legend", fontsize=BIGGER_SIZE)
    plt.rc("figure", titlesize=BIGGEST_SIZE)


setup_paper_plt(plt, latex=True)
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=_COLORS)


def graficar_resultados(df, out_dir="figures"):
    """
    Genera, para cada magnitud, UNA figura por cada tamaño N presente en
    los datos ("<nombre>_N{N}.png") MÁS una figura combinada con todos los
    tamaños superpuestos ("<nombre>.png").

    Magnitudes: magnetizacion_vs_T, energia_por_spin_vs_T,
    energia_enunciado_vs_T, calor_especifico_vs_T.

    `df` debe tener las columnas producidas por `barrido_temperaturas`.
    Solo se usan filas de la rejilla "gruesa" si existe la columna
    "malla" con ese valor (el barrido fino, si existe, se reserva para
    `estimar_Tc_por_maximo_calor`). Los colores son los de `_PALETA_N`,
    asignados por posición de N dentro de todos los tamaños disponibles
    (mismo color en la figura individual de un N y en la combinada).
    """
    os.makedirs(out_dir, exist_ok=True)
    if "malla" in df.columns:
        df = df[df["malla"] == "grueso"]
    N_values = sorted(df["N"].unique())
    combos = [(N_values, "")] + [([N], f"_N{N}") for N in N_values]

    especificaciones = [
        dict(
            col="m",
            err="m_err",
            ylabel=r"$m_N$",
            titulo="Magnetización media vs temperatura",
            fname="magnetizacion_vs_T",
            f_onsager=magnetizacion_onsager,
            mostrar_onsager=True,
            mostrar_Tc=True,
        ),
        dict(
            col="e_por_spin",
            err="e_por_spin_err",
            ylabel=r"$\langle E \rangle / N^2$ (energía por espín)",
            titulo="Energía media vs temperatura",
            fname="energia_por_spin_vs_T",
            f_onsager=energia_interna_onsager,
            mostrar_onsager=True,
            mostrar_Tc=True,
        ),
        dict(
            col="e_enunciado",
            err="e_enunciado_err",
            ylabel=r"$e_N = \langle E \rangle / (2N)$",
            titulo="Energía media vs temperatura",
            fname="energia_enunciado_vs_T",
            f_onsager=None,
            mostrar_onsager=False,
            mostrar_Tc=False,
        ),
        dict(
            col="c",
            err="c_err",
            ylabel=r"$c_N$",
            titulo="Calor específico vs temperatura",
            fname="calor_especifico_vs_T",
            f_onsager=None,
            mostrar_onsager=False,
            mostrar_Tc=True,
        ),
    ]

    T_fino = np.linspace(df["T"].min(), df["T"].max(), 400)

    for espec in especificaciones:
        for N_subset, sufijo in combos:
            fig, ax = plt.subplots(figsize=(7, 5))
            for N in N_subset:
                sub = df[df["N"] == N].sort_values("T")
                color, marker = _color_marker_por_N(N, N_values)
                ax.errorbar(
                    sub["T"],
                    sub[espec["col"]],
                    yerr=sub[espec["err"]],
                    marker=marker,
                    markersize=4,
                    linestyle="-",
                    linewidth=1.2,
                    capsize=3,
                    color=color,
                    label=rf"$N={N}$",
                )
            if espec["mostrar_onsager"]:
                ax.plot(
                    T_fino,
                    espec["f_onsager"](T_fino),
                    color="black",
                    linestyle="--",
                    linewidth=1.4,
                    label="Onsager (exacto)",
                )
            if espec["mostrar_Tc"]:
                ax.axvline(
                    TC_ONSAGER,
                    color="gray",
                    linestyle=":",
                    linewidth=1.2,
                    label=r"$T_c$ Onsager",
                )
            ax.set_xlabel(r"$T$")
            ax.set_ylabel(espec["ylabel"])
            ax.set_title(espec["titulo"])
            config_ax(ax)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{espec['fname']}{sufijo}.png"), dpi=300)
            plt.close(fig)


def estimar_Tc_por_maximo_calor(df, out_dir="figures", graficar=True):
    """
    Para cada N, estima T_c(N) como la temperatura del máximo del calor
    específico c_N(T), refinada mediante interpolación parabólica de los
    3 puntos de la rejilla alrededor del máximo (útil porque solo hay
    ~10 temperaturas, muy espaciadas).

    Si `df` contiene una rejilla "fina" (columna "malla"=="fino"), se
    combina con la gruesa para localizar el máximo con más resolución.

    Después ajusta la ley de escalado de tamaño finito con nu=1:

        T_c(N) = T_c^inf + a / N

    mediante regresión lineal de T_c(N) frente a 1/N, y extrapola a
    N -> infinito.

    Devuelve
    --------
    (tabla, Tc_extrapolado, error_Tc_extrapolado)
        tabla : DataFrame con columnas N, Tc_N, c_max.
    """
    os.makedirs(out_dir, exist_ok=True)

    filas = []
    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N].sort_values("T").reset_index(drop=True)
        idx_max = int(sub["c"].idxmax())

        if 0 < idx_max < len(sub) - 1:
            T1, T2, T3 = sub["T"].iloc[idx_max - 1 : idx_max + 2]
            c1, c2, c3 = sub["c"].iloc[idx_max - 1 : idx_max + 2]
            denom = (T1 - T2) * (T1 - T3) * (T2 - T3)
            if abs(denom) > 1e-12:
                A = (T3 * (c2 - c1) + T2 * (c1 - c3) + T1 * (c3 - c2)) / denom
                B = (T3**2 * (c1 - c2) + T2**2 * (c3 - c1) + T1**2 * (c2 - c3)) / denom
                Tc_N = -B / (2 * A) if A != 0 else sub["T"].iloc[idx_max]
            else:
                Tc_N = sub["T"].iloc[idx_max]
        else:
            # el máximo cae en un extremo de la rejilla: no se puede
            # interpolar de forma fiable, se usa el punto de la rejilla.
            Tc_N = sub["T"].iloc[idx_max]

        filas.append(dict(N=N, Tc_N=float(Tc_N), c_max=float(sub["c"].iloc[idx_max])))

    tabla = pd.DataFrame(filas)

    x = 1.0 / tabla["N"].values.astype(float)
    y = tabla["Tc_N"].values.astype(float)
    reg = stats.linregress(x, y)
    Tc_extrapolado = reg.intercept
    error_Tc = reg.intercept_stderr

    if graficar:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(
            x,
            y,
            marker="D",
            markersize=7,
            linestyle="none",
            color=_COLORS[0],
            label=r"$T_c(N)$ (máximo de $c_N$)",
        )
        x_fit = np.linspace(0, x.max() * 1.15, 100)
        ax.plot(
            x_fit,
            reg.intercept + reg.slope * x_fit,
            color=_COLORS[1],
            linestyle="-",
            linewidth=1.4,
            label=(
                r"ajuste $T_c(N)=T_c^{\infty}+a/N$"
                + "\n"
                + rf"$T_c^\infty={Tc_extrapolado:.4f} \pm {error_Tc:.4f}$"
            ),
        )
        ax.axhline(
            TC_ONSAGER,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label=rf"$T_c$ Onsager = {TC_ONSAGER:.4f}",
        )
        ax.set_xlabel(r"$1/N$")
        ax.set_ylabel(r"$T_c(N)$")
        ax.set_title(r"Extrapolación de tamaño finito de $T_c$")
        config_ax(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "Tc_vs_1_N.png"), dpi=300)
        plt.close(fig)

    return tabla, Tc_extrapolado, error_Tc


def _calcular_ajuste_beta(df, N_referencia, Tc=TC_ONSAGER):
    """
    Cálculo numérico (sin graficar) del ajuste log(m_N) vs log(Tc-T) para
    una única N: aplica la misma selección/exclusión de puntos que
    `ajuste_exponente_beta` y devuelve un dict con:
      x, y            : arrays SOLO de los puntos usados en el ajuste
                        (el punto más cercano a Tc, si se excluye, no se
                        incluye aquí).
      beta, beta_err  : pendiente e incertidumbre de la regresión.
      intercept       : ordenada en el origen de la regresión.
      puntos_usados, aviso : igual que en `ajuste_exponente_beta`.
    Si hay menos de 2 puntos utilizables, x/y quedan vacíos y
    beta/beta_err/intercept son NaN.
    """
    sub = df[df["N"] == N_referencia].sort_values("T").copy()
    sub = sub[sub["T"] < Tc]

    reduc = (Tc - sub["T"].values).astype(float)
    m = sub["m"].values.astype(float)

    orden = np.argsort(reduc)
    usados = np.ones(len(reduc), dtype=bool)
    if len(reduc) > 3:
        usados[orden[0]] = False  # el más cercano a Tc: redondeo de tamaño finito

    aviso = None
    if usados.sum() < 2:
        aviso = (
            "Menos de 2 puntos utilizables con T<Tc: no es posible "
            "realizar un ajuste fiable de beta con esta rejilla de "
            "temperaturas. Se recomienda anadir mas temperaturas por "
            "debajo de Tc (p.ej. con --sweep-fino o ampliando --n-T)."
        )
        return dict(
            N=N_referencia,
            x=np.array([]),
            y=np.array([]),
            beta=np.nan,
            beta_err=np.nan,
            intercept=np.nan,
            puntos_usados=int(usados.sum()),
            aviso=aviso,
        )
    elif usados.sum() < 4:
        aviso = (
            f"Solo {int(usados.sum())} puntos disponibles para el "
            "ajuste: el exponente beta obtenido es una estimacion "
            "cualitativa, con incertidumbre sistematica grande "
            "ademas del error estadistico indicado."
        )

    x = np.log(reduc[usados])
    y = np.log(m[usados])
    reg = stats.linregress(x, y)
    return dict(
        N=N_referencia,
        x=x,
        y=y,
        beta=reg.slope,
        beta_err=reg.stderr,
        intercept=reg.intercept,
        puntos_usados=int(usados.sum()),
        aviso=aviso,
    )


def _graficar_ajuste_beta_individual(r, N_values, out_dir="figures"):
    """Dibuja y guarda 'ajuste_beta_N{N}.png' para un resultado `r` de
    `_calcular_ajuste_beta`. No dibuja el punto excluido cerca de Tc. No
    hace nada (y devuelve False) si hay menos de 2 puntos usables."""
    if r["puntos_usados"] < 2:
        return False
    color, marker = _color_marker_por_N(r["N"], N_values)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        r["x"],
        r["y"],
        marker=marker,
        markersize=4,
        linestyle="none",
        color=color,
        label="puntos usados en el ajuste",
    )
    x_fit = np.linspace(r["x"].min(), r["x"].max(), 50)
    ax.plot(
        x_fit,
        r["intercept"] + r["beta"] * x_fit,
        color="black",
        linestyle="-",
        linewidth=1.4,
        label=rf"ajuste: $\beta={r['beta']:.3f} \pm {r['beta_err']:.3f}$",
    )
    ax.set_xlabel(r"$\log(T_c-T)$")
    ax.set_ylabel(r"$\log(m_N)$")
    ax.set_title(rf"Estimación de $\beta$ (N={r['N']})")
    config_ax(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"ajuste_beta_N{r['N']}.png"), dpi=300)
    plt.close(fig)
    return True


def ajuste_exponente_beta(
    df, N_referencia=None, Tc=TC_ONSAGER, graficar=True, out_dir="figures"
):
    """
    Estima el exponente crítico beta ajustando m_N ~ (Tc - T)^beta
    mediante regresión lineal de log(m_N) frente a log(Tc - T), usando
    únicamente puntos con T < Tc, para UNA red de referencia (N_referencia).

    Si `graficar=True` genera "ajuste_beta_N{N_referencia}.png". Para una
    figura por CADA N presente en los datos ver `graficar_ajuste_beta_cada_N`;
    para la versión con todos los tamaños superpuestos en un solo plot ver
    `graficar_ajuste_beta_todos_N` (genera "ajuste_beta.png"). El punto
    excluido por redondeo cerca de Tc NO se dibuja (solo se descarta del
    ajuste, en silencio).

    Elección de puntos (ver también el aviso devuelto):
      - Se usa por defecto la red más grande disponible (N_referencia),
        porque para N grande la magnetización se aproxima mejor al
        comportamiento del límite termodinámico lejos de Tc.
      - Se excluye el punto más cercano a Tc, porque en una red finita
        la transición está "redondeada" (no hay singularidad real) y
        ese punto no sigue la ley de potencias del límite termodinámico.
      - Se usa el Tc EXACTO de Onsager como referencia por defecto
        (parámetro `Tc`), para no mezclar el error de la estimación de
        Tc(N) con el del exponente; se puede pasar el Tc extrapolado
        numéricamente si se prefiere.

    ADVERTENCIA: el enunciado solo pide 10 temperaturas en [1.5, 3.5],
    lo que deja muy pocos puntos con T < Tc (típicamente 3-4), y el más
    próximo a Tc ya se descarta. El ajuste resultante debe interpretarse
    como una estimación *cualitativa* del orden de magnitud de beta, no
    como una medida precisa; el resultado incluye un aviso explícito
    cuando el número de puntos usados es bajo.

    Devuelve
    --------
    dict con: N, beta, beta_err, beta_exacto, puntos_usados, aviso.
    """
    os.makedirs(out_dir, exist_ok=True)

    if "malla" in df.columns:
        df = df[df["malla"] == "grueso"]

    if N_referencia is None:
        N_referencia = int(df["N"].max())

    r = _calcular_ajuste_beta(df, N_referencia, Tc)

    if graficar:
        N_values = sorted(df["N"].unique())
        _graficar_ajuste_beta_individual(r, N_values, out_dir)

    return dict(
        N=N_referencia,
        beta=r["beta"],
        beta_err=r["beta_err"],
        beta_exacto=BETA_ONSAGER,
        puntos_usados=r["puntos_usados"],
        aviso=r["aviso"],
    )


def graficar_ajuste_beta_cada_N(df, Tc=TC_ONSAGER, out_dir="figures"):
    """
    Genera "ajuste_beta_N{N}.png" para CADA tamaño N presente en los datos
    (equivalente a llamar `ajuste_exponente_beta` una vez por N). Los N con
    menos de 2 puntos usables (T<Tc) se omiten con un aviso. Devuelve la
    lista de N para los que sí se generó figura.
    """
    os.makedirs(out_dir, exist_ok=True)
    if "malla" in df.columns:
        df = df[df["malla"] == "grueso"]
    N_values = sorted(df["N"].unique())

    generados = []
    for N in N_values:
        r = _calcular_ajuste_beta(df, N, Tc)
        if _graficar_ajuste_beta_individual(r, N_values, out_dir):
            generados.append(N)
    if not generados:
        print(
            "[ajuste_beta (por N)] omitido: ningún N tiene suficientes "
            "puntos con T<Tc para el ajuste."
        )
    return generados


def graficar_ajuste_beta_todos_N(df, Tc=TC_ONSAGER, out_dir="figures"):
    """
    Versión de "todos los tamaños" del ajuste de beta: superpone, para cada
    N presente en los datos, los puntos log(m_N) vs log(Tc-T) usados en el
    ajuste (sin dibujar el excluido cerca de Tc) junto con su recta de
    ajuste individual, coloreados por N (`_PALETA_N`). Permite comparar
    visualmente cómo depende la estimación de beta del tamaño de red.
    Guarda "ajuste_beta.png". Los N sin suficientes puntos (<2 con T<Tc)
    se omiten. Devuelve True si pudo graficar algo, False en caso contrario.
    """
    os.makedirs(out_dir, exist_ok=True)
    if "malla" in df.columns:
        df = df[df["malla"] == "grueso"]
    N_values = sorted(df["N"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    algo_graficado = False
    for N in N_values:
        r = _calcular_ajuste_beta(df, N, Tc)
        if r["puntos_usados"] < 2:
            continue
        algo_graficado = True
        color, marker = _color_marker_por_N(N, N_values)
        ax.plot(
            r["x"], r["y"], marker=marker, markersize=4, linestyle="none", color=color
        )
        x_fit = np.linspace(r["x"].min(), r["x"].max(), 50)
        ax.plot(
            x_fit,
            r["intercept"] + r["beta"] * x_fit,
            color=color,
            linestyle="-",
            linewidth=1.2,
            label=rf"$N={N}$: $\beta={r['beta']:.3f}\pm{r['beta_err']:.3f}$",
        )

    if not algo_graficado:
        print(
            "[ajuste_beta (todos los tamaños)] omitido: ningún N tiene "
            "suficientes puntos con T<Tc para el ajuste."
        )
        plt.close(fig)
        return False

    ax.set_xlabel(r"$\log(T_c-T)$")
    ax.set_ylabel(r"$\log(m_N)$")
    ax.set_title(r"Estimación de $\beta$ para todos los tamaños")
    config_ax(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ajuste_beta.png"), dpi=300)
    plt.close(fig)
    return True


# =====================================================================
#  Plots adicionales (Grupo A): resumen multipanel y residuos vs Onsager
# =====================================================================


def graficar_panel_resumen(df, out_dir="figures"):
    """
    Figura resumen de 3 paneles apilados (a, b, c) que comparten el eje T:
      (a) magnetización m_N + curva exacta de Onsager,
      (b) energía por espín <E>/N^2 + energía interna exacta de Onsager,
      (c) calor específico c_N.

    Es el formato habitual de artículo: cuenta toda la fenomenología de la
    transición en un único lienzo, evitando repetir tres figuras casi
    idénticas y facilitando la comparación visual a T fija.

    Genera UNA figura por cada tamaño N ("figura_resumen_N{N}.png") MÁS
    una figura combinada con todos los tamaños ("figura_resumen.png").
    """
    os.makedirs(out_dir, exist_ok=True)
    if "malla" in df.columns:
        df = df[df["malla"] == "grueso"]
    N_values = sorted(df["N"].unique())
    combos = [(N_values, "")] + [([N], f"_N{N}") for N in N_values]
    T_fino = np.linspace(df["T"].min(), df["T"].max(), 400)

    for N_subset, sufijo in combos:
        fig, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=True)
        ax_m, ax_e, ax_c = axes

        for N in N_subset:
            sub = df[df["N"] == N].sort_values("T")
            color, marker = _color_marker_por_N(N, N_values)
            kw = dict(
                marker=marker,
                markersize=4,
                linestyle="-",
                linewidth=1.2,
                capsize=3,
                color=color,
                label=rf"$N={N}$",
            )
            ax_m.errorbar(sub["T"], sub["m"], yerr=sub["m_err"], **kw)
            ax_e.errorbar(sub["T"], sub["e_por_spin"], yerr=sub["e_por_spin_err"], **kw)
            ax_c.errorbar(sub["T"], sub["c"], yerr=sub["c_err"], **kw)

        ax_m.plot(
            T_fino,
            magnetizacion_onsager(T_fino),
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="Onsager (exacto)",
        )
        ax_e.plot(
            T_fino,
            energia_interna_onsager(T_fino),
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="Onsager (exacto)",
        )
        for ax in axes:
            ax.axvline(TC_ONSAGER, color="gray", linestyle=":", linewidth=1.2)
            config_ax(ax)

        ax_m.set_ylabel(r"$m_N$")
        ax_e.set_ylabel(r"$\langle E \rangle / N^2$")
        ax_c.set_ylabel(r"$c_N$")
        ax_c.set_xlabel(r"$T$")
        ax_m.legend(frameon=False)

        # etiquetas de panel (a, b, c) en la esquina superior izquierda
        for ax, etiqueta in zip(axes, "abc"):
            ax.text(
                0.02,
                0.94,
                rf"$\mathbf{{({etiqueta})}}$",
                transform=ax.transAxes,
                fontweight="bold",
                va="top",
                ha="left",
            )

        ax_m.set_title("Resumen de la transición de fase del Ising 2D")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"figura_resumen{sufijo}.png"), dpi=300)
        plt.close(fig)


def graficar_residuos_onsager(df, out_dir="figures"):
    """
    Residuo m_N(T) - m_Onsager(T) frente a T, con su banda de error.

    Mirar dos curvas casi superpuestas (simulación vs Onsager) oculta
    dónde y cuánto difieren. El residuo lo hace explícito: revela el
    "redondeo" de tamaño finito de la transición alrededor de T_c, que es
    justamente el efecto físico interesante en una red N x N finita.
    """
    os.makedirs(out_dir, exist_ok=True)
    if "malla" in df.columns:
        df = df[df["malla"] == "grueso"]
    N_values = sorted(df["N"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, N in enumerate(N_values):
        sub = df[df["N"] == N].sort_values("T")
        residuo = sub["m"].values - magnetizacion_onsager(sub["T"].values)
        color, marker = _serie(i)
        ax.errorbar(
            sub["T"],
            residuo,
            yerr=sub["m_err"],
            marker=marker,
            markersize=6,
            linestyle="-",
            linewidth=1.2,
            capsize=3,
            color=color,
            label=rf"$N={N}$",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(
        TC_ONSAGER, color="gray", linestyle=":", linewidth=1.2, label=r"$T_c$ Onsager"
    )
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$m_N - m_{\mathrm{Onsager}}$")
    ax.set_title("Desviación de la magnetización respecto a Onsager")
    config_ax(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "residuos_magnetizacion_vs_T.png"), dpi=300)
    plt.close(fig)


# =====================================================================
#  Plots adicionales (Grupo B): susceptibilidad y cumulante de Binder
#  (requieren las columnas 'chi'/'binder', producidas por el barrido
#   actualizado; si no existen se avisa y se omite el plot.)
# =====================================================================


def graficar_susceptibilidad(df, out_dir="figures"):
    """
    Susceptibilidad magnética chi_N(T) frente a T para cada N.

    Su pico diverge (crece con N) en T_c, lo que proporciona una segunda
    estimación independiente de T_c(N) —complementaria al máximo de c_N—
    y da acceso al exponente crítico gamma.

    Genera UNA figura por cada tamaño N ("susceptibilidad_vs_T_N{N}.png")
    MÁS una figura combinada con todos los tamaños
    ("susceptibilidad_vs_T.png"). Devuelve True si pudo graficar, False si
    faltan los datos (columna 'chi').
    """
    if "chi" not in df.columns:
        print(
            "[susceptibilidad] omitida: el CSV no contiene la columna 'chi'. "
            "Vuelve a ejecutar 'python -m src.main' para regenerar los datos."
        )
        return False
    os.makedirs(out_dir, exist_ok=True)
    if "malla" in df.columns:
        df = df[df["malla"] == "grueso"]
    N_values = sorted(df["N"].unique())
    combos = [(N_values, "")] + [([N], f"_N{N}") for N in N_values]

    for N_subset, sufijo in combos:
        fig, ax = plt.subplots(figsize=(7, 5))
        for N in N_subset:
            sub = df[df["N"] == N].sort_values("T")
            color, marker = _color_marker_por_N(N, N_values)
            ax.errorbar(
                sub["T"],
                sub["chi"],
                yerr=sub["chi_err"],
                marker=marker,
                markersize=4,
                linestyle="-",
                linewidth=1.2,
                capsize=3,
                color=color,
                label=rf"$N={N}$",
            )
        ax.axvline(
            TC_ONSAGER,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label=r"$T_c$ Onsager",
        )
        ax.set_xlabel(r"$T$")
        ax.set_ylabel(r"$\chi_N$")
        ax.set_title("Susceptibilidad magnética vs temperatura")
        config_ax(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"susceptibilidad_vs_T{sufijo}.png"), dpi=300)
        plt.close(fig)
    return True


def graficar_binder(df, out_dir="figures"):
    """
    Cumulante de Binder U_4(T) = 1 - <m^4>/(3<m^2>^2) para cada N.

    Es el método de referencia para localizar T_c: las curvas de distintos
    N se CRUZAN en T_c (punto fijo del grupo de renormalización), sin
    necesidad de extrapolar en 1/N. Tiende a 2/3 para T<Tc y a 0 para T>Tc.

    Genera UNA figura por cada tamaño N ("binder_vs_T_N{N}.png") MÁS una
    figura combinada con todos los tamaños ("binder_vs_T.png"). Devuelve
    True si pudo graficar, False si faltan los datos (columna 'binder').
    """
    if "binder" not in df.columns:
        print(
            "[binder] omitido: el CSV no contiene la columna 'binder'. "
            "Vuelve a ejecutar 'python -m src.main' para regenerar los datos."
        )
        return False
    os.makedirs(out_dir, exist_ok=True)
    if "malla" in df.columns:
        df = df[df["malla"] == "grueso"]
    N_values = sorted(df["N"].unique())
    combos = [(N_values, "")] + [([N], f"_N{N}") for N in N_values]

    for N_subset, sufijo in combos:
        fig, ax = plt.subplots(figsize=(7, 5))
        for N in N_subset:
            sub = df[df["N"] == N].sort_values("T")
            color, marker = _color_marker_por_N(N, N_values)
            ax.errorbar(
                sub["T"],
                sub["binder"],
                yerr=sub["binder_err"],
                marker=marker,
                markersize=4,
                linestyle="-",
                linewidth=1.2,
                capsize=3,
                color=color,
                label=rf"$N={N}$",
            )
        ax.axhline(
            2.0 / 3.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=r"$2/3$ (fase ordenada)",
        )
        ax.axvline(
            TC_ONSAGER,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label=r"$T_c$ Onsager",
        )
        ax.set_xlabel(r"$T$")
        ax.set_ylabel(r"$U_4$")
        ax.set_title("Cumulante de Binder vs temperatura")
        config_ax(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"binder_vs_T{sufijo}.png"), dpi=300)
        plt.close(fig)
    return True


# =====================================================================
#  Plot adicional (Grupo C): colapso de escala de tamaño finito
# =====================================================================


def graficar_colapso_magnetizacion(df, Tc=TC_ONSAGER, out_dir="figures"):
    """
    Colapso de escala de tamaño finito de la magnetización: se representa

        m_N * N^{beta/nu}   frente a   (T - Tc) * N^{1/nu}

    con los exponentes exactos de Onsager (beta=1/8, nu=1). Si la hipótesis
    de escalado es correcta, las curvas de todos los N caen sobre una única
    "curva maestra". Es la figura canónica de fenómenos críticos y valida
    los exponentes. Requiere al menos 2 tamaños N; si solo hay uno, se
    avisa y se omite. Devuelve True/False según pudo graficar.
    """
    os.makedirs(out_dir, exist_ok=True)
    if "malla" in df.columns:
        df = df[df["malla"] == "grueso"]
    N_values = sorted(df["N"].unique())
    if len(N_values) < 2:
        print(
            f"[colapso] omitido: se necesitan >=2 tamaños N y solo hay "
            f"{N_values}. Simula varios N (p.ej. 'python -m src.main "
            f"--N 16 32 64 128') para habilitar el colapso."
        )
        return False

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, N in enumerate(N_values):
        sub = df[df["N"] == N].sort_values("T")
        x = (sub["T"].values - Tc) * N ** (1.0 / NU_ONSAGER)
        y = sub["m"].values * N ** (BETA_ONSAGER / NU_ONSAGER)
        yerr = sub["m_err"].values * N ** (BETA_ONSAGER / NU_ONSAGER)
        color, marker = _serie(i)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=marker,
            markersize=6,
            linestyle="-",
            linewidth=1.0,
            capsize=3,
            color=color,
            label=rf"$N={N}$",
        )
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.2, label=r"$T=T_c$")
    ax.set_xlabel(r"$(T-T_c)\,N^{1/\nu}$")
    ax.set_ylabel(r"$m_N\,N^{\beta/\nu}$")
    ax.set_title(r"Colapso de escala de la magnetización ($\beta=1/8,\ \nu=1$)")
    config_ax(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "colapso_magnetizacion.png"), dpi=300)
    plt.close(fig)
    return True


# =====================================================================
#  Plots adicionales (Grupo C): requieren simulaciones propias
#  (configuraciones y series temporales), no el CSV del barrido.
#  Se lanzan simulaciones cortas y dedicadas dentro de cada función.
# =====================================================================


def _tres_temperaturas(Tc=TC_ONSAGER):
    """Temperaturas representativas: ordenada (T<Tc), crítica (~Tc) y
    desordenada (T>Tc), para las figuras cualitativas del Grupo C."""
    return [1.5, round(Tc, 3), 3.5]


# Mapa de dos colores (de _PALETA_N, alto contraste) para +-1 espín.
_CMAP_ESPINES = matplotlib.colors.ListedColormap([_PALETA_N[3], _PALETA_N[4]])


def _dibujar_red(ax, red, T, titulo_extra=""):
    ax.imshow(red, cmap=_CMAP_ESPINES, interpolation="nearest", vmin=-1, vmax=1)
    ax.set_title(rf"$T={T:g}$" + (f"\n({titulo_extra})" if titulo_extra else ""))
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_visible(True)
        lado.set_linewidth(1.0)


def _leyenda_espines(fig):
    """Leyenda indicando qué representa cada color (espín +1 / -1)."""
    handles = [
        Patch(facecolor=_PALETA_N[3], edgecolor="black", label=r"espín $s=-1$"),
        Patch(facecolor=_PALETA_N[4], edgecolor="black", label=r"espín $s=+1$"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )


def _rect_con_margen_superior(
    alto_fig_pulgadas, margen_pulgadas=0.9, margen_inferior=0.06
):
    """
    `rect` para `tight_layout` que reserva `margen_pulgadas` de alto FIJO
    (en pulgadas) para el suptitle, sea cual sea el número de filas del
    grid: sin esto, el suptitle se solapa con el título (a 2 líneas) de la
    fila superior en figuras de una sola fila, o queda con demasiado hueco
    en figuras de muchas filas.
    """
    return (0, margen_inferior, 1, 1 - margen_pulgadas / alto_fig_pulgadas)


def graficar_snapshots(
    df=None,
    N_values_todos=None,
    temperaturas=None,
    n_pmc=4000,
    medida_cada=50,
    semilla=7,
    out_dir="figures",
):
    """
    Configuraciones instantáneas de la red de espines a T<Tc, T~Tc y T>Tc.

    Visualiza directamente la física: dominios ordenados a baja T, grandes
    clústeres autosemejantes (fractales) en la criticidad, y desorden
    microscópico a alta T. Los dos colores son dos colores de `_PALETA_N`
    de alto contraste; una leyenda indica qué espín (+1 o -1) representa
    cada uno.

    Genera, para cada N presente en `df` (o en `N_values_todos` si no se
    pasa `df`), una figura individual "snapshots_configuracion_N{N}.png"
    (grid 1 x 3, una columna por temperatura representativa), MÁS una
    figura combinada "snapshots_configuracion.png" con todos los tamaños
    (un grid con una fila por N y una columna por temperatura).

    Parámetros
    ----------
    df : DataFrame o None
        Si se pasa, sus valores únicos de "N" determinan los tamaños a
        graficar. No se usan sus columnas de resultados, solo los N
        disponibles: las configuraciones se generan con simulaciones
        propias, cortas y dedicadas.
    N_values_todos : iterable de int o None
        Alternativa a `df` para fijar los tamaños a graficar (p.ej. si no
        se dispone de CSV). Por defecto (16,32,64,128).
    """
    os.makedirs(out_dir, exist_ok=True)
    if temperaturas is None:
        temperaturas = _tres_temperaturas()
    titulos_T = ["ordenada", "crítica", "desordenada"]

    if df is not None:
        N_disponibles = sorted(df["N"].unique())
    else:
        N_disponibles = sorted(N_values_todos or (16, 32, 64, 128))

    # --- (a) una figura individual por cada N ---
    alto_unico = 4.6
    for fila_semilla, N in enumerate(N_disponibles):
        fig, axes = plt.subplots(
            1, len(temperaturas), figsize=(4 * len(temperaturas), alto_unico)
        )
        if len(temperaturas) == 1:
            axes = [axes]
        for k, (ax, T) in enumerate(zip(axes, temperaturas)):
            _, _, red = simular_ising(
                N, T, n_pmc, medida_cada, semilla=semilla + fila_semilla * 10 + k
            )
            etiqueta = titulos_T[k] if k < len(titulos_T) else ""
            _dibujar_red(ax, red, T, etiqueta)
        fig.suptitle(rf"Configuraciones de espines del Ising 2D ($N={N}$)")
        _leyenda_espines(fig)
        fig.tight_layout(rect=_rect_con_margen_superior(alto_unico))
        fig.savefig(os.path.join(out_dir, f"snapshots_configuracion_N{N}.png"), dpi=300)
        plt.close(fig)

    # --- (b) versión con todos los tamaños ---
    n_filas = len(N_disponibles)
    alto_todos = 4.2 * n_filas
    fig, axes = plt.subplots(
        n_filas,
        len(temperaturas),
        figsize=(4 * len(temperaturas), alto_todos),
        squeeze=False,
    )
    for fila, N in enumerate(N_disponibles):
        for k, T in enumerate(temperaturas):
            _, _, red = simular_ising(
                N, T, n_pmc, medida_cada, semilla=semilla + fila * 10 + k
            )
            etiqueta = titulos_T[k] if (fila == 0 and k < len(titulos_T)) else ""
            _dibujar_red(axes[fila][k], red, T, etiqueta)
        axes[fila][0].set_ylabel(rf"$N={N}$", fontsize=12)
    fig.suptitle("Configuraciones de espines del Ising 2D (todos los tamaños)")
    _leyenda_espines(fig)
    fig.tight_layout(rect=_rect_con_margen_superior(alto_todos, margen_inferior=0.04))
    fig.savefig(os.path.join(out_dir, "snapshots_configuracion.png"), dpi=300)
    plt.close(fig)


def _autocorrelacion_normalizada(serie, max_lag):
    """Función de autocorrelación normalizada rho(k)=C(k)/C(0) de una
    serie temporal, para k=0..max_lag."""
    x = np.asarray(serie, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    c0 = np.dot(x, x) / n
    if c0 == 0:
        return np.zeros(max_lag + 1)
    rho = np.array(
        [np.dot(x[: n - k], x[k:]) / ((n - k) * c0) for k in range(max_lag + 1)]
    )
    return rho


def _tiempo_autocorrelacion_integrado(rho):
    """Tiempo de autocorrelación integrado tau_int = 1/2 + sum_{k>=1} rho(k),
    truncando la suma en el primer cruce por cero de rho (ventana
    automática robusta al ruido de la cola)."""
    tau = 0.5
    for k in range(1, len(rho)):
        if rho[k] <= 0:
            break
        tau += rho[k]
    return tau


def graficar_autocorrelacion(
    N=32, temperaturas=None, n_pmc=30000, max_lag=400, semilla=23, out_dir="figures"
):
    """
    Ralentización crítica ("critical slowing down"). Genera dos figuras:

      - autocorrelacion_vs_lag.png: rho(k) de la magnetización frente al
        desfase k (en pMC) para T<Tc, T~Tc y T>Tc. Cerca de Tc rho(k)
        decae mucho más lentamente: las medidas están mucho más
        correlacionadas.
      - tiempo_autocorrelacion_vs_T.png: tau_int(T) a lo largo de un
        barrido de temperaturas, con un pico pronunciado en torno a Tc.

    Explica visualmente POR QUÉ los errores crecen cerca de Tc y por qué
    hacen falta métodos como blocking/bootstrap (justifica src/errores.py).
    Aquí se mide cada pMC (medida_cada=1) para resolver la autocorrelación
    a escala de un paso Monte Carlo.
    """
    os.makedirs(out_dir, exist_ok=True)
    if temperaturas is None:
        temperaturas = _tres_temperaturas()

    # --- (1) rho(k) para las tres temperaturas representativas ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, T in enumerate(temperaturas):
        magnetizaciones, _, _ = simular_ising(
            N, T, n_pmc, medida_cada=1, semilla=semilla + i
        )
        rho = _autocorrelacion_normalizada(np.abs(magnetizaciones), max_lag)
        color, marker = _serie(i)
        # marcador en cada desfase k medido (sin submuestrear), con la línea
        # de fondo para guiar el ojo en el decaimiento.
        ax.plot(
            np.arange(max_lag + 1),
            rho,
            linestyle="-",
            linewidth=1.0,
            marker=marker,
            markersize=3,
            markeredgewidth=0.0,
            color=color,
            label=rf"$T={T:g}$",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"desfase $k$ (pMC)")
    ax.set_ylabel(r"$\rho(k)$")
    ax.set_title(rf"Autocorrelación de $|m|$ ($N={N}$)")
    config_ax(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "autocorrelacion_vs_lag.png"), dpi=300)
    plt.close(fig)

    # --- (2) tau_int(T) a lo largo de un barrido de temperaturas ---
    T_barrido = np.linspace(1.6, 3.4, 15)
    taus = []
    for j, T in enumerate(T_barrido):
        magnetizaciones, _, _ = simular_ising(
            N, float(T), n_pmc, medida_cada=1, semilla=semilla + 100 + j
        )
        rho = _autocorrelacion_normalizada(np.abs(magnetizaciones), max_lag)
        taus.append(_tiempo_autocorrelacion_integrado(rho))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        T_barrido,
        taus,
        marker="D",
        markersize=6,
        linestyle="-",
        linewidth=1.2,
        color=_COLORS[0],
        label=r"$\tau_{\mathrm{int}}(T)$",
    )
    ax.axvline(
        TC_ONSAGER, color="gray", linestyle=":", linewidth=1.2, label=r"$T_c$ Onsager"
    )
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$ (pMC)")
    ax.set_title(rf"Tiempo de autocorrelación vs temperatura ($N={N}$)")
    config_ax(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tiempo_autocorrelacion_vs_T.png"), dpi=300)
    plt.close(fig)
