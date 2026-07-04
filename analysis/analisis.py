"""
Post-procesado de los resultados del barrido: gráficas, estimación del
punto crítico T_c(N) por el máximo del calor específico y su
extrapolación a N -> infinito, y estimación del exponente crítico beta
de la magnetización.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.onsager import TC_ONSAGER, BETA_ONSAGER, magnetizacion_onsager


def graficar_resultados(df, out_dir="figures"):
    """
    Genera y guarda en `out_dir`:
      - magnetizacion_vs_T.png   : m_N vs T para cada N, + curva de Onsager.
      - energia_por_spin_vs_T.png: <E>/N^2 vs T para cada N.
      - energia_enunciado_vs_T.png: e_N=<E>/(2N) (ec. 15 literal) vs T.
      - calor_especifico_vs_T.png: c_N vs T para cada N.

    `df` debe tener las columnas producidas por `barrido_temperaturas`.
    Solo se usan filas de la rejilla "gruesa" si existe la columna
    "malla" con ese valor (el barrido fino, si existe, se reserva para
    `estimar_Tc_por_maximo_calor`).
    """
    os.makedirs(out_dir, exist_ok=True)
    if "malla" in df.columns:
        df = df[df["malla"] == "grueso"]
    N_values = sorted(df["N"].unique())

    # --- Magnetización vs T ---
    plt.figure(figsize=(7, 5))
    for N in N_values:
        sub = df[df["N"] == N].sort_values("T")
        plt.errorbar(sub["T"], sub["m"], yerr=sub["m_err"], marker="o",
                     linestyle="-", capsize=3, label=f"N={N}")
    T_fino = np.linspace(df["T"].min(), df["T"].max(), 400)
    plt.plot(T_fino, magnetizacion_onsager(T_fino), "k--", label="Onsager (exacto)")
    plt.axvline(TC_ONSAGER, color="gray", linestyle=":", label=r"$T_c$ Onsager")
    plt.xlabel("T")
    plt.ylabel(r"$m_N$")
    plt.title("Magnetización media vs temperatura")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "magnetizacion_vs_T.png"), dpi=150)
    plt.close()

    # --- Energía vs T (dos normalizaciones) ---
    especificaciones = [
        ("e_por_spin", "e_por_spin_err", r"$\langle E \rangle / N^2$ (energía por espín)",
         "energia_por_spin_vs_T.png"),
        ("e_enunciado", "e_enunciado_err", r"$e_N = \langle E \rangle / (2N)$ (ec. 15, literal)",
         "energia_enunciado_vs_T.png"),
    ]
    for col, err_col, ylabel, fname in especificaciones:
        plt.figure(figsize=(7, 5))
        for N in N_values:
            sub = df[df["N"] == N].sort_values("T")
            plt.errorbar(sub["T"], sub[col], yerr=sub[err_col], marker="o",
                         linestyle="-", capsize=3, label=f"N={N}")
        plt.xlabel("T")
        plt.ylabel(ylabel)
        plt.title("Energía media vs temperatura")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()

    # --- Calor específico vs T ---
    plt.figure(figsize=(7, 5))
    for N in N_values:
        sub = df[df["N"] == N].sort_values("T")
        plt.errorbar(sub["T"], sub["c"], yerr=sub["c_err"], marker="o",
                     linestyle="-", capsize=3, label=f"N={N}")
    plt.axvline(TC_ONSAGER, color="gray", linestyle=":", label=r"$T_c$ Onsager")
    plt.xlabel("T")
    plt.ylabel(r"$c_N$")
    plt.title("Calor específico vs temperatura")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calor_especifico_vs_T.png"), dpi=150)
    plt.close()


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
            T1, T2, T3 = sub["T"].iloc[idx_max - 1:idx_max + 2]
            c1, c2, c3 = sub["c"].iloc[idx_max - 1:idx_max + 2]
            denom = (T1 - T2) * (T1 - T3) * (T2 - T3)
            if abs(denom) > 1e-12:
                A = (T3 * (c2 - c1) + T2 * (c1 - c3) + T1 * (c3 - c2)) / denom
                B = (T3 ** 2 * (c1 - c2) + T2 ** 2 * (c3 - c1) + T1 ** 2 * (c2 - c3)) / denom
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
        plt.figure(figsize=(7, 5))
        plt.plot(x, y, "o", label=r"$T_c(N)$ (máximo de $c_N$)")
        x_fit = np.linspace(0, x.max() * 1.15, 100)
        plt.plot(x_fit, reg.intercept + reg.slope * x_fit, "r-",
                 label=(r"ajuste $T_c(N)=T_c^{\infty}+a/N$" + "\n" +
                        fr"$T_c^\infty={Tc_extrapolado:.4f} \pm {error_Tc:.4f}$"))
        plt.axhline(TC_ONSAGER, color="gray", linestyle=":",
                    label=fr"$T_c$ Onsager = {TC_ONSAGER:.4f}")
        plt.xlabel("1/N")
        plt.ylabel(r"$T_c(N)$")
        plt.title("Extrapolación de tamaño finito de $T_c$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Tc_vs_1_N.png"), dpi=150)
        plt.close()

    return tabla, Tc_extrapolado, error_Tc


def ajuste_exponente_beta(df, N_referencia=None, Tc=TC_ONSAGER,
                           graficar=True, out_dir="figures"):
    """
    Estima el exponente crítico beta ajustando m_N ~ (Tc - T)^beta
    mediante regresión lineal de log(m_N) frente a log(Tc - T), usando
    únicamente puntos con T < Tc.

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
        aviso = ("Menos de 2 puntos utilizables con T<Tc: no es posible "
                 "realizar un ajuste fiable de beta con esta rejilla de "
                 "temperaturas. Se recomienda anadir mas temperaturas por "
                 "debajo de Tc (p.ej. con --sweep-fino o ampliando --n-T).")
        return dict(N=N_referencia, beta=np.nan, beta_err=np.nan,
                    beta_exacto=BETA_ONSAGER, puntos_usados=int(usados.sum()),
                    aviso=aviso)
    elif usados.sum() < 4:
        aviso = (f"Solo {int(usados.sum())} puntos disponibles para el "
                 "ajuste: el exponente beta obtenido es una estimacion "
                 "cualitativa, con incertidumbre sistematica grande "
                 "ademas del error estadistico indicado.")

    x = np.log(reduc[usados])
    y = np.log(m[usados])
    reg = stats.linregress(x, y)
    beta_num = reg.slope
    beta_err = reg.stderr

    if graficar:
        plt.figure(figsize=(7, 5))
        plt.plot(x, y, "o", label="puntos usados en el ajuste")
        if (~usados).any():
            plt.plot(np.log(reduc[~usados]), np.log(m[~usados]), "x",
                     color="gray", label="excluidos (redondeo cerca de $T_c$)")
        x_fit = np.linspace(x.min(), x.max(), 50)
        plt.plot(x_fit, reg.intercept + reg.slope * x_fit, "r-",
                 label=fr"ajuste: $\beta={beta_num:.3f} \pm {beta_err:.3f}$")
        plt.xlabel(r"$\log(T_c-T)$")
        plt.ylabel(r"$\log(m_N)$")
        plt.title(fr"Estimación de $\beta$ (N={N_referencia})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ajuste_beta.png"), dpi=150)
        plt.close()

    return dict(N=N_referencia, beta=beta_num, beta_err=beta_err,
                beta_exacto=BETA_ONSAGER, puntos_usados=int(usados.sum()),
                aviso=aviso)
