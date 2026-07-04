"""
Barrido de temperaturas y tamaños de red: orquesta `simular_ising` sobre
la rejilla (N, T) pedida por el enunciado y calcula valores medios y
errores de magnetización, energía y calor específico.
"""

import time

import numpy as np
import pandas as pd

from src.ising import simular_ising
from src.errores import estimar_errores, bootstrap_bloques


def _calor_especifico(energias, N, T):
    """c_N = (1/(N^2 T)) [<E^2> - <E>^2], a partir de la serie de energías."""
    return (np.mean(energias ** 2) - np.mean(energias) ** 2) / (N ** 2 * T)


def barrido_temperaturas(N_values, T_values, n_pmc=1_000_000, medida_cada=100,
                          n_termalizacion=0, semilla=42, n_bootstrap=200,
                          etiqueta="grueso", ventana_critica=None,
                          factor_pmc_critico=1, verbose=True):
    """
    Recorre todas las combinaciones (N, T), simula el modelo de Ising y
    calcula valores medios y errores de:

      - m_N  : magnetización promedio, ec. (14)          -> columnas m, m_err
      - e_N  : energía normalizada como en el enunciado,
               ec. (15), <E>/(2N)                          -> e_enunciado, e_enunciado_err
        (además, energía por espín <E>/N^2, más habitual
         en la literatura, ya que la normalización de (15)
         no es intensiva para una red N x N)                -> e_por_spin, e_por_spin_err
      - c_N  : calor específico, ec. (16)                  -> c, c_err

    Los errores de m_N y e_N se calculan con blocking
    (`estimar_errores`); el de c_N con bootstrap por bloques
    (`bootstrap_bloques`), ya que c_N es una función no lineal de la
    serie de energías.

    Parámetros
    ----------
    N_values : iterable de int
        Tamaños de red a simular (p.ej. [16, 32, 64, 128]).
    T_values : iterable de float
        Temperaturas a simular (p.ej. 10 valores en [1.5, 3.5]).
    n_pmc, medida_cada, n_termalizacion, semilla :
        Ver `simular_ising`. `n_pmc` es el número de pasos Monte Carlo
        usado *fuera* de la ventana crítica (ver `ventana_critica`).
    n_bootstrap : int
        Número de remuestreos bootstrap para el error de c_N.
    etiqueta : str
        Se añade como columna "malla" al resultado, para poder
        distinguir un barrido "grueso" (rejilla principal) de un barrido
        "fino" (rejilla adicional alrededor de Tc, ver `--sweep-fino` en
        `main.py`) cuando luego se combinan varias tablas.
    ventana_critica : tuple(float, float) o None
        Si se indica (Tc_min, Tc_max), las temperaturas T tales que
        Tc_min <= T <= Tc_max se simulan con `n_pmc * factor_pmc_critico`
        pasos Monte Carlo en vez de `n_pmc`, para compensar la mayor
        varianza y la ralentización crítica cerca del punto crítico. Si
        es None (por defecto), todas las temperaturas usan `n_pmc`.
    factor_pmc_critico : float
        Factor multiplicativo aplicado a `n_pmc` dentro de la ventana
        crítica (ignorado si `ventana_critica` es None).
    verbose : bool
        Si True, imprime el progreso y una estimación del tiempo restante.

    Devuelve
    --------
    pandas.DataFrame con una fila por combinación (N, T), incluyendo la
    columna `n_pmc` con los pasos Monte Carlo realmente usados en cada
    fila (útil para verificar el refuerzo en la ventana crítica).
    """
    filas = []
    total = len(N_values) * len(T_values)
    contador = 0
    t_inicio = time.time()

    for N in N_values:
        for T in T_values:
            contador += 1
            t0 = time.time()

            if ventana_critica is not None and ventana_critica[0] <= T <= ventana_critica[1]:
                n_pmc_T = int(round(n_pmc * factor_pmc_critico))
            else:
                n_pmc_T = n_pmc

            semilla_run = None if semilla is None else int(semilla) + contador
            magnetizaciones, energias, _ = simular_ising(
                N, T, n_pmc_T, medida_cada, n_termalizacion, semilla_run)

            m_mean, m_err = estimar_errores(np.abs(magnetizaciones))

            e_mean, e_err = estimar_errores(energias)
            e_enunciado = e_mean / (2.0 * N)
            e_enunciado_err = e_err / (2.0 * N)
            e_por_spin = e_mean / (N ** 2)
            e_por_spin_err = e_err / (N ** 2)

            c_mean = _calor_especifico(energias, N, T)
            c_err = bootstrap_bloques(
                energias, lambda e: _calor_especifico(e, N, T),
                n_bootstrap=n_bootstrap, semilla=semilla_run)

            filas.append(dict(
                N=N, T=T, malla=etiqueta, n_pmc=n_pmc_T,
                m=m_mean, m_err=m_err,
                e_enunciado=e_enunciado, e_enunciado_err=e_enunciado_err,
                e_por_spin=e_por_spin, e_por_spin_err=e_por_spin_err,
                c=c_mean, c_err=c_err,
            ))

            if verbose:
                dt = time.time() - t0
                transcurrido = time.time() - t_inicio
                restante = transcurrido / contador * (total - contador)
                print(f"[{contador:3d}/{total}] N={N:4d} T={T:.4f} n_pmc={n_pmc_T:8d}  "
                      f"m={m_mean:.4f}+-{m_err:.4f}  "
                      f"e_spin={e_por_spin:.4f}+-{e_por_spin_err:.4f}  "
                      f"c={c_mean:.4f}+-{c_err:.4f}  "
                      f"({dt:.1f}s, restante ~{restante/60:.1f} min)")

    return pd.DataFrame(filas)
