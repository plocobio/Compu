"""
Barrido de temperaturas y tamaños de red: orquesta `simular_ising` sobre
la rejilla (N, T) pedida por el enunciado y calcula valores medios y
errores de magnetización, energía, calor específico, susceptibilidad y
cumulante de Binder.

Cada punto (N, T) es una simulación independiente, así que el barrido se
ejecuta EN PARALELO repartiendo los puntos entre los cores de la CPU con
`ProcessPoolExecutor` (ver `n_procesos` en `barrido_temperaturas`). La
semilla de cada punto se deriva de su índice fijo en el orden (N, T), de
modo que el resultado es idéntico bit a bit al de la ejecución en serie,
independientemente del número de procesos.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from src.ising import simular_ising
from src.errores import estimar_errores, bootstrap_bloques


def _calor_especifico(energias, N, T):
    """c_N = (1/(N^2 T)) [<E^2> - <E>^2], a partir de la serie de energías."""
    return (np.mean(energias ** 2) - np.mean(energias) ** 2) / (N ** 2 * T)


def _susceptibilidad(magnetizaciones, N, T):
    """
    Susceptibilidad magnética por espín, a partir de la serie de
    magnetizaciones (por espín, con signo):

        chi_N = (N^2 / T) [ <m^2> - <|m|>^2 ]

    Se usa <|m|> (no <m>) porque en una red finita por debajo de Tc el
    signo global fluctúa; la convención con valor absoluto es la habitual
    para que chi no diverja artificialmente por el cambio de signo.
    """
    m_abs = np.abs(magnetizaciones)
    return (N ** 2 / T) * (np.mean(m_abs ** 2) - np.mean(m_abs) ** 2)


def _binder(magnetizaciones):
    """
    Cumulante de Binder de cuarto orden:

        U_4 = 1 - <m^4> / (3 <m^2>^2)

    Es adimensional y (en el límite de N grande) tiende a 0 para T>Tc y a
    2/3 para T<Tc; las curvas U_4(T) de distintos N se cruzan en Tc, lo
    que permite localizar el punto crítico sin extrapolar en 1/N.
    """
    m2 = np.mean(magnetizaciones ** 2)
    m4 = np.mean(magnetizaciones ** 4)
    return 1.0 - m4 / (3.0 * m2 ** 2)


def _simular_punto(tarea):
    """
    Ejecuta UNA simulación (N, T) y devuelve la fila de resultados (valores
    medios y errores de m, e, c, chi, binder).

    Es el "worker" que se ejecuta en cada proceso al paralelizar el barrido.
    Está a nivel de módulo y recibe una tupla de argumentos simples (todos
    picklables) porque `ProcessPoolExecutor` en macOS arranca los procesos
    con el método 'spawn', que reimporta el módulo y necesita que la función
    y sus argumentos sean serializables e importables desde el top-level.

    `tarea` = (N, T, n_pmc_T, medida_cada, n_termalizacion, semilla_run,
               n_bootstrap, etiqueta)
    """
    (N, T, n_pmc_T, medida_cada, n_termalizacion, semilla_run,
     n_bootstrap, etiqueta) = tarea

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

    # Susceptibilidad y cumulante de Binder: ambos son funciones no lineales
    # de la serie de magnetizaciones, así que su error se propaga con
    # bootstrap por bloques (igual que c_N).
    chi_mean = _susceptibilidad(magnetizaciones, N, T)
    chi_err = bootstrap_bloques(
        magnetizaciones, lambda mm: _susceptibilidad(mm, N, T),
        n_bootstrap=n_bootstrap, semilla=semilla_run)
    binder_mean = _binder(magnetizaciones)
    binder_err = bootstrap_bloques(
        magnetizaciones, _binder,
        n_bootstrap=n_bootstrap, semilla=semilla_run)

    return dict(
        N=N, T=T, malla=etiqueta, n_pmc=n_pmc_T,
        m=m_mean, m_err=m_err,
        e_enunciado=e_enunciado, e_enunciado_err=e_enunciado_err,
        e_por_spin=e_por_spin, e_por_spin_err=e_por_spin_err,
        c=c_mean, c_err=c_err,
        chi=chi_mean, chi_err=chi_err,
        binder=binder_mean, binder_err=binder_err,
    )


def _log_progreso(k, total, fila, t_inicio):
    """Imprime el progreso de un punto ya completado (k de total)."""
    transcurrido = time.time() - t_inicio
    restante = transcurrido / k * (total - k)
    print(f"[{k:3d}/{total}] N={fila['N']:4d} T={fila['T']:.4f} "
          f"n_pmc={fila['n_pmc']:8d}  "
          f"m={fila['m']:.4f}+-{fila['m_err']:.4f}  "
          f"e_spin={fila['e_por_spin']:.4f}+-{fila['e_por_spin_err']:.4f}  "
          f"c={fila['c']:.4f}+-{fila['c_err']:.4f}  "
          f"(transcurrido {transcurrido/60:.1f} min, "
          f"restante ~{restante/60:.1f} min)")


def barrido_temperaturas(N_values, T_values, n_pmc=1_000_000, medida_cada=100,
                          n_termalizacion=0, semilla=42, n_bootstrap=200,
                          etiqueta="grueso", ventana_critica=None,
                          factor_pmc_critico=1, verbose=True, n_procesos=None):
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
    n_procesos : int o None
        Número de procesos para ejecutar las simulaciones (N, T) EN PARALELO
        (una tarea por punto de la rejilla). Cada simulación es independiente,
        así que se reparten entre los cores con `ProcessPoolExecutor`. Si es
        None se usa `os.cpu_count()`; si es 1 se ejecuta en serie (útil para
        depurar). La semilla de cada punto depende solo de su índice fijo en
        el orden (N, T), no del orden en que terminan los procesos, de modo
        que el resultado es idéntico bit a bit al de la ejecución en serie.

    Devuelve
    --------
    pandas.DataFrame con una fila por combinación (N, T), incluyendo la
    columna `n_pmc` con los pasos Monte Carlo realmente usados en cada
    fila (útil para verificar el refuerzo en la ventana crítica). El orden
    de las filas es el mismo (for N: for T:) que en serie.
    """
    # Se construye la lista de tareas en el MISMO orden (for N: for T:) que
    # la versión en serie, y la semilla de cada punto se deriva de su índice
    # fijo -> los resultados no dependen del planificador de procesos.
    tareas = []
    for idx, (N, T) in enumerate((N, T) for N in N_values for T in T_values):
        if ventana_critica is not None and ventana_critica[0] <= T <= ventana_critica[1]:
            n_pmc_T = int(round(n_pmc * factor_pmc_critico))
        else:
            n_pmc_T = n_pmc
        semilla_run = None if semilla is None else int(semilla) + idx + 1
        tareas.append((N, T, n_pmc_T, medida_cada, n_termalizacion,
                       semilla_run, n_bootstrap, etiqueta))

    total = len(tareas)
    if n_procesos is None:
        n_procesos = os.cpu_count() or 1
    n_procesos = max(1, min(int(n_procesos), total))
    t_inicio = time.time()

    if n_procesos == 1:
        # Ruta en serie (idéntica numéricamente; sin overhead de procesos).
        filas = []
        for k, tarea in enumerate(tareas, 1):
            filas.append(_simular_punto(tarea))
            if verbose:
                _log_progreso(k, total, filas[-1], t_inicio)
        return pd.DataFrame(filas)

    if verbose:
        print(f"Ejecutando {total} simulaciones (N, T) en paralelo con "
              f"{n_procesos} procesos...")

    # Warm-up del JIT en el proceso padre: compila las funciones @njit y
    # puebla el cache en disco (cache=True) para que los procesos hijos lo
    # carguen en vez de recompilar todos a la vez la primera ejecución.
    T_cal = float(np.mean(np.asarray(list(T_values), dtype=float)))
    simular_ising(int(min(N_values)), T_cal, 1, 1, 0, semilla)

    # Los resultados se recolocan por índice de tarea para que el orden de
    # las filas sea idéntico al de la versión en serie.
    filas_por_idx = [None] * total
    with ProcessPoolExecutor(max_workers=n_procesos) as ejecutor:
        futuros = {ejecutor.submit(_simular_punto, tarea): i
                   for i, tarea in enumerate(tareas)}
        for k, futuro in enumerate(as_completed(futuros), 1):
            i = futuros[futuro]
            filas_por_idx[i] = futuro.result()
            if verbose:
                _log_progreso(k, total, filas_por_idx[i], t_inicio)

    return pd.DataFrame(filas_por_idx)
