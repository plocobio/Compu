"""
Estimación de errores para series temporales Monte Carlo.

Las medidas sucesivas de una simulación de Monte Carlo están
correlacionadas temporalmente (el estado en el paso k depende del estado
en el paso k-1), por lo que la desviación estándar "ingenua" de la serie
dividida por sqrt(n) *subestima* el verdadero error de la media. Aquí se
implementan dos métodos que tienen en cuenta esa correlación:

1. `estimar_errores`: método de "blocking" (binning) de Flyvbjerg-Petersen
   para la media de una serie (usado para m_N y e_N).
2. `bootstrap_bloques`: bootstrap por bloques, usado para propagar el
   error a magnitudes *no lineales* derivadas de la serie, como el calor
   específico c_N = (1/(N^2 T)) [<E^2> - <E>^2], donde no basta con
   propagar errores de <E> y <E^2> por separado porque están fuertemente
   correlacionados.
"""

import numpy as np


def estimar_errores(serie):
    """
    Estima la media y el error estándar de la media de una serie temporal
    correlacionada mediante el método de blocking (Flyvbjerg-Petersen,
    1989).

    Idea: se promedia la serie en bloques de tamaño creciente (1, 2, 4,
    8, ...). Si los bloques son mayores que el tiempo de correlación, los
    datos resultantes son aproximadamente independientes y el error
    estándar calculado ingenuamente en ese nivel de bloque ya es
    correcto. Al aumentar el tamaño de bloque, el error estimado crece y
    alcanza un "plateau"; tomamos el máximo de la serie de errores como
    estimación conservadora de dicho plateau.

    Parámetros
    ----------
    serie : array_like
        Serie temporal de medidas (p.ej. magnetización o energía en cada
        medida de la simulación).

    Devuelve
    --------
    (media, error) : tuple(float, float)
    """
    datos = np.asarray(serie, dtype=np.float64)
    media = datos.mean()
    n = len(datos)
    if n < 2:
        return media, 0.0

    errores_por_nivel = []
    while len(datos) >= 2:
        n_i = len(datos)
        error_i = datos.std(ddof=1) / np.sqrt(n_i)
        errores_por_nivel.append(error_i)

        n_pares = n_i // 2
        if n_pares < 2:
            break
        datos = (datos[0:2 * n_pares:2] + datos[1:2 * n_pares:2]) / 2.0

    error = max(errores_por_nivel)
    return media, error


def _tamano_bloque_optimo(serie, minimo_bloques=8):
    """
    Estima un tamaño de bloque razonable para el bootstrap por bloques,
    identificando el nivel de blocking en el que el error se estabiliza
    (el mismo criterio que usa `estimar_errores`).
    """
    datos = np.asarray(serie, dtype=np.float64)
    errores = []
    tam = 1
    while len(datos) >= max(2, minimo_bloques):
        errores.append(datos.std(ddof=1) / np.sqrt(len(datos)))
        n_pares = len(datos) // 2
        if n_pares < minimo_bloques:
            break
        datos = (datos[0:2 * n_pares:2] + datos[1:2 * n_pares:2]) / 2.0
        tam *= 2
    nivel_plateau = int(np.argmax(errores))
    return 2 ** nivel_plateau


def bootstrap_bloques(serie, funcion, n_bootstrap=200, tam_bloque=None,
                       semilla=None):
    """
    Estima el error de una magnitud derivada `funcion(serie)` mediante
    bootstrap por bloques: la serie se divide en bloques contiguos (de
    tamaño >= tiempo de autocorrelación estimado), se generan
    `n_bootstrap` remuestreos con reemplazo de esos bloques y se evalúa
    `funcion` en cada remuestreo. El error se estima como la desviación
    estándar de los valores obtenidos.

    Esto es necesario para magnitudes no lineales (como el calor
    específico) donde no es válido propagar por separado los errores de
    cada momento estadístico involucrado, ya que están correlacionados.

    Parámetros
    ----------
    serie : array_like
        Serie temporal (p.ej. energía) sobre la que se calcula `funcion`.
    funcion : callable
        Función que toma un array 1D (una versión remuestreada de
        `serie`, del mismo tamaño total) y devuelve un escalar.
    n_bootstrap : int
        Número de remuestreos bootstrap.
    tam_bloque : int o None
        Tamaño de bloque. Si None, se estima automáticamente a partir de
        la propia serie (ver `_tamano_bloque_optimo`).
    semilla : int o None
        Semilla del generador aleatorio de NumPy usado para el
        remuestreo (independiente del RNG interno de Numba).

    Devuelve
    --------
    error : float
    """
    rng = np.random.default_rng(semilla)
    datos = np.asarray(serie, dtype=np.float64)
    n = len(datos)

    if tam_bloque is None:
        tam_bloque = max(1, _tamano_bloque_optimo(datos))

    n_bloques = max(1, n // tam_bloque)
    datos_recortados = datos[: n_bloques * tam_bloque]
    bloques = datos_recortados.reshape(n_bloques, tam_bloque)

    valores = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_bloques, size=n_bloques)
        muestra = bloques[idx].reshape(-1)
        valores[b] = funcion(muestra)

    return float(valores.std(ddof=1))
