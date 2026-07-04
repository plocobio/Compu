"""
Núcleo de la simulación Monte Carlo del modelo de Ising 2D.

Hamiltoniano (J=1, k_B=1), condiciones periódicas de contorno:

    E = -J * sum_{<i,j>} s_i * s_j

donde la suma recorre cada par de espines primeros vecinos una única vez.

NOTA DE RENDIMIENTO:
Las funciones marcadas con @njit se compilan a código máquina la primera
vez que se ejecutan (Numba). Esto es imprescindible aquí: un barrido
completo (N=16,32,64,128; 10 temperaturas; 10^6 pMC) implica del orden de
2*10^11 intentos de flip de espín. En Python puro esto tardaría horas o
días; con Numba, del orden de minutos-decenas de minutos según la máquina
(usa `python -m src.main --estimar-tiempo` para calibrar en tu equipo
antes de lanzar el barrido completo).
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _seed_numba(semilla):
    """Fija la semilla del generador de números aleatorios *interno* de
    Numba (independiente del de NumPy/Python), para que las simulaciones
    sean reproducibles."""
    np.random.seed(semilla)


def inicializar_red(N):
    """
    Configuración inicial ordenada: s(i,j) = 1 para todo i, j.

    Parámetros
    ----------
    N : int
        Tamaño de la red cuadrada (N x N).

    Devuelve
    --------
    np.ndarray de forma (N, N) y dtype int8, con todos los espines +1.
    """
    return np.ones((N, N), dtype=np.int8)


@njit(cache=True)
def calcular_energia(red):
    """
    Energía total E(S) de la configuración, con condiciones periódicas
    de contorno:

        E = -J * sum_{<i,j>} s_i s_j ,   J = 1

    Cada enlace se cuenta una única vez: para cada sitio (i,j) se suman
    sus vecinos "derecha" y "abajo" (los vecinos "izquierda" y "arriba"
    ya quedan contados al visitar los sitios vecinos correspondientes).
    Con contorno periódico hay exactamente 2*N^2 enlaces en total.
    """
    N = red.shape[0]
    E = 0.0
    for i in range(N):
        for j in range(N):
            s = red[i, j]
            vecino_derecha = red[i, (j + 1) % N]
            vecino_abajo = red[(i + 1) % N, j]
            E -= s * (vecino_derecha + vecino_abajo)
    return E


@njit(cache=True)
def calcular_magnetizacion(red):
    """
    Magnetización por espín de la configuración (sin valor absoluto):

        M = (1/N^2) * sum_{i,j} s(i,j)

    El valor absoluto |M|, necesario para la ecuación (14) del enunciado,
    se aplica en la etapa de promediado (ver `barrido.py`), no aquí,
    para poder disponer también de la serie temporal "cruda" si hiciera
    falta en el futuro.
    """
    N = red.shape[0]
    total = 0
    for i in range(N):
        for j in range(N):
            total += red[i, j]
    return total / (N * N)


@njit(cache=True)
def metropolis_step(red, T):
    """
    Realiza 1 paso Monte Carlo (pMC) completo sobre la red, definido como
    N^2 intentos de actualización de espín elegido al azar (algoritmo de
    Metropolis de espín único):

      1. Elegir un sitio (i,j) al azar (uniforme).
      2. Calcular Delta_E = 2 * J * s(i,j) * sum(vecinos), J=1.
      3. Si Delta_E <= 0: aceptar el flip.
         Si Delta_E > 0: aceptar con probabilidad exp(-Delta_E / T).

    Modifica `red` in-place y también la devuelve por comodidad.
    """
    N = red.shape[0]
    for _ in range(N * N):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        s = red[i, j]
        vecinos = (red[(i - 1) % N, j] + red[(i + 1) % N, j] +
                   red[i, (j - 1) % N] + red[i, (j + 1) % N])
        dE = 2.0 * s * vecinos
        if dE <= 0.0:
            red[i, j] = -s
        elif np.random.random() < np.exp(-dE / T):
            red[i, j] = -s
    return red


@njit(cache=True)
def _simular_nucleo(red, T, n_pmc, medida_cada):
    """
    Bucle principal de la simulación, compilado íntegramente con Numba
    para minimizar el overhead de llamadas Python entre pasos. Ejecuta
    `n_pmc` pasos Monte Carlo y mide magnetización y energía cada
    `medida_cada` pasos.
    """
    n_medidas = n_pmc // medida_cada
    magnetizaciones = np.empty(n_medidas)
    energias = np.empty(n_medidas)
    idx = 0
    for paso in range(1, n_pmc + 1):
        metropolis_step(red, T)
        if paso % medida_cada == 0:
            magnetizaciones[idx] = calcular_magnetizacion(red)
            energias[idx] = calcular_energia(red)
            idx += 1
    return magnetizaciones, energias


def simular_ising(N, T, n_pmc, medida_cada, n_termalizacion=0, semilla=None):
    """
    Simula el modelo de Ising N x N a temperatura T.

    Parámetros
    ----------
    N : int
        Tamaño de la red.
    T : float
        Temperatura (en unidades de J/k_B).
    n_pmc : int
        Número total de pasos Monte Carlo a evolucionar (p.ej. 10^6).
    medida_cada : int
        Cada cuántos pMC se toma una medida (p.ej. 100 -> 10^4 medidas
        si n_pmc=10^6).
    n_termalizacion : int, opcional
        Pasos de termalización (equilibrado) descartados *antes* de
        empezar a medir. El enunciado no pide explícitamente descartar
        una fase de termalización (parte de una configuración ordenada y
        mide desde el principio); se deja en 0 por defecto para seguir
        la letra del enunciado, pero se puede activar para estudiar su
        efecto (ver limitaciones en el README).
    semilla : int o None
        Semilla para el generador aleatorio de Numba. Si es None no se
        fija semilla explícita (resultados no reproducibles).

    Devuelve
    --------
    (magnetizaciones, energias, red_final) :
        magnetizaciones : np.ndarray, serie temporal de la magnetización
            (con signo, sin valor absoluto) en cada medida.
        energias : np.ndarray, serie temporal de la energía total E(S)
            en cada medida.
        red_final : configuración de la red al finalizar la simulación.
    """
    if semilla is not None:
        _seed_numba(semilla)

    red = inicializar_red(N)

    for _ in range(n_termalizacion):
        metropolis_step(red, T)

    magnetizaciones, energias = _simular_nucleo(red, T, n_pmc, medida_cada)
    return magnetizaciones, energias, red
