"""
Resultados exactos de Onsager (1944) para el modelo de Ising 2D en el
límite termodinámico (N -> infinito), usados como referencia para
comparar con los resultados de la simulación Monte Carlo.
"""

import numpy as np
from scipy.special import ellipk

#: Temperatura crítica exacta, en unidades de J/k_B (J=1, k_B=1).
TC_ONSAGER = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ~= 2.269185314213

#: Exponente crítico exacto de la magnetización espontánea en 2D.
BETA_ONSAGER = 1.0 / 8.0

#: Exponente crítico exacto de la longitud de correlación en 2D.
NU_ONSAGER = 1.0

#: Exponente crítico exacto de la susceptibilidad en 2D.
GAMMA_ONSAGER = 7.0 / 4.0


def magnetizacion_onsager(T):
    """
    Magnetización espontánea exacta del modelo de Ising 2D:

        m(T) = [1 - sinh(2/T)^-4]^(1/8)   si T < Tc
        m(T) = 0                           si T >= Tc

    Acepta un escalar o un array de temperaturas.
    """
    T_arr = np.atleast_1d(np.asarray(T, dtype=np.float64))
    m = np.zeros_like(T_arr)
    mask = T_arr < TC_ONSAGER
    m[mask] = (1.0 - np.sinh(2.0 / T_arr[mask]) ** (-4.0)) ** (1.0 / 8.0)
    return m.item() if m.size == 1 else m


def energia_interna_onsager(T):
    """
    Energía interna por espín exacta del modelo de Ising 2D (Onsager),
    con J=1, k_B=1:

        u(T) = -coth(2/T) [ 1 + (2/pi) (2 tanh^2(2/T) - 1) K(k) ]

    donde k = 2 sinh(2/T) / cosh^2(2/T) es el módulo y K(k) la integral
    elíptica completa de primera especie. `scipy.special.ellipk` recibe el
    parámetro m = k^2 (no el módulo k), de ahí el `k**2` en la llamada.

    Esta u(T) es directamente comparable con la columna `e_por_spin`
    (= <E>/N^2) de la simulación. Acepta escalar o array de temperaturas.
    """
    T_arr = np.atleast_1d(np.asarray(T, dtype=np.float64))
    beta = 1.0 / T_arr
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        s = np.sinh(2.0 * beta)
        c = np.cosh(2.0 * beta)
        coth = c / s
        k = 2.0 * s / c ** 2
        tanh2 = np.tanh(2.0 * beta) ** 2
        u = -coth * (1.0 + (2.0 / np.pi) * (2.0 * tanh2 - 1.0) * ellipk(k ** 2))
    return u.item() if u.size == 1 else u
