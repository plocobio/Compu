"""
Resultados exactos de Onsager (1944) para el modelo de Ising 2D en el
límite termodinámico (N -> infinito), usados como referencia para
comparar con los resultados de la simulación Monte Carlo.
"""

import numpy as np

#: Temperatura crítica exacta, en unidades de J/k_B (J=1, k_B=1).
TC_ONSAGER = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ~= 2.269185314213

#: Exponente crítico exacto de la magnetización espontánea en 2D.
BETA_ONSAGER = 1.0 / 8.0


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
