"""
Punto de entrada de la simulación. Ejecuta el barrido (N, T) pedido por
el enunciado y guarda los resultados en un CSV.

Ejemplos de uso (desde la raíz del repositorio):

    # Prueba rápida para comprobar que todo funciona (segundos):
    python -m src.main --quick

    # Calibrar la velocidad de la máquina antes de lanzar el barrido
    # completo (no simula nada, solo estima el tiempo total):
    python -m src.main --estimar-tiempo

    # Barrido completo tal como pide el enunciado
    # (N=16,32,64,128; 10 T en [1.5,3.5]; 10^6 pMC; medida cada 100 pMC).
    # La rejilla de temperaturas NO es uniforme: dentro de la ventana
    # critica [2.0, 2.6] (donde teoricamente esta Tc) se usa el triple
    # de densidad de puntos que fuera de ella, y ademas cada uno de esos
    # puntos usa el triple de pMC (n_pmc x factor-pmc-critico), para
    # compensar la mayor varianza y la ralentizacion critica cerca de Tc:
    python -m src.main

    # Igual que el anterior, añadiendo un barrido fino adicional de
    # temperaturas alrededor de Tc de Onsager (mejora aun mas la
    # estimación de Tc(N)):
    python -m src.main --sweep-fino
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ising import inicializar_red, metropolis_step, _seed_numba  # noqa: E402
from src.barrido import barrido_temperaturas  # noqa: E402
from src.onsager import TC_ONSAGER  # noqa: E402


def generar_temperaturas_no_uniforme(T_min, T_max, n_T, ventana_critica,
                                      factor_densidad=3):
    """
    Genera una rejilla de temperaturas NO uniforme: dentro de
    `ventana_critica` = (Tc_min, Tc_max) la densidad de puntos (puntos
    por unidad de T) es `factor_densidad` veces mayor que fuera de ella.

    Esto concentra el esfuerzo de muestreo alrededor de la temperatura
    crítica teórica, donde el calor específico varía más rápido y es
    más difícil de resolver con una rejilla gruesa uniforme.

    `n_T` fija la densidad de referencia como si la rejilla fuese
    uniforme en todo [T_min, T_max]: densidad_base = n_T / (T_max - T_min).
    A partir de ahí se reparten los puntos por tramos: fuera de la
    ventana con densidad_base, dentro con factor_densidad * densidad_base.

    Devuelve un np.ndarray de temperaturas ordenado de menor a mayor.
    """
    Tc_min, Tc_max = ventana_critica
    if not (T_min < Tc_min < Tc_max < T_max):
        raise ValueError("la ventana critica debe quedar estrictamente "
                          "dentro de [T_min, T_max]")

    densidad_base = n_T / (T_max - T_min)

    ancho_izq = Tc_min - T_min
    ancho_der = T_max - Tc_max
    ancho_ventana = Tc_max - Tc_min

    n_izq = max(1, round(densidad_base * ancho_izq))
    n_der = max(1, round(densidad_base * ancho_der))
    n_ventana = max(2, round(factor_densidad * densidad_base * ancho_ventana))

    T_izq = np.linspace(T_min, Tc_min, n_izq, endpoint=False)
    T_ventana = np.linspace(Tc_min, Tc_max, n_ventana)
    T_der = np.linspace(Tc_max, T_max, n_der + 1)[1:]  # se excluye Tc_max: ya está en T_ventana

    return np.concatenate([T_izq, T_ventana, T_der])


def estimar_tiempo(N_values, T_values, n_pmc, ventana_critica=None,
                    factor_pmc_critico=1, n_calibracion=2000, semilla=42,
                    n_procesos=1):
    """
    Calibra el coste de un pMC en esta máquina para cada N (incluye la
    compilación JIT de Numba, que solo ocurre una vez) y extrapola el
    tiempo total del barrido completo, sin llegar a ejecutarlo.

    Tiene en cuenta que las temperaturas dentro de `ventana_critica` se
    simulan con `n_pmc * factor_pmc_critico` pasos en vez de `n_pmc`
    (ver `barrido_temperaturas`).
    """
    T_cal = float(np.mean(T_values))
    print("Calibrando velocidad de la simulacion (compilacion JIT de Numba "
          "incluida en la primera llamada, no contabilizada)...\n")

    if ventana_critica is not None:
        n_dentro = sum(1 for T in T_values if ventana_critica[0] <= T <= ventana_critica[1])
        n_fuera = len(T_values) - n_dentro
        n_pmc_total_equiv = n_fuera + n_dentro * factor_pmc_critico
    else:
        n_dentro, n_fuera = 0, len(T_values)
        n_pmc_total_equiv = n_fuera

    tiempo_por_pmc = {}
    for N in sorted(set(N_values)):
        _seed_numba(semilla)
        red = inicializar_red(N)
        metropolis_step(red, T_cal)  # fuerza la compilación, no se cronometra

        t0 = time.time()
        for _ in range(n_calibracion):
            metropolis_step(red, T_cal)
        dt = (time.time() - t0) / n_calibracion
        tiempo_por_pmc[N] = dt
        print(f"  N={N:4d}: {dt*1e3:7.4f} ms/pMC  "
              f"-> {dt*n_pmc:8.1f} s por punto fuera de la ventana critica "
              f"(n_pmc={n_pmc}), {dt*n_pmc*factor_pmc_critico:8.1f} s dentro "
              f"(n_pmc={int(n_pmc*factor_pmc_critico)})")

    total = sum(t * n_pmc * n_pmc_total_equiv for t in tiempo_por_pmc.values())
    print(f"\nRejilla de temperaturas: {n_fuera} puntos fuera de la ventana "
          f"critica + {n_dentro} puntos dentro (factor pMC x{factor_pmc_critico}).")
    print(f"Tiempo total de CPU (suma de todas las simulaciones): "
          f"~{total/60:.1f} min (~{total/3600:.2f} h)")

    if n_procesos and n_procesos > 1:
        # Estimacion ideal (reparto perfecto) y cota inferior de Amdahl: el
        # tiempo de pared no puede bajar del punto individual mas costoso
        # (tipicamente el N mayor dentro de la ventana critica).
        t_max_por_pmc = max(tiempo_por_pmc.values())
        cota_inferior = t_max_por_pmc * n_pmc * factor_pmc_critico
        paralelo_ideal = max(total / n_procesos, cota_inferior)
        print(f"Tiempo de PARED estimado con {n_procesos} procesos: "
              f"~{paralelo_ideal/60:.1f} min (~{paralelo_ideal/3600:.2f} h) "
              f"[cota inferior por la simulacion mas larga: "
              f"~{cota_inferior/60:.1f} min]")
    print("\n(Aviso: es solo una estimacion; el rendimiento real depende de "
          "la carga de la maquina y puede variar.)")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Simulacion Monte Carlo (Metropolis) del modelo de "
                     "Ising 2D en una red N x N con condiciones periodicas.")
    parser.add_argument("--N", type=int, nargs="+", default=[16, 32, 64, 128],
                         help="Tamanos de red a simular.")
    parser.add_argument("--T-min", type=float, default=1.5)
    parser.add_argument("--T-max", type=float, default=3.5)
    parser.add_argument("--n-T", type=int, default=10,
                         help="Densidad de referencia de temperaturas: numero "
                              "de puntos que habria en [T-min, T-max] si la "
                              "rejilla fuese uniforme (la rejilla real no lo "
                              "es, ver --Tc-ventana-* y --factor-densidad-critico).")
    parser.add_argument("--n-pmc", type=int, default=1_000_000,
                         help="Pasos Monte Carlo por simulacion, fuera de la "
                              "ventana critica.")
    parser.add_argument("--Tc-ventana-min", type=float, default=2.0,
                         help="Extremo inferior de la ventana critica de temperaturas.")
    parser.add_argument("--Tc-ventana-max", type=float, default=2.6,
                         help="Extremo superior de la ventana critica de temperaturas.")
    parser.add_argument("--factor-densidad-critico", type=float, default=3.0,
                         help="Cuantas veces mayor es la densidad de puntos de "
                              "T dentro de la ventana critica respecto a fuera.")
    parser.add_argument("--factor-pmc-critico", type=float, default=3.0,
                         help="Cuantas veces mayor es n_pmc dentro de la "
                              "ventana critica respecto a fuera (compensa la "
                              "mayor varianza y la ralentizacion critica cerca de Tc).")
    parser.add_argument("--medida-cada", type=int, default=100,
                         help="Cada cuantos pMC se toma una medida.")
    parser.add_argument("--n-termalizacion", type=int, default=0,
                         help="pMC de termalizacion descartados antes de medir "
                              "(el enunciado no pide ninguno; ver README).")
    parser.add_argument("--semilla", type=int, default=42,
                         help="Semilla base para reproducibilidad.")
    parser.add_argument("--n-procesos", type=int, default=os.cpu_count(),
                         help="Numero de procesos para ejecutar las "
                              "simulaciones (N, T) en paralelo (una por core). "
                              "Por defecto usa todos los cores; usa 1 para "
                              "ejecucion en serie. El resultado es identico "
                              "sea cual sea el numero de procesos.")
    parser.add_argument("--salida", type=str, default="data/resultados.csv")
    parser.add_argument("--quick", action="store_true",
                         help="Modo de prueba rapido (N y n_pmc pequenos) "
                              "solo para verificar que el codigo funciona; "
                              "los resultados NO son fisicamente fiables.")
    parser.add_argument("--estimar-tiempo", action="store_true",
                         help="Solo calibra la velocidad y estima el tiempo "
                              "total del barrido completo, sin simular.")
    parser.add_argument("--sweep-fino", action="store_true",
                         help="Anade un segundo barrido de temperaturas mas "
                              "denso alrededor de Tc de Onsager, guardado en "
                              "data/resultados_fino.csv, para mejorar la "
                              "estimacion de Tc(N).")
    parser.add_argument("--sweep-fino-ancho", type=float, default=0.3,
                         help="Semianchura (en T) del barrido fino alrededor de Tc.")
    parser.add_argument("--sweep-fino-n", type=int, default=7,
                         help="Numero de temperaturas del barrido fino.")
    args = parser.parse_args()

    ventana_critica = (args.Tc_ventana_min, args.Tc_ventana_max)

    if args.quick:
        N_values = [8, 16]
        T_values = generar_temperaturas_no_uniforme(
            args.T_min, args.T_max, n_T=4, ventana_critica=ventana_critica,
            factor_densidad=args.factor_densidad_critico)
        n_pmc = 2000
        medida_cada = 20
        print("*** MODO RAPIDO (--quick): parametros reducidos SOLO para "
              "comprobar que el codigo corre sin errores. Los resultados "
              "no son representativos de la fisica del modelo. ***\n")
    else:
        N_values = args.N
        T_values = generar_temperaturas_no_uniforme(
            args.T_min, args.T_max, n_T=args.n_T, ventana_critica=ventana_critica,
            factor_densidad=args.factor_densidad_critico)
        n_pmc = args.n_pmc
        medida_cada = args.medida_cada

    if args.estimar_tiempo:
        estimar_tiempo(N_values, T_values, n_pmc, ventana_critica=ventana_critica,
                        factor_pmc_critico=args.factor_pmc_critico,
                        semilla=args.semilla, n_procesos=args.n_procesos)
        return

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)

    print(f"Barrido principal: N={N_values}\n"
          f"T ({len(T_values)} puntos, ventana critica {ventana_critica} con "
          f"factor de densidad x{args.factor_densidad_critico}): "
          f"{list(np.round(T_values, 4))}\n"
          f"n_pmc={n_pmc} fuera de la ventana, "
          f"x{args.factor_pmc_critico} dentro; medida_cada={medida_cada}\n")
    df = barrido_temperaturas(N_values, T_values, n_pmc=n_pmc,
                               medida_cada=medida_cada,
                               n_termalizacion=args.n_termalizacion,
                               semilla=args.semilla, etiqueta="grueso",
                               ventana_critica=ventana_critica,
                               factor_pmc_critico=args.factor_pmc_critico,
                               n_procesos=args.n_procesos)
    df.to_csv(args.salida, index=False)
    print(f"\nResultados del barrido principal guardados en {args.salida}")

    if args.sweep_fino and not args.quick:
        T_finas = np.linspace(TC_ONSAGER - args.sweep_fino_ancho,
                               TC_ONSAGER + args.sweep_fino_ancho,
                               args.sweep_fino_n)
        print(f"\nBarrido fino alrededor de Tc={TC_ONSAGER:.4f}: "
              f"T en {list(np.round(T_finas, 4))}\n")
        df_fino = barrido_temperaturas(N_values, T_finas, n_pmc=n_pmc,
                                        medida_cada=medida_cada,
                                        n_termalizacion=args.n_termalizacion,
                                        semilla=args.semilla + 10_000,
                                        etiqueta="fino",
                                        n_procesos=args.n_procesos)
        salida_fino = os.path.join(os.path.dirname(args.salida) or ".",
                                    "resultados_fino.csv")
        df_fino.to_csv(salida_fino, index=False)
        print(f"\nResultados del barrido fino guardados en {salida_fino}")


if __name__ == "__main__":
    main()
