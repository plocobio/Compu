"""
Script de análisis: lee los CSV generados por `src/main.py`, produce las
gráficas pedidas y muestra por pantalla la estimación del punto crítico
T_c(N->infinito) y del exponente beta, comparados con Onsager.

Uso (desde la raíz del repositorio):

    python -m analysis.run_analisis
    python -m analysis.run_analisis --datos data/resultados.csv --datos-fino data/resultados_fino.csv
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.analisis import (  # noqa: E402
    graficar_resultados, estimar_Tc_por_maximo_calor, ajuste_exponente_beta,
    graficar_ajuste_beta_cada_N, graficar_ajuste_beta_todos_N,
    graficar_panel_resumen, graficar_residuos_onsager,
    graficar_susceptibilidad, graficar_binder, graficar_colapso_magnetizacion,
    graficar_snapshots, graficar_autocorrelacion,
)
from src.onsager import TC_ONSAGER, BETA_ONSAGER  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Análisis de resultados de Ising 2D")
    parser.add_argument("--datos", type=str, default="data/resultados.csv")
    parser.add_argument("--datos-fino", type=str, default="data/resultados_fino.csv",
                         help="CSV del barrido fino alrededor de Tc (opcional).")
    parser.add_argument("--figuras", type=str, default="figures")
    parser.add_argument("--sin-extras", action="store_true",
                         help="Omite las figuras basadas en simulaciones propias "
                              "(snapshots, autocorrelacion), que lanzan "
                              "simulaciones cortas adicionales y tardan un "
                              "poco mas.")
    args = parser.parse_args()

    df = pd.read_csv(args.datos)

    print("=== Tabla de resultados (rejilla principal) ===")
    columnas = ["N", "T", "n_pmc", "m", "m_err", "e_por_spin", "e_por_spin_err",
                "e_enunciado", "e_enunciado_err", "c", "c_err"]
    print(df[columnas].to_string(index=False))

    print("\nGenerando graficas en", args.figuras)
    graficar_resultados(df, out_dir=args.figuras)

    # --- Figuras adicionales del Grupo A (solo requieren el CSV) ---
    graficar_panel_resumen(df, out_dir=args.figuras)
    graficar_residuos_onsager(df, out_dir=args.figuras)

    # --- Figuras adicionales del Grupo B (susceptibilidad y Binder) ---
    graficar_susceptibilidad(df, out_dir=args.figuras)
    graficar_binder(df, out_dir=args.figuras)

    # --- Colapso de escala (Grupo C, requiere >=2 tamaños N) ---
    graficar_colapso_magnetizacion(df, out_dir=args.figuras)

    df_tc = df
    if args.datos_fino and os.path.exists(args.datos_fino):
        df_fino = pd.read_csv(args.datos_fino)
        df_tc = pd.concat([df, df_fino], ignore_index=True)
        print(f"(Se incluye el barrido fino de {args.datos_fino} para "
              "afinar la estimacion de Tc(N).)")

    tabla_Tc, Tc_extrap, Tc_extrap_err = estimar_Tc_por_maximo_calor(
        df_tc, out_dir=args.figuras)

    print("\n=== Estimacion de Tc(N) a partir del maximo de c_N ===")
    print(tabla_Tc.to_string(index=False))
    print(f"\nTc extrapolado (ajuste Tc(N)=Tc_inf + a/N, N->infinito): "
          f"{Tc_extrap:.4f} +- {Tc_extrap_err:.4f}")
    print(f"Tc exacto (Onsager):                                    "
          f"{TC_ONSAGER:.6f}")
    print(f"Diferencia relativa: "
          f"{abs(Tc_extrap - TC_ONSAGER) / TC_ONSAGER * 100:.2f} %")

    resultado_beta = ajuste_exponente_beta(df, out_dir=args.figuras, graficar=False)
    print("\n=== Estimacion del exponente critico beta (magnetizacion) ===")
    print(f"N usado como referencia: {resultado_beta['N']}")
    print(f"Puntos usados en el ajuste: {resultado_beta['puntos_usados']}")
    print(f"beta (numerico):  {resultado_beta['beta']:.4f} +- {resultado_beta['beta_err']:.4f}")
    print(f"beta (Onsager, exacto): {BETA_ONSAGER:.4f}")
    if resultado_beta["aviso"]:
        print(f"\nAVISO: {resultado_beta['aviso']}")
    graficar_ajuste_beta_cada_N(df, out_dir=args.figuras)
    graficar_ajuste_beta_todos_N(df, out_dir=args.figuras)

    # --- Figuras adicionales del Grupo C que lanzan simulaciones propias
    #     (configuraciones, series temporales): snapshots y autocorrelacion
    #     / ralentizacion critica. ---
    if not args.sin_extras:
        print("\nGenerando figuras extra (simulaciones cortas propias): "
              "snapshots y autocorrelacion...")
        graficar_snapshots(df=df, out_dir=args.figuras)
        graficar_autocorrelacion(out_dir=args.figuras)
    else:
        print("\n(--sin-extras: se omiten snapshots y autocorrelacion.)")

    print(f"\nFiguras guardadas en: {os.path.abspath(args.figuras)}")


if __name__ == "__main__":
    main()
