# Simulación Monte Carlo del Modelo de Ising 2D

Este repositorio implementa una simulación del **modelo de Ising bidimensional** mediante el método de **Monte Carlo (algoritmo de Metropolis)**, con el objetivo de estudiar el comportamiento crítico del sistema y compararlo con el resultado exacto de Onsager.

## Descripción del ejercicio

**Voluntario 1**

Dado el tamaño de red \(N\), se parte de una configuración inicial ordenada (\(s(i,j)=1\) para todo \(i,j\)) y se deja evolucionar el sistema durante \(10^6\) pasos Monte Carlo (pMC).

Se pide obtener los valores medios y sus correspondientes errores para las siguientes magnitudes:

- **Magnetización promedio**

$$
m_N=\left\langle\left|\frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N}s(i,j)\right|\right\rangle
$$

- **Energía media**

$$
e_N=\frac{\langle E(S)\rangle}{2N}
$$

- **Calor específico**

$$
c_N=\frac{1}{N^2T}\left[\langle E(S)^2\rangle-\langle E(S)\rangle^2\right]
$$

donde \(\langle\cdot\rangle\) denota el promedio sobre medidas tomadas cada 100 pMC, es decir, sobre \(10^4\) medidas por simulación.

### Parámetros del experimento

- **Temperaturas:** 10 valores en el intervalo \(T \in [1{,}5,\ 3{,}5]\)
- **Tamaños de red:** \(N = 16,\ 32,\ 64,\ 128\)
- **Pasos Monte Carlo:** \(10^6\) por simulación
- **Frecuencia de medida:** cada 100 pMC (\(10^4\) medidas totales)

### Análisis requerido

1. Describir el comportamiento de \(m_N\), \(e_N\) y \(c_N\) en función de la temperatura y del tamaño de red. Comparar con el resultado exacto de Onsager y describir el efecto de \(N\) sobre cada magnitud.
2. Revisar en la literatura los exponentes críticos y la teoría de escala de tamaño finito (*finite-size scaling*).
3. Estimar el punto crítico: para cada \(N\), obtener \(T_c(N)\) a partir del máximo del calor específico y extrapolar su comportamiento para \(N\to\infty\).
4. Obtener numéricamente el exponente crítico \(\beta\) de la magnetización y compararlo con el valor exacto.

## Estructura del repositorio

```
.
├── src/
│   ├── ising.py        # inicializar_red, calcular_energia, calcular_magnetizacion,
│   │                    # metropolis_step, simular_ising (núcleo Numba)
│   ├── errores.py       # estimar_errores (blocking) y bootstrap_bloques (calor específico)
│   ├── onsager.py        # Tc, m(T) y beta exactos de Onsager
│   ├── barrido.py         # barrido_temperaturas: orquesta el barrido (N, T) completo
│   └── main.py             # CLI: lanza el barrido, --quick, --estimar-tiempo, --sweep-fino
├── analysis/
│   ├── analisis.py       # graficar_resultados, estimar_Tc_por_maximo_calor, ajuste_exponente_beta
│   └── run_analisis.py    # CLI: lee los CSV, genera gráficas y el resumen final
├── data/                    # CSV de resultados (generados, no versionados salvo .gitkeep)
├── figures/                  # PNG de las gráficas (generados, no versionados salvo .gitkeep)
└── README.md
```

## Requisitos

```
pip install -r requirements.txt
```

- Python ≥ 3.10
- `numpy`, `pandas`, `matplotlib`, `scipy`, `numba` (imprescindible para que la simulación sea viable en tiempo razonable, ver más abajo)

## Uso

Todos los comandos se ejecutan desde la raíz del repositorio.

**1. Prueba rápida** (segundos; solo verifica que el código corre sin errores, los resultados no son físicamente representativos):

```
python -m src.main --quick
python -m analysis.run_analisis
```

**2. Calibrar el tiempo de ejecución en tu máquina** antes de lanzar el barrido completo (no simula nada, solo estima):

```
python -m src.main --estimar-tiempo
```

**3. Barrido completo** tal como pide el enunciado (N=16,32,64,128; 10 T en [1.5,3.5]; 10⁶ pMC; medida cada 100 pMC). **Aviso de coste computacional:** esto puede tardar del orden de una a varias horas según la máquina (la mayor parte del coste la impone N=128); usa el paso 2 para estimarlo antes de lanzarlo, y considera ejecutarlo en segundo plano:

```
python -m src.main
```

Parámetros configurables (ver `python -m src.main --help`): `--N`, `--T-min`, `--T-max`, `--n-T`, `--n-pmc`, `--medida-cada`, `--semilla`, `--salida`.

Opcionalmente, añade un segundo barrido más fino de temperaturas alrededor de \(T_c\) de Onsager (mejora la estimación de \(T_c(N)\), ya que con solo 10 temperaturas la rejilla es muy gruesa):

```
python -m src.main --sweep-fino
```

**4. Análisis y gráficas**, a partir de los CSV generados en `data/`:

```
python -m analysis.run_analisis
```

Genera en `figures/`: `magnetizacion_vs_T.png`, `energia_por_spin_vs_T.png`, `energia_enunciado_vs_T.png`, `calor_especifico_vs_T.png`, `Tc_vs_1_N.png`, `ajuste_beta.png`; e imprime por pantalla la tabla de resultados, la estimación de \(T_c(N\to\infty)\) y la estimación numérica de \(\beta\), comparadas con Onsager.

## Notas de diseño

- **Reproducibilidad:** semilla configurable (`--semilla`); Numba usa un generador aleatorio propio (independiente de NumPy), fijado explícitamente en cada simulación.
- **Normalización de la energía:** el enunciado define \(e_N=\langle E\rangle/(2N)\) (ec. 15), pero esa normalización no es intensiva para una red \(N\times N\) (el número de enlaces es \(2N^2\), no \(2N\)). El código calcula **ambas** cantidades: `e_enunciado` (literal, ec. 15) y `e_por_spin` = \(\langle E\rangle/N^2\) (energía por espín, la magnitud intensiva estándar), para poder comparar.
- **Errores:** las medidas Monte Carlo sucesivas están correlacionadas, así que no se usa la desviación estándar ingenua. `m_N` y `e_N` usan el método de *blocking* (Flyvbjerg–Petersen); `c_N`, al ser una función no lineal de la serie de energías, usa *bootstrap* por bloques.
- **Termalización:** por defecto no se descarta ninguna fase de equilibrado (`--n-termalizacion 0`), siguiendo la letra del enunciado (se parte de la configuración ordenada y se mide desde el principio). Ver la sección de limitaciones más abajo.

## Referencias

- L. Onsager, *Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition*, Phys. Rev. **65**, 117 (1944).
- N. Metropolis et al., *Equation of State Calculations by Fast Computing Machines*, J. Chem. Phys. **21**, 1087 (1953).
- Literatura sobre exponentes críticos y *finite-size scaling* del modelo de Ising 2D.
