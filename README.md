# Túnel cuántico macroscópico en una unión Josephson

Código del proyecto final de Mecánica Cuántica (UACH, Ingeniería Física).
Calcula los niveles de energía de una unión Josephson polarizada por corriente
resolviendo la ecuación de Schrödinger por diferencias finitas. También compara
el potencial completo con la aproximación cúbica que se usa en el cálculo WKB.

## Modelo

La diferencia de fase $\gamma$ se comporta como una partícula de masa
$m_\gamma = C(\hbar/2e)^2$ en el potencial

$$U(\gamma) = -E_J\left(\cos\gamma + \frac{I}{I_c}\gamma\right),\qquad E_J = \frac{\hbar I_c}{2e}.$$

## Archivos

```
josephson.py              Programa principal (niveles, barrido en I/Ic, comparación cúbica)
oscilador_armonico.py     Validación del método con el oscilador armónico (solución exacta)
validacion_I0.py          Validación con el potencial Josephson sin corriente (I = 0)
escape.py                 Tasas de escape cuántico y térmico, temperatura de escape
requirements.txt          numpy, scipy, matplotlib
tests/                    Pruebas automáticas
resultados/               Salidas (se generan al ejecutar)
```

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

```powershell
python josephson.py
```

Por defecto usa $I_c = 9.489\,\mu$A, $C = 6.35$ pF e $I = 9.40\,\mu$A. Para usar
otros parámetros (en unidades SI):

```powershell
python josephson.py --ic 9.489e-6 --capacitancia 6.35e-12 --corriente 9.40e-6
```

Otras opciones: `--s-min`, `--s-max` y `--puntos-barrido` controlan el barrido
en $I/I_c$, y `--salida` cambia la carpeta de resultados. Escribe
`python josephson.py -h` para ver la ayuda.

Para la validación con el oscilador armónico:

```powershell
python oscilador_armonico.py
```

Para la validación con el potencial Josephson sin corriente ($I = 0$, pequeñas
oscilaciones), que genera `resultados/validacion_I0/` (`resumen_I0.txt`,
`tabla_I0.tex` y `validacion_I0.png`):

```powershell
python validacion_I0.py
```

Para las tasas de escape y la temperatura de escape, que generan
`resultados/escape/` (`resumen_escape.txt`, `tabla_parametros.tex`,
`escape_vs_T.csv` y `escape_vs_T.png`):

```powershell
python escape.py            # sin disipación
python escape.py --Q 30     # con la corrección por disipación (Devoret et al.)
```

## Salidas (`resultados/josephson/`)

| Archivo | Contenido |
|---|---|
| `resumen.txt` | $E_J$, $m_\gamma$, $\omega_p$, $\Delta U$, exponente WKB, $T^*$, niveles con incertidumbre y comparación con los modelos armónico y cúbico |
| `potencial_josephson_simple_zoom.png` | Potencial con su mínimo (negro) y máximo (rojo), Figura 2.2 |
| `comparacion_josephson_armonico_zoom.png` | Potencial Josephson contra la aproximación armónica, Figura 2.3 |
| `regiones_wkb.png` | Las tres regiones del cálculo WKB (pozo, bajo la barrera, onda saliente) sobre el potencial cúbico |
| `pozo_y_niveles.png` | Potencial con los niveles y densidades $\lvert\psi_n\rvert^2$ |
| `estabilizacion_frontera.png` | Autovalores contra la posición de la pared derecha |
| `niveles_vs_corriente.png` / `.csv` | Niveles del pozo contra $I/I_c$ |
| `comparacion_cubico.png` / `.csv` | Potencial completo contra cúbico: barrera y exponente WKB contra $I/I_c$ |

## Método numérico

1. **Diferencias finitas.** $\psi''$ se aproxima con la fórmula centrada de
   tres puntos, con error $O(\Delta\gamma^2)$. El hamiltoniano queda como una
   matriz tridiagonal simétrica, que se diagonaliza con
   `scipy.linalg.eigh_tridiagonal`. Se usan 200 puntos entre el mínimo y el
   máximo del pozo.
2. **Estados cuasiligados.** El potencial inclinado no tiene estados ligados
   verdaderos. La pared izquierda se coloca en el máximo izquierdo, que en la
   práctica es una barrera infinita. La pared derecha se coloca *más allá* de
   la barrera, en 9 posiciones distintas. Solo se aceptan los estados con
   probabilidad mayor que 0.9 dentro del pozo. Se reporta la mediana sobre las
   posiciones de la pared, y su intervalo máx − mín se reporta como
   incertidumbre.
3. **Error de malla.** Se estima repitiendo el cálculo con la mitad del paso.
   La convergencia es de orden 2, como se espera.
4. **Comparaciones.**
   - Oscilador armónico, $\hbar\omega_p(n+\tfrac12)$.
   - Potencial cúbico resuelto con el mismo método.
   - Exponente WKB $B = \frac{2}{\hbar}\int\sqrt{2m_\gamma U}\,d\gamma$,
     calculado por integración numérica en ambos potenciales.

## Pruebas

```powershell
python -m unittest discover tests
```

Las pruebas verifican lo siguiente:
- las fórmulas cerradas de $\omega_p$ y $\Delta U$;
- que en un pozo profundo el nivel base tienda al del oscilador armónico;
- la estabilidad de los niveles respecto a la pared;
- la convergencia de segundo orden;
- que la integral WKB en el pozo cúbico reproduzca $\frac{36}{5}\Delta U_{cub}/\hbar\omega_p$;
- que el potencial cúbico sobrestime la barrera y converja al exacto cuando $I \to I_c$.


## Figuras y tablas del reporte

| Reporte | Comando | Archivo generado |
|---|---|---|
| Figura 2.1 (circuito) | No la genera el código | — |
| Figura 2.2 | `python josephson.py` | `resultados/josephson/potencial_josephson_simple_zoom.png` |
| Figura 2.3 | `python josephson.py` | `resultados/josephson/comparacion_josephson_armonico_zoom.png` |
| Sec. 3.2 (oscilador armónico) | `python oscilador_armonico.py` | `resultados/informe.txt` |
| Figura 3.1 y Tabla 3.1 | `python validacion_I0.py` | `resultados/validacion_I0/validacion_I0.png`, `tabla_I0.tex` |
| Figura 4.1 | `python josephson.py` | `resultados/josephson/regiones_wkb.png` |
| Tabla 5.1 | `python escape.py` | `resultados/escape/tabla_parametros.tex` (número de niveles en `resultados/josephson/resumen.txt`) |
| Figura 5.1 | `python josephson.py` | `resultados/josephson/pozo_y_niveles.png` |
| Tabla 5.2 | `python josephson.py` | `resultados/josephson/resumen.txt` |
| Figura 5.2 | `python josephson.py` | `resultados/josephson/niveles_vs_corriente.png` |
| Figura 5.3 | `python josephson.py` | `resultados/josephson/comparacion_cubico.png` |
| Figura 5.4 | `python escape.py` | `resultados/escape/escape_vs_T.png` |
| Tabla 5.3 (con disipación) | `python escape.py --Q 30` | `resultados/escape/resumen_escape.txt` |