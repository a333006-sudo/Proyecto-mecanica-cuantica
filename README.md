# Schrödinger 1D: oscilador armónico y pozo Josephson

Este programa valida el Hamiltoniano estacionario de diferencias finitas antes
de reutilizarlo con otro potencial. Calcula cuatro niveles con
`scipy.linalg.eigh_tridiagonal`; **no es todavía una implementación de
Crank–Nicolson**, que propaga la función de onda en el tiempo. La diagonalización
proporciona una referencia para validar posteriormente esa propagación.

## Pozo Josephson

El script separado `niveles_energia_josephson.py` importa `resolver_hamiltoniano`
y `normalizar_y_agregar_fronteras` del ejemplo del oscilador. El código del
oscilador y su lanzador se conservan.

Dentro de esta carpeta (`Proyecto mecanica cuantica`), ejecuta:

```powershell
.\ejecutar_josephson.cmd
```

También puedes hacer doble clic en `ejecutar_josephson.cmd`. Para guardar los
resultados sin abrir la ventana de la gráfica:

```powershell
.\ejecutar_josephson.cmd --sin-mostrar
```

El lanzador utiliza el entorno `.venv` ya preparado. La ejecución directa con
ese intérprete es:

```powershell
.\.venv\Scripts\python.exe niveles_energia_josephson.py --sin-mostrar
```

Los valores iniciales son `I_c=9.489e-6 A`, `C=6.35e-12 F`, `I=9.40e-6 A`.
Las opciones `--ic`, `--capacitancia` y `--corriente` permiten cambiarlos, siempre
en SI. Se requiere `0 < I < I_c` y `C > 0`.

El potencial completo es `U(gamma)=-E_J*(cos(gamma)+s*gamma)`, con
`s=I/I_c`, `E_J=hbar*I_c/(2*e)` y `m_gamma=C*(hbar/(2*e))**2`.
La coordenada es la fase: `m_gamma` tiene unidades J s², no kg.
El dominio inicial es `[-pi-alpha, pi-alpha]`, `alpha=arcsin(s)`, con extremos
Dirichlet. El Hamiltoniano recibe el potencial `U(gamma)-U(alpha)` en J y
`c=hbar**2/(2*m_gamma*dgamma**2)`. Las funciones se normalizan con la medida
de fase, `dgamma*sum(abs(psi)**2)=1`, incluidos los ceros de frontera.

Se generan `resultados_josephson/informe.txt` y
`resultados_josephson/niveles_y_densidades.png`. La figura amplía la región del
pozo, muestra los niveles entre sus puntos de retorno clásicos y compara sus
densidades. La zona sombreada está fuera del intervalo de diagonalización;
allí solo se prolonga el potencial para mostrar la forma del máximo.

Resultados con los parámetros iniciales y `N=4095`:

| n | E [J], desde el mínimo | E/h [GHz] | Bajo la barrera |
|---|---:|---:|---|
| 0 | 1.277282509083e-24 | 1.927662219 | Sí |
| 1 | 3.702015207014e-24 | 5.587044995 | Sí |
| 2 | 6.038003412510e-24 | 9.112495455 | No |
| 3 | 8.620825843336e-24 | 13.010465703 | No |

Se obtiene `omega_p/(2*pi)=3.964287348 GHz` y `Delta U/h=8.076295330 GHz`.
`E/h` es una frecuencia; `omega_p` es una frecuencia angular.

El estudio con `N=1023,2047,4095` conserva ambos extremos y obtiene orden
de convergencia aproximadamente 2. Entre las dos últimas mallas, `E0/h`
aumenta `0.129178 MHz` y `E1/h` aumenta `0.567177 MHz`.

Luego se desplaza **un solo extremo** en `-0.02` o `+0.02 rad`, manteniendo el
otro fijo. Se ajusta N al entero más próximo para conservar el paso; la variación
relativa del paso es menor que `0.001 %`. Las diferencias siguientes se miden
respecto al dominio original con `N=4095`:

| Extremo | Desplazamiento [rad] | Cambio E0/h | Cambio E1/h |
|---|---:|---:|---:|
| Izquierdo | -0.02 | -0.791 Hz | -3.495 Hz |
| Izquierdo | +0.02 | +0.804 Hz | +3.517 Hz |
| Derecho | -0.02 | +1.972305 MHz | +43.871914 MHz |
| Derecho | +0.02 | -0.823780 MHz | -25.224987 MHz |

Para cada candidato, el informe verifica que el potencial del extremo movido
siga por encima de la energía inicial y de la nueva. En los dos casos de la
frontera derecha su altura es aproximadamente `7.95 GHz` o `7.94 GHz`, todavía
por encima de los dos candidatos. La altura física de la barrera se sigue
calculando en el máximo `pi-alpha`, incluso cuando se mueve la frontera.

Como N debe ser entero, los cambios brutos también incluyen un pequeño efecto
de discretización. Cada dominio se vuelve a resolver a paso mitad y se aplica
`E*=(4*E_fina-E_gruesa)/3` para cancelar el término de error cuadrático.
El informe muestra tanto la diferencia bruta como la extrapolada y su diferencia.
Tras esa corrección, los cambios por la izquierda son menores que `0.01 Hz`
(al nivel del redondeo numérico); las diferencias brutas de unos Hz provienen
principalmente del pequeño cambio de paso. Por la derecha, los cambios
extrapolados son `+1.972168, +43.880272 MHz` al desplazar `-0.02 rad`, y
`-0.823689, -25.229908 MHz` al desplazar `+0.02 rad`.

Por tanto, `n=0,1` son **candidatos a niveles localizados**, con sensibilidad
distinta a la frontera derecha. Este cálculo con Dirichlet entrega autovalores
de un intervalo cerrado; no calcula resonancias complejas ni tiempos de escape.

## Ejecución

En este equipo quedó preparado un entorno local en `.venv`. Puedes hacer
**doble clic en `ejecutar.cmd`**: muestra las tablas y abre la gráfica; al cerrar
la gráfica, la consola espera una tecla para que puedas leer el resultado.
También puedes ejecutarlo desde PowerShell, dentro de esta carpeta:

```powershell
.\ejecutar.cmd
```

Para guardar los resultados sin abrir la gráfica:

```powershell
.\ejecutar.cmd --sin-mostrar
```

El comando `python` de este equipo apunta a un acceso de WindowsApps que no
se pudo ejecutar. El lanzador usa directamente `.venv\Scripts\python.exe` y
no depende de ese comando ni de activar el entorno. Este entorno reutiliza
las bibliotecas de `.verification-deps` mediante un archivo `.pth`; conserva
ambas carpetas. Está basado en el Python ya disponible en este equipo y no
es un entorno portátil para copiar a otra computadora.

Para instalar el proyecto en otro entorno, se requiere Python 3.10 o posterior
y un comando `python` funcional:

```powershell
python -m pip install -r requirements.txt
python oscilador_armonico.py
```

Para guardar la figura sin abrir una ventana:

```powershell
python oscilador_armonico.py --sin-mostrar
```

La masa predeterminada es la del electrón (`scipy.constants.m_e`) y la frecuencia
angular es `1e15 rad/s`. Son parámetros de ejemplo, modificables:

```powershell
python oscilador_armonico.py --masa 9.1093837139e-31 --omega 1e15
```

Los cálculos usan metros, kilogramos, segundos y joules. La longitud
`ell = sqrt(hbar/(masa*omega))` fija el dominio; los cocientes `E/(hbar*omega)`
y `dx/ell` se usan solo para presentar y comparar resultados.

## Resultados

La consola y `resultados/informe.txt` muestran:

1. Energías numéricas y analíticas en J, `E_num/(hbar*omega)` y error relativo
   absoluto porcentual para `N=399`, `L=6*ell`.
2. Errores para `N=99,199,399`, con `L=6*ell` fijo, y orden observado de
   convergencia. Los pasos son `0.12*ell`, `0.06*ell` y `0.03*ell`.
3. Un estudio de frontera con `L/ell=3,4.5,6,7.5,9` y `dx=0.03*ell` fijo.
   Usa `N=199,299,399,499,599`, respectivamente. Incluye errores con signo y
   diferencias respecto al dominio mayor, que conserva el error de discretización.
4. Las normas discretas de los cuatro estados.

`resultados/funciones_de_onda.png` contiene cuatro paneles, con ejes en metros
y `m**(-1/2)`, y la solución analítica como referencia. Los extremos numéricos
son exactamente cero. El signo global se alinea con el estado analítico solo
para comparar visualmente.

Con tres puntos se espera un error dominante proporcional a `dx**2`: al reducir
el paso a la mitad, el error baja aproximadamente por un factor cuatro.
Las fronteras finitas elevan las energías y la discretización las reduce;
pueden compensarse. Por ello, una coincidencia aislada con la energía analítica
no sustituye los dos estudios de convergencia.

## Reutilizar el Hamiltoniano

```python
from oscilador_armonico import resolver_hamiltoniano, normalizar_y_agregar_fronteras

# c en J; U_interior contiene el potencial en J, evaluado en N puntos interiores.
energias, vectores = resolver_hamiltoniano(c, U_interior, numero_niveles=4)
psi = normalizar_y_agregar_fronteras(vectores, dx)
```

La función usa `diagonal=2*c+U_interior`, `vecina=-c` y devuelve los estados
como columnas. La normalización posterior impone `dx*sum(abs(psi)**2)=1`
y añade ambos ceros de frontera. Con otra coordenada, como la fase, habrá que
definir el coeficiente cinético y las condiciones de frontera del modelo;
el ensamblaje tridiagonal no depende de la forma del potencial.

Se conserva `tol=0` para que la tolerancia del diagonalizador siga la escala
del Hamiltoniano. Véase la [documentación de SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh_tridiagonal.html).

## Verificación

En el entorno preparado de este equipo:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Las pruebas verifican el espectro discreto de una caja de potencial constante,
el residuo del problema de autovalores, normalización, ortogonalidad,
convergencia cuadrática y sensibilidad a las fronteras. Las comparaciones de
energías usan escalas físicas explícitas: una tolerancia absoluta genérica
podría aceptar errores enormes en cantidades expresadas en joules.
