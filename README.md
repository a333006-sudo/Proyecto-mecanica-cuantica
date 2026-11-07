# Oscilador armónico 1D en unidades físicas

Este programa valida el Hamiltoniano estacionario de diferencias finitas antes
de reutilizarlo con otro potencial. Calcula cuatro niveles con
`scipy.linalg.eigh_tridiagonal`; **no es todavía una implementación de
Crank–Nicolson**, que propaga la función de onda en el tiempo. La diagonalización
proporciona una referencia para validar posteriormente esa propagación.

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
