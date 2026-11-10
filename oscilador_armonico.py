"""Estados estacionarios 1D por diferencias finitas, con magnitudes en SI.

Ejecutar: python oscilador_armonico.py
Sin ventana gráfica: python oscilador_armonico.py --sin-mostrar

Este programa diagonaliza H; no implementa propagación Crank–Nicolson.
"""

import argparse
from dataclasses import dataclass
from math import factorial
from pathlib import Path

import numpy as np
from scipy.constants import hbar, m_e
from scipy.linalg import eigh_tridiagonal
from scipy.special import eval_hermite


def resolver_hamiltoniano(c, potencial, numero_niveles):
    """Devuelve energías [J] y autovectores columna de norma euclídea uno.

    c = hbar**2 / (2*m*dx**2), en J.
    potencial: vector real de N valores U(x_j), en J, solo puntos interiores.
    numero_niveles: entero entre 1 y N; se seleccionan los niveles más bajos.

    Se supone una malla uniforme y condiciones de Dirichlet homogéneas.
    La función no depende de la forma del potencial. La normalización física
    se hace fuera de ella porque requiere conocer dx.
    """
    potencial = np.asarray(potencial, dtype=float)
    if potencial.ndim != 1 or potencial.size == 0:
        raise ValueError("potencial debe ser un vector no vacío.")
    if not np.isfinite(c) or c <= 0 or not np.all(np.isfinite(potencial)):
        raise ValueError("c debe ser positivo y c y potencial deben ser finitos.")
    if (not isinstance(numero_niveles, (int, np.integer))
            or not 1 <= numero_niveles <= potencial.size):
        raise ValueError("numero_niveles debe ser un entero entre 1 y N.")

    diagonal = 2.0 * c + potencial
    vecina = np.full(potencial.size - 1, -c)
    return eigh_tridiagonal(
        diagonal, vecina,
        select="i", select_range=(0, numero_niveles - 1),
        # tol=0 usa una tolerancia proporcional a la escala del Hamiltoniano.
        # Una tolerancia absoluta, p. ej. 1e-12 J, sería demasiado grande.
        tol=0.0,
    )


def normalizar_y_agregar_fronteras(autovectores, dx):
    """Añade ceros en los extremos y exige dx*sum(|psi|**2)=1 por columna.

    Si dx se expresa en metros, psi tiene unidades m**(-1/2).
    """
    autovectores = np.asarray(autovectores, dtype=float)
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx debe ser positivo y finito.")
    if autovectores.ndim != 2 or not np.all(np.isfinite(autovectores)):
        raise ValueError("Los autovectores deben ser columnas de una matriz finita.")
    normas = np.sqrt(dx * np.sum(np.abs(autovectores)**2, axis=0))
    if np.any(normas == 0):
        raise ValueError("No se puede normalizar un autovector nulo.")
    interiores = autovectores / normas
    return np.pad(interiores, ((1, 1), (0, 0)), mode="constant")


@dataclass
class Solucion:
    x: np.ndarray             # N+2 puntos, incluidos -L y L, en m
    dx: float                 # paso en m
    L: float                  # semiancho en m
    ell: float                # longitud característica en m
    c: float                  # coeficiente cinético en J
    potencial: np.ndarray     # N valores interiores en J
    energias: np.ndarray      # niveles ordenados en J
    psi: np.ndarray           # columnas normalizadas, en m**(-1/2)


def resolver_oscilador(N=399, factor_L=6.0, masa=m_e, omega=1e15,
                      numero_niveles=4):
    """Resuelve el oscilador en [-factor_L*ell, factor_L*ell], en SI."""
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N debe ser un entero positivo de puntos interiores.")
    if any(not np.isfinite(v) or v <= 0 for v in (factor_L, masa, omega)):
        raise ValueError("factor_L, masa y omega deben ser positivos y finitos.")
    ell = np.sqrt(hbar / (masa * omega))
    L = factor_L * ell
    dx = 2.0 * L / (N + 1)
    x = np.linspace(-L, L, N + 2)
    potencial = 0.5 * masa * omega**2 * x[1:-1]**2
    c = hbar**2 / (2.0 * masa * dx**2)
    energias, vectores = resolver_hamiltoniano(c, potencial, numero_niveles)
    psi = normalizar_y_agregar_fronteras(vectores, dx)
    return Solucion(x, dx, L, ell, c, potencial, energias, psi)


def psi_analitica(x, n, ell):
    """Autofunción del oscilador en la recta infinita, en m**(-1/2)."""
    xi = x / ell
    prefactor = 1.0 / np.sqrt(2.0**n * factorial(n) * np.sqrt(np.pi) * ell)
    return prefactor * eval_hermite(n, xi) * np.exp(-0.5 * xi**2)


def crear_informe(mallas, dominios, masa, omega):
    """Construye las tablas de energías, discretización y frontera."""
    base = mallas[-1]
    escala_E = hbar * omega
    exactas = escala_E * (np.arange(base.energias.size) + 0.5)
    errores = [100.0 * np.abs(s.energias - exactas) / exactas for s in mallas]
    lineas = [
        "OSCILADOR ARMÓNICO 1D: DIAGONALIZACIÓN TRIDIAGONAL",
        f"m = {masa:.10e} kg; omega = {omega:.10e} rad/s",
        f"hbar = {hbar:.10e} J s; hbar*omega = {escala_E:.10e} J",
        f"ell = {base.ell:.10e} m; L = {base.L:.10e} m = 6 ell",
        "N cuenta puntos interiores; dx = 2L/(N+1).",
        "",
        "1. Cuatro niveles, N=399, L=6 ell",
        " n       E_num [J]         E_analítica [J]    E_num/(hbar*omega)   error relativo [%]",
    ]
    for n, (energia, exacta) in enumerate(zip(base.energias, exactas)):
        lineas.append(f" {n}    {energia:.10e}    {exacta:.10e}"
                      f"       {energia/escala_E:.10f}         {errores[-1][n]:.8f}")

    lineas += [
        "", "2. Discretización: L=6 ell fijo, errores relativos absolutos [%]",
        "   N       dx [m]       dx/ell       n=0          n=1          n=2          n=3",
    ]
    for s, error in zip(mallas, errores):
        lineas.append(f" {s.x.size-2:3d}   {s.dx:.6e}   {s.dx/s.ell:.5f}   "
                      + "  ".join(f"{valor:11.7f}" for valor in error))
    lineas += ["", "Orden observado p = log(error_anterior/error_actual) / log(dx_anterior/dx_actual)",
               "   cambio N         p(n=0)     p(n=1)     p(n=2)     p(n=3)"]
    for i in range(1, len(mallas)):
        orden = np.log(errores[i-1] / errores[i]) / np.log(mallas[i-1].dx / mallas[i].dx)
        lineas.append(f" {mallas[i-1].x.size-2:3d} -> {mallas[i].x.size-2:3d}      "
                      + "  ".join(f"{p:9.5f}" for p in orden))

    lineas += [
        "", "3. Frontera: dx/ell=0.03 fijo, errores relativos CON SIGNO [%]",
        "Referencia de dominio: L=9 ell con el mismo dx (no es una solución exacta).",
        " L/ell    N    dx/ell      n=0          n=1          n=2          n=3      max|E-E_ref|/(hbar*omega)",
    ]
    referencia = dominios[-1].energias
    for s in dominios:
        error = 100.0 * (s.energias - exactas) / exactas
        cambio = np.max(np.abs(s.energias - referencia)) / escala_E
        lineas.append(f" {s.L/s.ell:5.1f}   {s.x.size-2:3d}   {s.dx/s.ell:.5f}   "
                      + "  ".join(f"{valor:+11.7f}" for valor in error)
                      + f"        {cambio:.5e}")
    lineas += [
        "", "La frontera de Dirichlet eleva los niveles; el error dominante de diferencias",
        "finitas los reduce. Pueden compensarse: un error pequeño aislado no prueba convergencia.",
        "La diferencia respecto al dominio mayor estima la sensibilidad a la frontera.",
        "El error que persiste al ampliar L, a dx fijo, se debe principalmente a la discretización.",
        "", "4. Normalización: dx * sum_j |psi_n(x_j)|^2 (incluidos los ceros extremos)",
    ]
    for n, norma in enumerate(base.dx * np.sum(np.abs(base.psi)**2, axis=0)):
        lineas.append(f" n={n}: {norma:.15f}")
    return "\n".join(lineas)


def graficar_funciones(solucion, omega, ruta, mostrar=True):
    """Grafica cada estado en un panel, conservando unidades físicas en los ejes."""
    import matplotlib
    if not mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    for n, ax in enumerate(ejes.flat):
        exacta = psi_analitica(solucion.x, n, solucion.ell)
        numerica = solucion.psi[:, n].copy()
        # El signo global de un autovector es arbitrario; alinearlo facilita comparar.
        if np.dot(numerica, exacta) < 0:
            numerica *= -1
        ax.plot(solucion.x, numerica, color="#155e75", lw=2.2, label="Numérica")
        ax.plot(solucion.x, exacta, color="#c2410c", lw=1.4, ls="--", label="Analítica")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_title(rf"$n={n}$; $E_n/(\hbar\omega)={solucion.energias[n]/(hbar*omega):.6f}$")
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(rf"$\psi_{n}(x)$ [m$^{{-1/2}}$]")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
        ax.set_xlim(-solucion.L, solucion.L)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle(r"Oscilador armónico: $N=399$, $L=6\ell$, $\Delta x=0.03\ell$", fontsize=14)
    fig.savefig(ruta, dpi=180)
    if mostrar:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masa", type=float, default=m_e, help="Masa en kg (por defecto: electrón).")
    parser.add_argument("--omega", type=float, default=1e15, help="Frecuencia angular en rad/s.")
    parser.add_argument("--sin-mostrar", action="store_true", help="Guarda la figura sin abrir una ventana.")
    parser.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent / "resultados",
                        help="Carpeta para el informe y la figura.")
    args = parser.parse_args()
    if any(not np.isfinite(v) or v <= 0 for v in (args.masa, args.omega)):
        parser.error("masa y omega deben ser positivos y finitos.")

    mallas = [resolver_oscilador(N=N, factor_L=6.0, masa=args.masa, omega=args.omega)
              for N in (99, 199, 399)]
    # Elegimos múltiplos de 1.5 ell: el paso permanece igual incluso al redondear N.
    dx_objetivo = mallas[-1].dx
    ell = mallas[-1].ell
    dominios = []
    for factor_L in (3.0, 4.5, 6.0, 7.5, 9.0):
        N = int(round(2.0 * factor_L * ell / dx_objetivo)) - 1
        dominios.append(resolver_oscilador(N=N, factor_L=factor_L,
                                          masa=args.masa, omega=args.omega))

    informe = crear_informe(mallas, dominios, args.masa, args.omega)
    print(informe)
    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "informe.txt").write_text(informe + "\n", encoding="utf-8")
    ruta_figura = args.salida / "funciones_de_onda.png"
    graficar_funciones(mallas[-1], args.omega, ruta_figura, mostrar=not args.sin_mostrar)
    print(f"\nInforme y figura guardados en: {args.salida.resolve()}")


if __name__ == "__main__":
    main()
