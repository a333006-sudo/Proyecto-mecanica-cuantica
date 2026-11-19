"""Validación del método con el potencial Josephson sin corriente (I = 0).

Con I = 0 el potencial es U(gamma) = E_J (1 - cos gamma), con mínimo en 0 y
máximos en +-pi. Para oscilaciones pequeñas, 1 - cos gamma ~ gamma^2/2, y la
unión es un oscilador armónico con

    m_gamma omega_p0^2 = E_J,    omega_p0 = sqrt(2 e I_c / (hbar C)),

así que los niveles bajos deben ser E_n = hbar omega_p0 (n + 1/2).

Se resuelve la ecuación de Schrödinger con las mismas diferencias finitas de
tres puntos que en josephson.py, en [-pi, pi] y con psi = 0 en los extremos.
Ahí la barrera mide 2 E_J ~ 880 hbar omega_p0, así que las paredes no afectan
a los niveles bajos. Se repite con tres mallas para separar el error de
discretización de la diferencia física con el oscilador: el coseno es más
abierto que la parábola (término -gamma^4/24), por lo que los niveles quedan
ligeramente por debajo de hbar omega_p0 (n + 1/2) y la diferencia crece con n.

Uso:
    python validacion_I0.py
    python validacion_I0.py --ic 9.489e-6 --capacitancia 6.35e-12

Salidas (carpeta resultados/validacion_I0/): resumen_I0.txt, tabla_I0.tex,
validacion_I0.png
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.constants import e, hbar
from scipy.linalg import eigh_tridiagonal

from josephson import GHz, estilo

N_TABLA = 4                         # niveles en la tabla (como en la sección 3.2)
N_FIGURA = 40                       # niveles en la figura
PASOS = (0.12, 0.06, 0.03)          # Delta gamma / a_0 de las tres mallas


class UnionSinCorriente:
    """Unión Josephson con I = 0: U = E_J (1 - cos gamma), medido desde el mínimo."""

    def __init__(self, Ic, C):
        if not (Ic > 0 and C > 0):
            raise ValueError("Se requiere Ic > 0 y C > 0.")
        self.Ic, self.C = Ic, C
        self.E_J = hbar * Ic / (2 * e)                 # Ec. 2.8
        self.m = C * (hbar / (2 * e))**2               # Ec. 2.16
        self.omega_p0 = np.sqrt(self.E_J / self.m)     # Ec. 2.25 con I = 0
        self.a0 = np.sqrt(hbar / (self.m * self.omega_p0))   # ancho del estado base

    def U(self, g):
        return self.E_J * (1 - np.cos(g))

    def E_armonico(self, n):
        return hbar * self.omega_p0 * (np.asarray(n) + 0.5)


def niveles(u, paso, k):
    """Primeros k niveles [J] y funciones de onda en [-pi, pi] con psi = 0 en +-pi.
    paso = Delta gamma / a_0."""
    N = int(np.ceil(2 * np.pi / (paso * u.a0))) - 1     # puntos interiores
    g = np.linspace(-np.pi, np.pi, N + 2)[1:-1]
    dg = g[1] - g[0]
    c = hbar**2 / (2 * u.m * dg**2)                     # mismo C que la Ec. 3.6
    E, psi = eigh_tridiagonal(2 * c + u.U(g), np.full(N - 1, -c),
                              select="i", select_range=(0, k - 1))
    return g, dg, E, psi / np.sqrt(dg)


def resumen(u, mallas):
    hw = hbar * u.omega_p0
    g, dg, E, _ = mallas[-1]
    exactas = u.E_armonico(np.arange(N_TABLA))
    L = [
        "VALIDACIÓN CON EL POTENCIAL JOSEPHSON SIN CORRIENTE (I = 0)",
        f"I_c = {u.Ic*1e6:.4f} uA   C = {u.C*1e12:.4f} pF",
        f"E_J/h            = {u.E_J/GHz:.1f} GHz",
        f"omega_p0/2pi     = {u.omega_p0/(2*np.pi)/1e9:.4f} GHz",
        f"2 E_J/(hbar w)   = {2*u.E_J/hw:.0f}   (barrera en +-pi, en cuantos)",
        f"a_0              = {u.a0:.5f} rad",
        "",
        f"Malla más fina: Delta gamma = {PASOS[-1]} a_0 ({len(g)} puntos interiores)",
        " n   E_num/h [GHz]   hbar w_p0 (n+1/2)/h [GHz]   diferencia relativa [%]",
    ]
    for n in range(N_TABLA):
        L.append(f" {n}   {E[n]/GHz:12.5f}      {exactas[n]/GHz:12.5f}"
                 f"               {100*(E[n]-exactas[n])/exactas[n]:+.4f}")
    L += ["", "Convergencia con la malla, (E_n - E_n^arm)/(hbar w_p0):",
          " Dg/a0   " + "".join(f"     n={n}    " for n in range(N_TABLA))]
    for p, (_, _, Em, _) in zip(PASOS, mallas):
        L.append(f" {p:5.2f}  " + "".join(f"  {(Em[n]-exactas[n])/hw:+.3e}"
                                          for n in range(N_TABLA)))
    d = [(m[2][:N_TABLA] - exactas) for m in mallas]
    orden = np.log2(np.abs((d[0] - d[1]) / (d[1] - d[2])))
    ext = extrapolar(mallas)[:N_TABLA] - exactas
    L += [" ->0   " + "".join(f"  {v/hw:+.3e}" for v in ext)
          + "   (extrapolación de Richardson)",
          "",
          "Orden de convergencia observado (mallas sucesivas, factor 2 en el paso): "
          + ", ".join(f"{o:.2f}" for o in orden),
          "",
          "El error de discretización baja como Delta gamma^2. Lo que queda al extrapolar a",
          "Delta gamma -> 0 es físico: el coseno es más abierto que la parábola, así que los",
          "niveles quedan por debajo del armónico y la diferencia crece con n.",
          f"Para n = 0 el corrimiento es {ext[0]/hw:+.1e} hbar w_p0; el valor esperado del término",
          f"-E_J gamma^4/24 en el estado base es -(hbar w_p0)^2/(32 E_J) = "
          f"{-hw/(32*u.E_J):+.1e} hbar w_p0."]
    return "\n".join(L)


def extrapolar(mallas):
    """Extrapolación de Richardson con error O(Delta gamma^2) y paso a la mitad:
    E(0) ~ E(h/2) + [E(h/2) - E(h)] / 3."""
    return mallas[-1][2] + (mallas[-1][2] - mallas[-2][2]) / 3


def tabla_latex(u, mallas):
    E = mallas[-1][2]
    exactas = u.E_armonico(np.arange(N_TABLA))
    L = [r"\begin{table}[H]", r"    \centering",
         r"    \caption{Niveles del potencial Josephson con $I=0$ ($\omega_{p0}/2\pi="
         + f"{u.omega_p0/(2*np.pi)/1e9:.3f}" + r"\ \mathrm{GHz}$) contra el oscilador armónico.}",
         r"    \label{tab:validacionI0}", r"    \begin{tabular}{cccc}", r"        \hline",
         r"        $n$ & Numérico (GHz) & $\hbar\omega_{p0}(n+\frac12)/h$ (GHz) & Diferencia (\%) \\",
         r"        \hline"]
    for n in range(N_TABLA):
        L.append(f"        {n} & {E[n]/GHz:.4f} & {exactas[n]/GHz:.4f} & "
                 f"${100*(E[n]-exactas[n])/exactas[n]:+.3f}$ \\\\")
    L += [r"        \hline", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(L)


def figura(u, mallas, ruta):
    plt = estilo()
    hw = hbar * u.omega_p0
    g, _, E, psi = mallas[-1]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")

    # (a) potencial, parábola y primeros niveles
    x = np.linspace(-8 * u.a0, 8 * u.a0, 600)
    ax.plot(x, u.U(x) / GHz, color="#1d4ed8", lw=2.2, label=r"$E_J(1-\cos\gamma)$")
    ax.plot(x, 0.5 * u.E_J * x**2 / GHz, color="#ea580c", lw=1.8, ls="--",
            label=r"$\frac{1}{2}m_\gamma\omega_{p0}^2\gamma^2$")
    dentro = np.abs(g) < x[-1]
    escala = 0.45 * hw / GHz / np.max(psi[:, 0]**2)
    for n in range(6):
        ax.axhline(E[n] / GHz, color="gray", lw=0.8, ls=":")
        ax.plot(g[dentro], E[n] / GHz + escala * psi[dentro, n]**2, color="#176b87", lw=1.3)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 8.2 * hw / GHz)
    ax.set_xlabel(r"Diferencia de fase $\gamma$ [rad]")
    ax.set_ylabel(r"Energía$/h$ [GHz]")
    ax.set_title(rf"(a) Pozo con $I=0$: primeros niveles ($\omega_{{p0}}/2\pi$ = "
                 rf"{u.omega_p0/(2*np.pi)/1e9:.3f} GHz)", fontsize=12)
    ax.legend(loc="upper right")

    # (b) diferencia con el oscilador contra n, para las tres mallas
    n = np.arange(N_FIGURA)
    marcas = ["o", "s", "^"]
    for (p, (_, _, Em, _)), mk in zip(zip(PASOS, mallas), marcas):
        bx.plot(n, (Em - u.E_armonico(n)) / hw, mk, ms=4, ls="-", lw=0.8,
                label=rf"$\Delta\gamma = {p}\,a_0$")
    bx.plot(n, (extrapolar(mallas) - u.E_armonico(n)) / hw, color="black", lw=2,
            label=r"Extrapolado a $\Delta\gamma\to 0$")
    bx.axhline(0, color="gray", lw=1)
    bx.set_xlabel(r"$n$")
    bx.set_ylabel(r"$[E_n-\hbar\omega_{p0}(n+\frac{1}{2})]\,/\,\hbar\omega_{p0}$")
    bx.set_title("(b) Diferencia con el oscilador armónico", fontsize=12)
    bx.legend(loc="lower left")
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ic", type=float, default=9.489e-6, help="Corriente crítica [A]")
    ap.add_argument("--capacitancia", type=float, default=6.35e-12, help="Capacitancia [F]")
    ap.add_argument("--salida", type=Path,
                    default=Path(__file__).resolve().parent / "resultados" / "validacion_I0")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    a.salida.mkdir(parents=True, exist_ok=True)

    u = UnionSinCorriente(a.ic, a.capacitancia)
    mallas = [niveles(u, p, N_FIGURA) for p in PASOS]
    texto = resumen(u, mallas)
    print(texto)
    (a.salida / "resumen_I0.txt").write_text(texto + "\n", encoding="utf-8")
    (a.salida / "tabla_I0.tex").write_text(tabla_latex(u, mallas) + "\n", encoding="utf-8")
    figura(u, mallas, a.salida / "validacion_I0.png")
    print(f"\nResultados guardados en: {a.salida.resolve()}")


if __name__ == "__main__":
    main()
