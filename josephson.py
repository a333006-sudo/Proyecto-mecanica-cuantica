"""Niveles cuasiligados de una unión Josephson polarizada por corriente.

Resuelve  -hbar^2/(2 m) psi'' + U(gamma) psi = E psi  por diferencias finitas,
con  U(gamma) = -E_J [cos(gamma) + s gamma],  s = I/I_c,  m = C (hbar/2e)^2.

Corrección respecto a la versión anterior: la pared derecha NO se coloca en
la cima de la barrera. Se coloca más allá, en gamma_max + Delta_b, y se
repite el cálculo para varios Delta_b. Un nivel del pozo es un autovalor
con E < Delta U cuya probabilidad dentro del pozo es mayor que P_MIN; los
demás autovalores son estados "de caja" del lado derecho y se descartan.
La energía reportada es la mediana sobre los Delta_b y la incertidumbre es
el intervalo (máx - mín).

Uso:
    python josephson.py                  # parámetros por defecto
    python josephson.py --ic 9.489e-6 --capacitancia 6.35e-12 --corriente 9.40e-6

Salidas (carpeta resultados/josephson/): resumen.txt, niveles_vs_corriente.csv,
comparacion_cubico.csv, pozo_y_niveles.png, estabilizacion_frontera.png,
niveles_vs_corriente.png, comparacion_cubico.png
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.constants import e, h, hbar, k as k_B
from scipy.linalg import eigh_tridiagonal

# ---------------------------------------------------------------------------
# Parámetros numéricos (todos explicados en el reporte)
# ---------------------------------------------------------------------------
PUNTOS_POZO = 200                             # puntos entre gamma_min y gamma_max
FRACCIONES_PARED = np.linspace(0.4, 2.0, 9)   # Delta_b / (gamma_max - gamma_min)
P_MIN = 0.9                                   # probabilidad mínima dentro del pozo


# ---------------------------------------------------------------------------
# Física del circuito
# ---------------------------------------------------------------------------
class Union:
    """Unión Josephson con corriente crítica Ic [A], capacitancia C [F] y s = I/Ic."""

    def __init__(self, Ic, C, s):
        if not (Ic > 0 and C > 0 and 0 < s < 1):
            raise ValueError("Se requiere Ic > 0, C > 0 y 0 < I/Ic < 1.")
        self.Ic, self.C, self.s = Ic, C, s
        self.E_J = hbar * Ic / (2 * e)              # J      (Ec. 2.8)
        self.m = C * (hbar / (2 * e))**2            # J s²   (Ec. 2.16)
        self.g_min = np.arcsin(s)                   # rad    (Ec. 2.19)
        self.g_max = np.pi - np.arcsin(s)           # rad
        self.omega_p = np.sqrt(self.E_J * np.sqrt(1 - s**2) / self.m)   # (Ec. 2.25)
        self.dU = self.U(self.g_max)                # J      (Ec. 2.20)
        self.g_izq = -np.pi - self.g_min            # pared izquierda (máximo izquierdo)

    def U(self, g):
        """Potencial medido desde el mínimo, en J."""
        g = np.asarray(g, dtype=float)
        U = lambda x: -self.E_J * (np.cos(x) + self.s * x)
        return U(g) - U(self.g_min)

    # --- Aproximaciones analíticas para comparar -------------------------
    def E_armonico(self, n):
        return hbar * self.omega_p * (n + 0.5)

    def E_perturbativo(self, n):
        """Oscilador + término cúbico a 2.º orden (Griffiths cap. 7):
        E_n = hbar w (n+1/2) - (hbar w)^2 (30 n^2 + 30 n + 11) / (432 Delta U)."""
        hw = hbar * self.omega_p
        return hw * (n + 0.5) - hw**2 * (30 * n**2 + 30 * n + 11) / (432 * self.dU)

    def exponente_wkb(self):
        """Fórmula cerrada del pozo cúbico, B = 36 Delta U / (5 hbar omega_p),
        evaluada con la barrera EXACTA Delta U (Ec. 2.20)."""
        return 36 * self.dU / (5 * hbar * self.omega_p)

    def g_salida(self):
        """Punto de salida del túnel con E = 0: U(gamma) = 0 a la derecha de la barrera."""
        from scipy.optimize import brentq
        return brentq(self.U, self.g_max, self.g_max + np.pi)

    def T_cruce(self):
        """Temperatura de cruce T* = hbar omega_p / (2 pi k_B), en K."""
        return hbar * self.omega_p / (2 * np.pi * k_B)


class PozoCubico:
    """Aproximación cúbica del potencial alrededor del mínimo (Ec. 4.10):
        V(gamma) = (1/2) m omega_p^2 x^2 (1 - x/l),   x = gamma - gamma_min,
        l = 3 sqrt(1 - s^2) / s                       (Ec. 4.9)
    Tiene la misma masa, el mismo mínimo y la misma omega_p que la unión, por lo
    que se resuelve con las mismas funciones que el potencial completo."""

    def __init__(self, u):
        self.u, self.s, self.m, self.omega_p = u, u.s, u.m, u.omega_p
        self.l = 3 * np.sqrt(1 - u.s**2) / u.s
        self.g_min = u.g_min
        self.g_max = u.g_min + 2 * self.l / 3                       # Ec. 4.13
        self.dU = 2 * self.m * self.omega_p**2 * self.l**2 / 27     # Ec. 4.16
        self.g_izq = u.g_min - 3 * self.l      # ahí V ~ 240 Delta U: pared efectiva

    def U(self, g):
        x = np.asarray(g, dtype=float) - self.g_min
        return 0.5 * self.m * self.omega_p**2 * x**2 * (1 - x / self.l)

    def g_salida(self):
        return self.g_min + self.l               # V = 0 a la derecha de la barrera

    def exponente_wkb(self):
        return 36 * self.dU / (5 * hbar * self.omega_p)


def exponente_wkb_numerico(pozo):
    """B = (2/hbar) * integral de sqrt(2 m U(gamma)) entre gamma_min y el punto de
    salida, con E = 0 (fondo del pozo). Es el exponente de la tasa, Gamma ~ e^-B."""
    from scipy.integrate import quad
    f = lambda g: np.sqrt(max(2 * pozo.m * float(pozo.U(g)), 0.0))
    return 2 / hbar * quad(f, pozo.g_min, pozo.g_salida(), limit=200)[0]


def comparar_cubico(Ic, C, s_valores):
    """Barrera y exponente WKB de ambos potenciales para cada I/Ic."""
    filas = []
    for s in s_valores:
        u = Union(Ic, C, s)
        c = PozoCubico(u)
        filas.append((s, u.dU / (hbar * u.omega_p), c.dU / u.dU,
                      exponente_wkb_numerico(u), exponente_wkb_numerico(c),
                      u.exponente_wkb()))
    return np.array(filas)


# ---------------------------------------------------------------------------
# Diferencias finitas
# ---------------------------------------------------------------------------
def diagonalizar(u, g_der, dg=None):
    """Autoestados con E < Delta U en [g_izq, g_der], psi = 0 en los extremos.

    Sirve para cualquier pozo con atributos U, m, g_izq, g_min, g_max y dU.
    Para la unión, la pared izquierda está en el máximo izquierdo,
    -pi - arcsin(s), donde la barrera mide ~10^4 GHz (en la práctica infinita).
    Devuelve gamma (puntos interiores), energías [J], psi (columnas
    normalizadas: dg * sum |psi|^2 = 1) y P_pozo de cada estado.
    """
    g_izq = u.g_izq
    if dg is None:
        dg = (u.g_max - u.g_min) / PUNTOS_POZO
    N = int(round((g_der - g_izq) / dg)) - 1      # puntos interiores
    g = np.linspace(g_izq, g_der, N + 2)[1:-1]
    dg = g[1] - g[0]

    c = hbar**2 / (2 * u.m * dg**2)               # H_jj = 2c + U_j, H_j,j±1 = -c
    E, psi = eigh_tridiagonal(2 * c + u.U(g), np.full(N - 1, -c),
                              select="v", select_range=(-np.inf, u.dU))
    psi /= np.sqrt(dg)                            # normalización continua
    P_pozo = dg * np.sum(psi[g < u.g_max]**2, axis=0)
    return g, E, psi, P_pozo


def niveles_pozo(u, dg=None):
    """Niveles del pozo por estabilización respecto a la pared derecha.

    Devuelve un diccionario con:
      E      energías (mediana sobre las paredes), J
      dE     intervalo máx - mín sobre las paredes, J
      barrido  lista de (Delta_b, E, P_pozo) para graficar la estabilización
    """
    ancho = u.g_max - u.g_min
    aceptados, barrido = [], []
    for f in FRACCIONES_PARED:
        _, E, _, P = diagonalizar(u, u.g_max + f * ancho, dg)
        barrido.append((f * ancho, E, P))
        aceptados.append(E[P > P_MIN])

    # Cerca de un cruce evitado con un estado de caja, un nivel del pozo puede
    # quedar con P_pozo < P_MIN para alguna pared. Por eso no se emparejan los
    # niveles por su índice, sino por cercanía a una lista de referencia (la
    # pared con más niveles aceptados), con tolerancia de un cuarto de cuanto.
    referencia = max(aceptados, key=len)
    tol = 0.25 * hbar * u.omega_p
    E, dE, paredes = [], [], []
    for Er in referencia:
        valores = [a[np.argmin(np.abs(a - Er))] for a in aceptados
                   if len(a) and np.min(np.abs(a - Er)) < tol]
        E.append(np.median(valores))
        dE.append(np.max(valores) - np.min(valores))
        paredes.append(len(valores))
    return {"E": np.array(E), "dE": np.array(dE), "paredes": np.array(paredes),
            "barrido": barrido}


def estado_representativo(u, dg=None):
    """Funciones de onda del pozo con la pared intermedia, para graficar."""
    f = FRACCIONES_PARED[len(FRACCIONES_PARED) // 2]
    g, E, psi, P = diagonalizar(u, u.g_max + f * (u.g_max - u.g_min), dg)
    return g, E[P > P_MIN], psi[:, P > P_MIN]


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------
GHz = h * 1e9   # divide una energía en J entre GHz para obtener E/h en GHz


def resumen(u, res, res_fino):
    """Texto corto con todo lo que va al reporte."""
    E, dE = res["E"], res["dE"]
    n = np.arange(len(E))
    k = min(len(E), len(res_fino["E"]))
    malla = np.abs(res_fino["E"][:k] - E[:k])
    L = [
        "UNIÓN JOSEPHSON: NIVELES DEL POZO",
        f"I_c = {u.Ic*1e6:.4f} uA   C = {u.C*1e12:.4f} pF   "
        f"I = {u.s*u.Ic*1e6:.4f} uA   I/I_c = {u.s:.5f}",
        "",
        f"E_J            = {u.E_J:.4e} J   (E_J/h = {u.E_J/GHz:.1f} GHz)",
        f"m_gamma        = {u.m:.4e} J s^2",
        f"omega_p/2pi    = {u.omega_p/(2*np.pi)/1e9:.4f} GHz",
        f"Delta U / h    = {u.dU/GHz:.4f} GHz",
        f"Delta U / hbar omega_p = {u.dU/(hbar*u.omega_p):.3f}",
        f"Exponente WKB B = 36 DU/(5 hbar w_p) = {u.exponente_wkb():.2f}",
        f"Temperatura de cruce T* = {u.T_cruce()*1e3:.1f} mK",
        "",
        f"Niveles dentro del pozo: {len(E)}",
        " n   E_num/h [GHz]  +- (pared) [MHz]  paredes  error malla [MHz]"
        "   armónico [GHz]  perturbativo [GHz]",
    ]
    for i in n:
        em = f"{malla[i]/(h*1e6):9.3f}" if i < len(malla) else "      ---"
        L.append(f" {i}   {E[i]/GHz:10.4f}     {dE[i]/(h*1e6):10.2f}      {res['paredes'][i]}/{len(FRACCIONES_PARED)}   {em}"
                 f"          {u.E_armonico(i)/GHz:8.4f}        {u.E_perturbativo(i)/GHz:8.4f}")
    if len(E) >= 2:
        w01 = (E[1] - E[0]) / GHz
        L += ["",
              f"Transición 0->1: numérica {w01:.4f} GHz | armónica "
              f"{u.omega_p/(2*np.pi)/1e9:.4f} GHz | perturbativa "
              f"{(u.E_perturbativo(1)-u.E_perturbativo(0))/GHz:.4f} GHz",
              f"Anarmonicidad: (E1-E0)/(hbar omega_p) - 1 = "
              f"{(E[1]-E[0])/(hbar*u.omega_p) - 1:+.2%}"]
    L += ["",
          f"Malla: {PUNTOS_POZO} puntos entre gamma_min y gamma_max; error de malla = |E(dg) - E(dg/2)|.",
          f"Pared derecha en gamma_max + Delta_b, Delta_b/(gamma_max-gamma_min) = "
          f"{FRACCIONES_PARED[0]:.1f} ... {FRACCIONES_PARED[-1]:.1f}; nivel aceptado si P_pozo > {P_MIN}.",
          "Valor reportado = mediana; '+-' = intervalo máx-mín sobre las paredes."]
    return "\n".join(L)


def estilo():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "legend.fontsize": 11,
                         "axes.grid": True, "grid.alpha": 0.25})
    return plt


COLORES = ["#176b87", "#b45309", "#7e22ce", "#15803d", "#be123c", "#334155"]


def figura_pozo(u, res, ruta):
    """Potencial y niveles. Cada |psi_n|^2 se dibuja desplazada verticalmente a la
    altura de su energía E_n (línea discontinua), con escala vertical arbitraria:
    la energía de cada nivel es constante; lo que varía con gamma es |psi_n|^2."""
    plt = estilo()
    g, E, psi = estado_representativo(u)
    ancho = u.g_max - u.g_min
    x = np.linspace(u.g_min - 1.2 * ancho, u.g_max + 0.6 * ancho, 800)
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    ax.plot(x, u.U(x) / GHz, color="#334155", lw=2.2, label=r"$U(\gamma)-U_{min}$")
    ax.axhline(u.dU / GHz, color="#be123c", ls=":", lw=1.2,
               label=rf"$\Delta U/h$ = {u.dU/GHz:.3f} GHz")
    escala = 0.35 * u.dU / GHz / np.max(psi**2)
    dentro = (g > x[0]) & (g < x[-1])
    for n, En in enumerate(res["E"]):
        c = COLORES[n % len(COLORES)]
        ax.plot(g[dentro], En / GHz + escala * psi[dentro, n]**2, color=c, lw=1.6)
        ax.axhline(En / GHz, color=c, ls="--", lw=1)
        ax.text(x[0] + 0.02 * (x[-1] - x[0]), En / GHz + 0.12,
                rf"$n={n}$: {En/GHz:.3f} GHz", color=c, fontsize=11)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.1 * u.dU / GHz, 1.6 * u.dU / GHz)
    ax.set_xlabel(r"Diferencia de fase $\gamma$ [rad]")
    ax.set_ylabel(r"Energía$/h$ [GHz]")
    ax.set_title(rf"$I_c$ = {u.Ic*1e6:.3f} μA, $C$ = {u.C*1e12:.2f} pF, "
                 rf"$I/I_c$ = {u.s:.5f}", fontsize=12)
    ax.legend(loc="upper right")
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def figura_estabilizacion(u, res, ruta):
    """Autovalores contra la posición de la pared: los del pozo no se mueven."""
    plt = estilo()
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for db, E, P in res["barrido"]:
        pozo = P > P_MIN
        ax.scatter(np.full((~pozo).sum(), db), E[~pozo] / GHz, s=18,
                   facecolors="none", edgecolors="#94a3b8")
        ax.scatter(np.full(pozo.sum(), db), E[pozo] / GHz, s=30, color="#176b87")
    ax.scatter([], [], s=30, color="#176b87", label=rf"Estados del pozo ($P_{{pozo}}>{P_MIN}$)")
    ax.scatter([], [], s=18, facecolors="none", edgecolors="#94a3b8",
               label="Estados de caja (fuera del pozo)")
    ax.axhline(u.dU / GHz, color="#be123c", ls=":", lw=1.2, label=r"$\Delta U/h$")
    ax.set_ylim(0, 1.05 * u.dU / GHz)
    ax.set_xlabel(r"Posición de la pared, $\Delta_b = \gamma_b-\gamma_{max}$ [rad]")
    ax.set_ylabel(r"Autovalor $E/h$ [GHz]")
    ax.set_title(rf"Estabilización respecto a la frontera ($I/I_c$ = {u.s:.5f})", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def barrido_corriente(Ic, C, s_valores):
    """Niveles del pozo para cada I/Ic. Devuelve filas para el CSV."""
    filas = []
    for s in s_valores:
        u = Union(Ic, C, s)
        res = niveles_pozo(u)
        for n, (En, dEn) in enumerate(zip(res["E"], res["dE"])):
            filas.append((s, n, En / GHz, dEn / GHz, u.E_armonico(n) / GHz,
                          u.E_perturbativo(n) / GHz, u.dU / GHz,
                          u.omega_p / (2 * np.pi) / 1e9))
    return np.array(filas)


def figura_vs_corriente(datos, ruta):
    plt = estilo()
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    s_unicos = np.unique(datos[:, 0])
    dU = np.array([datos[datos[:, 0] == s][0, 6] for s in s_unicos])
    ax.plot(s_unicos, dU, color="#be123c", lw=2, label=r"$\Delta U/h$")
    for n in range(int(datos[:, 1].max()) + 1):
        d = datos[datos[:, 1] == n]
        ax.errorbar(d[:, 0], d[:, 2], yerr=d[:, 3] / 2, fmt="o", ms=4,
                    color="#176b87", label="Numérico (diferencias finitas)" if n == 0 else None)
        ax.plot(d[:, 0], d[:, 4], color="gray", ls="--", lw=1,
                label=r"Armónico $\hbar\omega_p(n+1/2)/h$" if n == 0 else None)
        ax.plot(d[:, 0], d[:, 5], color="#b45309", ls=":", lw=1.5,
                label="Perturbativo (2.º orden, cúbico)" if n == 0 else None)
    ax.set_xlabel(r"$I/I_c$")
    ax.set_ylabel(r"Energía desde el mínimo$/h$ [GHz]")
    ax.set_title("Niveles del pozo contra la corriente de polarización", fontsize=12)
    ax.legend(ncol=2, fontsize=10)
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def resumen_cubico(u, res, res_cub):
    """Comparación potencial completo vs. cúbico en el punto de operación."""
    c = PozoCubico(u)
    B, Bc = exponente_wkb_numerico(u), exponente_wkb_numerico(c)
    L = ["", "COMPARACIÓN CON EL POTENCIAL CÚBICO (Ec. 4.10)",
         f"omega_p igual por construcción: {u.omega_p/(2*np.pi)/1e9:.4f} GHz",
         f"Delta U/h:   completo {u.dU/GHz:.4f} GHz | cúbico {c.dU/GHz:.4f} GHz"
         f" | cociente {c.dU/u.dU:.4f}",
         f"gamma_max:   completo {u.g_max:.4f} rad | cúbico {c.g_max:.4f} rad",
         " n   E/h completo [GHz]   E/h cúbico [GHz]   diferencia [MHz]"]
    for n in range(min(len(res["E"]), len(res_cub["E"]))):
        L.append(f" {n}      {res['E'][n]/GHz:10.4f}         {res_cub['E'][n]/GHz:10.4f}"
                 f"        {(res_cub['E'][n]-res['E'][n])/(h*1e6):+8.2f}")
    L += ["Exponente WKB con E = 0, B = (2/hbar) int sqrt(2 m U) dgamma:",
          f"  integral numérica, potencial completo     B = {B:.3f}",
          f"  integral numérica, potencial cúbico       B = {Bc:.3f}"
          f"  (fórmula 36 DU_cub/(5 hbar w_p) = {c.exponente_wkb():.3f})",
          f"  fórmula cúbica con la barrera EXACTA     B = {u.exponente_wkb():.3f}",
          f"  error en la tasa si se usa el cúbico: e^(B - B_cub) = {np.exp(B - Bc):.3f}"]
    return "\n".join(L)


def figura_cubico(u, datos, ruta):
    """Izquierda: ambos potenciales. Derecha: cocientes contra I/Ic."""
    plt = estilo()
    c = PozoCubico(u)
    ancho = u.g_max - u.g_min
    x = np.linspace(u.g_min - 1.2 * ancho, u.g_max + 0.8 * ancho, 800)
    fig, (ax, ad) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    ax.plot(x, u.U(x) / GHz, color="#334155", lw=2.2, label="Josephson completo")
    ax.plot(x, c.U(x) / GHz, color="#b45309", lw=2, ls="--", label="Aproximación cúbica")
    ax.scatter([u.g_max, c.g_max], [u.dU / GHz, c.dU / GHz],
               color=["#334155", "#b45309"], zorder=5)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.6 * u.dU / GHz, 1.8 * u.dU / GHz)
    ax.set_xlabel(r"Diferencia de fase $\gamma$ [rad]")
    ax.set_ylabel(r"$[U(\gamma)-U_{min}]/h$ [GHz]")
    ax.set_title(rf"Potenciales ($I/I_c$ = {u.s:.5f})", fontsize=12)
    ax.legend(loc="upper right")

    s, dUc, B, Bc = datos[:, 0], datos[:, 2], datos[:, 3], datos[:, 4]
    ad.plot(s, dUc, color="#176b87", lw=2, label=r"$\Delta U_{cub}/\Delta U$")
    ad.plot(s, Bc / B, color="#be123c", lw=2, ls="--",
            label=r"$B_{cub}/B$ (exponente WKB)")
    ad.plot(s, datos[:, 5] / B, color="#15803d", lw=2, ls="-.",
            label=r"$\frac{36}{5}\frac{\Delta U}{\hbar\omega_p}\,/\,B$ ($\Delta U$ exacta)")
    ad.axhline(1, color="gray", lw=1)
    ad.axvline(u.s, color="gray", ls=":", lw=1, label="Punto de operación")
    ad.set_xlabel(r"$I/I_c$")
    ad.set_ylabel("Cociente cúbico / completo")
    ad.set_title("Calidad de la aproximación cúbica", fontsize=12)
    ad.legend(loc="upper right")
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ic", type=float, default=9.489e-6, help="Corriente crítica [A]")
    ap.add_argument("--capacitancia", type=float, default=6.35e-12, help="Capacitancia [F]")
    ap.add_argument("--corriente", type=float, default=9.40e-6, help="Corriente de polarización [A]")
    ap.add_argument("--s-min", type=float, default=0.980, help="I/Ic mínimo del barrido")
    ap.add_argument("--s-max", type=float, default=0.995, help="I/Ic máximo del barrido")
    ap.add_argument("--puntos-barrido", type=int, default=31)
    ap.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent / "resultados" / "josephson")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")          # solo se guardan las figuras, no se abren ventanas
    a.salida.mkdir(parents=True, exist_ok=True)

    # 1. Punto de operación
    u = Union(a.ic, a.capacitancia, a.corriente / a.ic)
    res = niveles_pozo(u)
    dg = (u.g_max - u.g_min) / PUNTOS_POZO
    res_fino = niveles_pozo(u, dg / 2)
    res_cub = niveles_pozo(PozoCubico(u))
    texto = resumen(u, res, res_fino) + "\n" + resumen_cubico(u, res, res_cub)
    print(texto)
    (a.salida / "resumen.txt").write_text(texto + "\n", encoding="utf-8")
    figura_pozo(u, res, a.salida / "pozo_y_niveles.png")
    figura_estabilizacion(u, res, a.salida / "estabilizacion_frontera.png")

    # 2. Barrido en I/Ic
    s_valores = np.linspace(a.s_min, a.s_max, a.puntos_barrido)
    datos = barrido_corriente(a.ic, a.capacitancia, s_valores)
    np.savetxt(a.salida / "niveles_vs_corriente.csv", datos, delimiter=",", fmt="%.8g",
               header="I/Ic,n,E_n/h [GHz],intervalo pared [GHz],armonico [GHz],"
                      "perturbativo [GHz],DeltaU/h [GHz],omega_p/2pi [GHz]", comments="")
    figura_vs_corriente(datos, a.salida / "niveles_vs_corriente.png")

    # 3. Potencial completo contra cúbico (solo integrales, es rápido)
    comp = comparar_cubico(a.ic, a.capacitancia, np.linspace(0.90, 0.999, 100))
    np.savetxt(a.salida / "comparacion_cubico.csv", comp, delimiter=",", fmt="%.8g",
               header="I/Ic,DeltaU/(hbar omega_p),DeltaU_cub/DeltaU,B completo,"
                      "B cubico,B formula con DeltaU exacta", comments="")
    figura_cubico(u, comp, a.salida / "comparacion_cubico.png")

    print(f"\nResultados guardados en: {a.salida.resolve()}")


if __name__ == "__main__":
    main()
