"""Tasa de escape de la unión Josephson: túnel cuántico (WKB) y activación térmica.

Usa la misma unión que josephson.py (clase Union) y calcula:
  - exponente WKB  B/hbar = 7.2 Delta U/(hbar omega_p)            (Ec. 4.25)
  - prefactor      a_q = [120 pi (7.2 Delta U/hbar omega_p)]^(1/2)  (Ec. 4.72)
  - tasa cuántica  Gamma_q = a_q (omega_p/2pi) exp(-B/hbar)         (Ec. 4.71)
  - tasa térmica   Gamma_t = a_t (omega_p/2pi) exp(-Delta U/k_B T), con a_t = 1
  - temperatura de escape, Gamma = (omega_p/2pi) exp(-Delta U/k_B T_esc)
  - temperatura de cruce T* = hbar omega_p/(2 pi k_B) (la que marca Devoret et al.,
    30 mK) y, como referencia, hbar omega_p/(7.2 k_B) (igualando exponentes)
  - opcional (--Q): disipación débil, exponente multiplicado por (1 + 0.87/Q)
    (Devoret et al., Ec. 2; Caldeira y Leggett). Devoret et al. reportan Q = 30 +- 15.

Barrera: por defecto se usa la barrera EXACTA Delta U (Ec. 2.20) dentro de la
fórmula cúbica. Con ella, 7.2 Delta U/(hbar omega_p) coincide con la integral
WKB numérica del potencial completo (ver resultados/josephson/resumen.txt).
Con --barrera cubica se usa Delta U_cub (Ec. 4.16), como en la derivación.

Uso:
    python escape.py
    python escape.py --ic 9.489e-6 --capacitancia 6.35e-12 --corriente 9.40e-6
    python escape.py --Q 30          # incluye la corrección por disipación

Salidas (carpeta resultados/escape/): resumen_escape.txt, tabla_parametros.tex,
escape_vs_T.csv, escape_vs_T.png
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.constants import h, hbar, k as k_B

from josephson import Union, PozoCubico, exponente_wkb_numerico, estilo, COLORES, GHz


def tasas(u, dU, T, Q=None):
    """Gamma_q, Gamma_t(T) y Gamma total (1/s) para una barrera dU [J].
    Si se da Q, el exponente se multiplica por (1 + 0.87/Q); a_q no cambia."""
    hw = hbar * u.omega_p
    nu = u.omega_p / (2 * np.pi)                    # frecuencia de intentos
    b = 7.2 * dU / hw                               # B/hbar sin disipación
    a_q = np.sqrt(120 * np.pi * b)
    G_q = a_q * nu * np.exp(-b * (1 + (0.87 / Q if Q else 0.0)))
    with np.errstate(over="ignore", divide="ignore"):
        G_t = nu * np.exp(-dU / (k_B * np.asarray(T, dtype=float)))   # a_t = 1
    return b, a_q, G_q, G_t, G_t + G_q


def T_escape(u, dU, Gamma):
    """Temperatura de escape definida por Gamma = (omega_p/2pi) exp(-dU/k_B T_esc)."""
    return dU / (k_B * np.log(u.omega_p / (2 * np.pi * Gamma)))


def resumen(u, dU, etiqueta, T, Q=None):
    b, a_q, G_q, _, _ = tasas(u, dU, T)
    G_qQ = tasas(u, dU, T, Q)[2] if Q else None
    hw = hbar * u.omega_p
    Tq = T_escape(u, dU, G_q)
    lam = 27 / 2 * dU / hw
    c = PozoCubico(u)
    L = [
        "ESCAPE DE LA UNIÓN JOSEPHSON",
        f"I_c = {u.Ic*1e6:.4f} uA   C = {u.C*1e12:.4f} pF   I/I_c = {u.s:.5f}",
        f"Barrera usada en las fórmulas: {etiqueta}",
        "",
        f"E_J/h                    = {u.E_J/GHz:.1f} GHz",
        f"m_gamma                  = {u.m:.4e} J s^2",
        f"omega_p/2pi              = {u.omega_p/(2*np.pi)/1e9:.4f} GHz",
        f"Delta U/h  (exacta)      = {u.dU/GHz:.4f} GHz",
        f"Delta U_cub/h (cúbica)   = {c.dU/GHz:.4f} GHz   (cociente {c.dU/u.dU:.4f})",
        f"Delta U/(hbar omega_p)   = {dU/hw:.4f}",
        f"lambda = 27 DU/(2 hbar w)= {lam:.2f}",
        f"Niveles ~ DU/(hbar w)    = {dU/hw:.2f}  (diferencias finitas: ver resumen.txt)",
        "",
        f"B/hbar = 7.2 DU/(hbar w) = {b:.4f}",
        f"B/hbar numérico (pot. completo, E=0) = {exponente_wkb_numerico(u):.4f}",
        f"a_q                      = {a_q:.2f}",
        f"exp(-B/hbar)             = {np.exp(-b):.4e}",
        f"Gamma_q                  = {G_q:.4e} 1/s   (vida media 1/Gamma_q = {1/G_q*1e6:.2f} us)",
        f"T_esc^q (saturación, sin disipación) = {Tq*1e3:.2f} mK   (= T donde Gamma_t = Gamma_q, a_t = 1)",
        f"T* = hbar w_p/(2 pi k_B) = {hw/(2*np.pi*k_B)*1e3:.2f} mK   (cruce; Devoret et al. marcan 30 mK)",
        f"hbar w_p/(7.2 k_B)       = {hw/(7.2*k_B)*1e3:.2f} mK   (igualando solo exponentes)",
        f"ln(omega_p/2pi Gamma_q)  = {np.log(u.omega_p/(2*np.pi*G_q)):.2f}   (Devoret et al. comparan en 11)",
        "",
        "Gamma total = Gamma_t + Gamma_q, con a_t = 1 (sin corrección de Kramers).",
    ]
    if Q:
        L += ["", f"CON DISIPACIÓN, Q = {Q:g}: exponente x (1 + 0.87/Q) = x {1+0.87/Q:.4f}",
              f"Gamma_q                  = {G_qQ:.4e} 1/s",
              f"T_esc^q                  = {T_escape(u, dU, G_qQ)*1e3:.2f} mK"]
    TqQ = T_escape(u, dU, G_qQ) if Q else None
    return "\n".join(L), dict(b=b, a_q=a_q, G_q=G_q, Tq=Tq, lam=lam, cub=c, TqQ=TqQ, Q=Q)


def tabla_latex(u, d, dU):
    hw = hbar * u.omega_p
    filas = [
        (r"Corriente crítica $I_c$", rf"${u.Ic*1e6:.3f}\ \mu\mathrm{{A}}$"),
        (r"Capacitancia $C$", rf"${u.C*1e12:.2f}\ \mathrm{{pF}}$"),
        (r"Corriente de polarización $I/I_c$", rf"${u.s:.5f}$"),
        (r"$E_J/h$", rf"${u.E_J/GHz:.0f}\ \mathrm{{GHz}}$"),
        (r"$m_\gamma$", rf"${u.m:.3e}".replace("e-", r"\times10^{-") + r"}\ \mathrm{J\,s^2}$"),
        (r"$\omega_p/2\pi$", rf"${u.omega_p/(2*np.pi)/1e9:.3f}\ \mathrm{{GHz}}$"),
        (r"$\Delta U/h$ (exacta)", rf"${u.dU/GHz:.3f}\ \mathrm{{GHz}}$"),
        (r"$\Delta U_{cub}/h$", rf"${d['cub'].dU/GHz:.3f}\ \mathrm{{GHz}}$"),
        (r"$\Delta U/\hbar\omega_p$", rf"${dU/hw:.3f}$"),
        (r"$\lambda$", rf"${d['lam']:.1f}$"),
        (r"$B/\hbar$", rf"${d['b']:.2f}$"),
        (r"$a_q$", rf"${d['a_q']:.1f}$"),
        (r"$\Gamma_q$", rf"${d['G_q']:.2e}".replace("e+0", r"\times10^{") + r"}\ \mathrm{s^{-1}}$"),
        (r"$T_{esc}^{q}$ (sin disipación)", rf"${d['Tq']*1e3:.1f}\ \mathrm{{mK}}$"),
        (r"$T^{*}=\hbar\omega_p/2\pi k_B$", rf"${hw/(2*np.pi*k_B)*1e3:.1f}\ \mathrm{{mK}}$"),
    ]
    if d["Q"]:
        filas.append((rf"$T_{{esc}}^{{q}}$ con $Q={d['Q']:g}$", rf"${d['TqQ']*1e3:.1f}\ \mathrm{{mK}}$"))
    L = [r"\begin{table}[H]", r"    \centering",
         r"    \caption{Parámetros de la unión y resultados del escape.}",
         r"    \label{tab:parametros}", r"    \begin{tabular}{lr}", r"        \hline",
         r"        Cantidad & Valor \\", r"        \hline"]
    L += [f"        {a} & {b} \\\\" for a, b in filas]
    L += [r"        \hline", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(L)


def figura(u, dU, T, G_t, G_q, G, Tq, ruta):
    Tstar = hbar * u.omega_p / (2 * np.pi * k_B)
    plt = estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    Tm = T * 1e3

    a1.semilogy(Tm, G_t, color=COLORES[1], ls="--", lw=1.8, label=r"Térmico $\Gamma_t$")
    a1.axhline(G_q, color=COLORES[2], ls=":", lw=1.8, label=r"Cuántico $\Gamma_q$")
    a1.semilogy(Tm, G, color=COLORES[0], lw=2.4, label=r"Total $\Gamma_t+\Gamma_q$")
    a1.axvline(Tstar * 1e3, color="gray", lw=1)
    a1.set_ylim(G_q / 50, G.max() * 3)
    a1.set_xlabel(r"$T$ [mK]")
    a1.set_ylabel(r"$\Gamma$ [s$^{-1}$]")
    a1.set_title("(a) Tasa de escape", fontsize=12)
    a1.legend(loc="lower right")
    a1.text(Tstar * 1e3 + 1, G.max() / 2, rf"$T^*=\hbar\omega_p/2\pi k_B={Tstar*1e3:.1f}$ mK",
            fontsize=10, ha="left", va="top")

    a2.plot(Tm, Tm, color=COLORES[1], ls="--", lw=1.8, label=r"Solo térmico, $T_{esc}=T$")
    a2.axhline(Tq * 1e3, color=COLORES[2], ls=":", lw=1.8, label=rf"Solo cuántico, $T_{{esc}}^q={Tq*1e3:.1f}$ mK")
    a2.plot(Tm, T_escape(u, dU, G) * 1e3, color=COLORES[0], lw=2.4, label="Total")
    a2.set_xlabel(r"$T$ [mK]")
    a2.set_ylabel(r"$T_{esc}$ [mK]")
    a2.set_xlim(0, Tm.max()); a2.set_ylim(0, Tm.max())
    a2.set_title("(b) Temperatura de escape", fontsize=12)
    a2.legend(loc="upper left")

    fig.suptitle(rf"$I_c={u.Ic*1e6:.3f}\,\mu$A, $C={u.C*1e12:.2f}$ pF, $I/I_c={u.s:.5f}$, "
                 rf"$\omega_p/2\pi={u.omega_p/(2*np.pi)/1e9:.3f}$ GHz, "
                 rf"$\Delta U/\hbar\omega_p={dU/(hbar*u.omega_p):.2f}$", fontsize=11)
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ic", type=float, default=9.489e-6, help="Corriente crítica [A]")
    ap.add_argument("--capacitancia", type=float, default=6.35e-12, help="Capacitancia [F]")
    ap.add_argument("--corriente", type=float, default=9.40e-6, help="Corriente de polarización [A]")
    ap.add_argument("--barrera", choices=["exacta", "cubica"], default="exacta",
                    help="Barrera usada en las fórmulas (por defecto la exacta, Ec. 2.20)")
    ap.add_argument("--Q", type=float, default=None,
                    help="Factor de calidad para la corrección por disipación (Devoret et al.: 30)")
    ap.add_argument("--T-max", type=float, default=100e-3, help="Temperatura máxima [K]")
    ap.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent / "resultados" / "escape")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    a.salida.mkdir(parents=True, exist_ok=True)

    u = Union(a.ic, a.capacitancia, a.corriente / a.ic)
    dU = u.dU if a.barrera == "exacta" else PozoCubico(u).dU
    etiqueta = "exacta (Ec. 2.20)" if a.barrera == "exacta" else "cúbica (Ec. 4.16)"

    T = np.linspace(1e-4, a.T_max, 600)
    texto, d = resumen(u, dU, etiqueta, T, a.Q)
    print(texto)
    (a.salida / "resumen_escape.txt").write_text(texto + "\n", encoding="utf-8")
    (a.salida / "tabla_parametros.tex").write_text(tabla_latex(u, d, dU) + "\n", encoding="utf-8")

    _, _, G_q, G_t, G = tasas(u, dU, T)
    np.savetxt(a.salida / "escape_vs_T.csv",
               np.column_stack([T * 1e3, G_t, np.full_like(T, G_q), G, T_escape(u, dU, G) * 1e3]),
               delimiter=",", fmt="%.8g",
               header="T [mK],Gamma_t [1/s],Gamma_q [1/s],Gamma total [1/s],T_esc [mK]", comments="")
    figura(u, dU, T, G_t, G_q, G, d["Tq"], a.salida / "escape_vs_T.png")
    print(f"\nResultados guardados en: {a.salida.resolve()}")


if __name__ == "__main__":
    main()
