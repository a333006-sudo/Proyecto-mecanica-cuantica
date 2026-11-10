"""Niveles de un pozo Josephson con potencial completo y fronteras Dirichlet.

python niveles_energia_josephson.py --sin-mostrar
Las energías se calculan en J, relativas al mínimo; E/h es una frecuencia.
Se importa el solucionador del ejemplo del oscilador, sin modificarlo.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.constants import e, h, hbar
from scipy.optimize import brentq

from oscilador_armonico import (
    normalizar_y_agregar_fronteras,
    resolver_hamiltoniano,
)


@dataclass(frozen=True)
class ParametrosJosephson:
    I_c: float = 9.489e-6     # A
    C: float = 6.35e-12       # F
    I: float = 9.40e-6        # A

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.I_c, self.C, self.I)):
            raise ValueError("Los parámetros deben ser finitos.")
        if self.C <= 0 or not 0 < self.I < self.I_c:
            raise ValueError("Se requiere C > 0 y 0 < I < I_c para este pozo inclinado.")

    @property
    def s(self):
        return self.I / self.I_c

    @property
    def E_J(self):
        return hbar * self.I_c / (2.0 * e)

    @property
    def m_gamma(self):
        return self.C * (hbar / (2.0 * e))**2  # J s²; no es una masa en kg

    @property
    def alpha(self):
        return np.arcsin(self.s)

    @property
    def gamma_izq(self):
        return -np.pi - self.alpha

    @property
    def gamma_der(self):
        return np.pi - self.alpha

    def potencial(self, gamma):
        gamma = np.asarray(gamma)
        return -self.E_J * (np.cos(gamma) + self.s * gamma)

    def potencial_desplazado(self, gamma):
        return self.potencial(gamma) - self.potencial(self.alpha)

    @property
    def omega_p(self):
        # U''(alpha) = E_J cos(alpha).
        return np.sqrt(self.E_J * np.cos(self.alpha) / self.m_gamma)

    @property
    def barrera(self):
        # El máximo físico no cambia cuando se mueve una frontera numérica.
        return float(self.potencial_desplazado(self.gamma_der))


@dataclass
class SolucionJosephson:
    parametros: ParametrosJosephson
    gamma: np.ndarray           # N+2 puntos; fase en radianes
    dgamma: float
    c: float                   # J
    potencial: np.ndarray      # N+2 valores U(gamma)-U(alpha), J
    energias: np.ndarray       # cuatro niveles, J desde el mínimo
    psi: np.ndarray            # columnas; dgamma * sum(|psi|²) = 1

    @property
    def candidatos(self):
        return np.flatnonzero(self.energias < self.parametros.barrera)


def resolver_josephson(N=4095, parametros=None, gamma_izq=None, gamma_der=None):
    """Resuelve los cuatro niveles más bajos sobre un intervalo uniforme."""
    p = parametros if parametros is not None else ParametrosJosephson()
    izquierda = p.gamma_izq if gamma_izq is None else gamma_izq
    derecha = p.gamma_der if gamma_der is None else gamma_der
    if not isinstance(N, (int, np.integer)) or N < 4:
        raise ValueError("N debe ser un entero >= 4 de puntos interiores.")
    if not (np.isfinite(izquierda) and np.isfinite(derecha)
            and izquierda < p.alpha < derecha):
        raise ValueError("Los extremos deben ser finitos y contener el mínimo.")
    gamma = np.linspace(izquierda, derecha, N + 2)
    dgamma = (derecha - izquierda) / (N + 1)
    potencial = p.potencial_desplazado(gamma)
    c = hbar**2 / (2.0 * p.m_gamma * dgamma**2)
    energias, vectores = resolver_hamiltoniano(c, potencial[1:-1], 4)
    psi = normalizar_y_agregar_fronteras(vectores, dgamma)
    return SolucionJosephson(p, gamma, dgamma, c, potencial, energias, psi)


@dataclass
class CambioFrontera:
    lado: str
    desplazamiento: float
    solucion: SolucionJosephson
    aceptados: np.ndarray
    cambio: np.ndarray          # E_movida - E_base en la malla comparable, J
    cambio_richardson: np.ndarray  # estimación sin el término O(dgamma²), J


def estudiar_fronteras(base):
    """Mueve un extremo ±0.02 rad manteniendo el otro exactamente fijo.

    Redondea N para conservar aproximadamente dgamma. Una malla adicional
    con la mitad del paso en cada dominio permite estimar el cambio de
    frontera mediante extrapolación de Richardson: E*=(4 E_fina-E_gruesa)/3.
    Esta extrapolación es un diagnóstico, no sustituye los resultados N=4095.
    """
    N_base = base.gamma.size - 2
    fina_base = resolver_josephson(2 * N_base + 1, base.parametros,
                                  base.gamma[0], base.gamma[-1])
    cambios = []
    for lado in ("izquierda", "derecha"):
        for desplazamiento in (-0.02, 0.02):
            izq, der = base.gamma[[0, -1]].copy()
            if lado == "izquierda":
                izq += desplazamiento
                extremo = izq
            else:
                der += desplazamiento
                extremo = der
            N = max(4, int(round((der - izq) / base.dgamma)) - 1)
            movida = resolver_josephson(N, base.parametros, izq, der)
            fina = resolver_josephson(2 * N + 1, base.parametros, izq, der)
            U_extremo = base.parametros.potencial_desplazado(extremo)
            # Solo interpretar como prueba en región prohibida si la condición
            # se cumple tanto para el nivel inicial como para el desplazado.
            aceptados = ((base.energias < base.parametros.barrera)
                         & (movida.energias < base.parametros.barrera)
                         & (U_extremo > base.energias)
                         & (U_extremo > movida.energias)
                         & (U_extremo > fina.energias))
            cambio = movida.energias - base.energias
            cambio_fino = fina.energias - fina_base.energias
            cambios.append(CambioFrontera(
                lado, desplazamiento, movida, aceptados, cambio,
                (4.0 * cambio_fino - cambio) / 3.0,
            ))
    return cambios


def crear_informe(mallas, cambios):
    base = mallas[-1]
    p = base.parametros
    lineas = [
        "POZO JOSEPHSON: POTENCIAL COMPLETO, FRONTERAS DIRICHLET",
        f"I_c={p.I_c:.10e} A; C={p.C:.10e} F; I={p.I:.10e} A; s={p.s:.12f}",
        f"E_J={p.E_J:.12e} J; m_gamma={p.m_gamma:.12e} J s²",
        f"alpha={p.alpha:.12f} rad; U(alpha)={p.potencial(p.alpha):.12e} J",
        f"Dominio inicial: [{p.gamma_izq:.12f}, {p.gamma_der:.12f}] rad",
        f"omega_p={p.omega_p:.12e} rad/s; omega_p/(2 pi)={p.omega_p/(2*np.pi*1e9):.9f} GHz",
        f"Delta U={p.barrera:.12e} J; Delta U/h={p.barrera/(h*1e9):.9f} GHz",
        "Energías relativas al mínimo; E/h es frecuencia, no frecuencia angular.",
        "", "1. Cuatro autovalores con N=4095",
        " n       E [J]              E/h [Hz]          E/h [GHz]    E < Delta U",
    ]
    for n, energia in enumerate(base.energias):
        lineas.append(f" {n}   {energia:.12e}   {energia/h:.9e}   "
                      f"{energia/(h*1e9):.9f}     {'sí' if n in base.candidatos else 'no'}")
    lineas += ["", "2. Convergencia, extremos iniciales fijos",
               "   N     dgamma [rad]       E0/h [GHz]    E1/h [GHz]    E2/h [GHz]    E3/h [GHz]"]
    for s in mallas:
        lineas.append(f" {s.gamma.size-2:4d}   {s.dgamma:.10f}   "
                      + "  ".join(f"{v/(h*1e9):12.9f}" for v in s.energias))
    lineas += ["", "Cambios entre mallas (E_fina-E_gruesa)/h [MHz]:"]
    for anterior, actual in zip(mallas[:-1], mallas[1:]):
        lineas.append(f" {anterior.gamma.size-2} -> {actual.gamma.size-2}: "
                      + "  ".join(f"n={n}: {v/(h*1e6):+.9f}"
                                  for n, v in enumerate(actual.energias-anterior.energias)))
    diferencias = [mallas[i+1].energias - mallas[i].energias for i in range(2)]
    orden = np.log2(np.abs(diferencias[0] / diferencias[1]))
    lineas.append(" Orden p estimado por tres mallas: "
                  + "  ".join(f"n={n}: {v:.6f}" for n, v in enumerate(orden)))
    lineas.append(" Corrección estimada (E_continuo-E4095)/h = (E4095-E2047)/(3h) [MHz]: "
                  + "  ".join(f"n={n}: {v/(3*h*1e6):+.9f}"
                              for n, v in enumerate(diferencias[-1])))

    lineas += ["", "3. Fronteras: solo se mueve el extremo indicado; el otro queda fijo.",
               "delta gamma es el desplazamiento con signo en la coordenada gamma.",
               "Cambios de energía con signo respecto al dominio inicial N=4095.",
               "Delta E* elimina el término de error O(dgamma²) usando mallas con paso mitad.",
               "Delta E - Delta E* estima la contribución de discretización al cambio medido."]
    for cambio in cambios:
        s = cambio.solucion
        extremo = s.gamma[0 if cambio.lado == "izquierda" else -1]
        lineas += [
            "", f" {cambio.lado}, desplazamiento={cambio.desplazamiento:+.5f} rad; N={s.gamma.size-2}",
            f" dominio=[{s.gamma[0]:.12f}, {s.gamma[-1]:.12f}] rad; dgamma={s.dgamma:.12e} rad",
            f" variación del paso={100*(s.dgamma/base.dgamma-1):+.8f} %; U(extremo)/h={p.potencial_desplazado(extremo)/(h*1e9):.9f} GHz (desde el mínimo)",
            " n    E_movida/h [GHz]    Delta E [J]       Delta E/h [MHz]    Delta E*/h [MHz]   (Delta E-Delta E*)/h [MHz]",
        ]
        for n in base.candidatos:
            if not cambio.aceptados[n]:
                lineas.append(f" {n}: OMITIDO: el nivel no cumple la condición de región prohibida o de barrera.")
                continue
            bruto = cambio.cambio[n]
            corregido = cambio.cambio_richardson[n]
            lineas.append(f" {n}    {s.energias[n]/(h*1e9):.9f}    {bruto:+.8e}   "
                          f"{bruto/(h*1e6):+.9e}   {corregido/(h*1e6):+.9e}   "
                          f"{(bruto-corregido)/(h*1e6):+.9e}")
    lineas += ["", "4. Normas discretas, incluidos los ceros de frontera"]
    for n, norma in enumerate(base.dgamma * np.sum(np.abs(base.psi)**2, axis=0)):
        lineas.append(f" n={n}: {norma:.15f}")
    lineas += ["", "Los estados bajo la barrera son candidatos a niveles localizados del pozo.",
               "Son autovalores reales de un intervalo cerrado con Dirichlet, no energías",
               "complejas de resonancias ni un cálculo de tiempos de escape por túnel.",
               "La sensibilidad a la frontera derecha debe acompañar la precisión de malla.",
               "La extrapolación también está limitada por errores de orden superior y redondeo."]
    return "\n".join(lineas)


def graficar(base, ruta, mostrar=True):
    import matplotlib
    if not mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = base.parametros
    # Ampliación alrededor del pozo: el máximo izquierdo es muchísimo más alto.
    visibles = np.flatnonzero(base.potencial <= 1.6 * p.barrera)
    izq = base.gamma[max(0, visibles[0] - 3)]
    der = p.gamma_der + 0.35 * (p.gamma_der - p.alpha)
    gamma_fig = np.linspace(izq, der, 1400)
    escala = h * 1e9
    colores = ["#176b87", "#b45309", "#7e22ce", "#15803d"]
    fig, (ax, densidad) = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
    ax.plot(gamma_fig, p.potencial_desplazado(gamma_fig)/escala, color="#334155", lw=2,
            label=r"$[U(\gamma)-U(\alpha)]/h$")
    ax.scatter([p.alpha], [0], color="#15803d", zorder=5, label="Mínimo")
    ax.scatter([p.gamma_der], [p.barrera/escala], color="#be123c", zorder=5,
               label="Máximo derecho")
    ax.axhline(p.barrera/escala, color="#be123c", ls=":", lw=1)
    for n in base.candidatos:
        energia = base.energias[n]
        izquierda = brentq(lambda x: p.potencial_desplazado(x)-energia, p.gamma_izq, p.alpha)
        derecha = brentq(lambda x: p.potencial_desplazado(x)-energia, p.alpha, p.gamma_der)
        ax.hlines(energia/escala, izquierda, derecha, color=colores[n], lw=2.5,
                  label=rf"$n={n}$: {energia/escala:.4f} GHz")
        densidad.plot(base.gamma, np.abs(base.psi[:, n])**2, color=colores[n], lw=2,
                      label=rf"$|\psi_{n}|^2$")
    for panel in (ax, densidad):
        panel.axvline(p.gamma_der, color="#be123c", ls="--", lw=1)
        panel.axvspan(p.gamma_der, der, color="#e2e8f0", alpha=0.7)
        panel.set_xlim(izq, der)
        panel.set_xlabel(r"Fase $\gamma$ [rad]")
        panel.grid(alpha=0.2)
        panel.legend(fontsize=9, loc="upper left")
    densidad.axvline(p.alpha, color="#15803d", ls=":", lw=1)
    ax.set_ylim(-0.07*p.barrera/escala, 1.65*p.barrera/escala)
    ax.set_ylabel(r"Energía / $h$ [GHz], desde el mínimo")
    densidad.set_ylabel(r"Densidad $|\psi_n(\gamma)|^2$ [rad$^{-1}$]")
    densidad.set_ylim(bottom=0)
    ax.set_title("Potencial completo y niveles bajo la barrera")
    densidad.set_title("Densidades normalizadas: ampliación del pozo")
    fig.suptitle(f"Unión Josephson · N={base.gamma.size-2} · "
                 f"dominio [{base.gamma[0]:.4f}, {base.gamma[-1]:.4f}] rad\n"
                 "Línea roja vertical: frontera derecha; sombreado: fuera del dominio", fontsize=11)
    fig.savefig(ruta, dpi=180)
    if mostrar:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic", type=float, default=9.489e-6, help="Corriente crítica en A.")
    parser.add_argument("--capacitancia", type=float, default=6.35e-12, help="Capacitancia en F.")
    parser.add_argument("--corriente", type=float, default=9.40e-6, help="Corriente de polarización en A.")
    parser.add_argument("--sin-mostrar", action="store_true", help="Guarda la gráfica sin abrir ventana.")
    parser.add_argument("--salida", type=Path,
                        default=Path(__file__).resolve().parent / "resultados_josephson")
    args = parser.parse_args()
    try:
        parametros = ParametrosJosephson(args.ic, args.capacitancia, args.corriente)
    except ValueError as error:
        parser.error(str(error))
    mallas = [resolver_josephson(N, parametros) for N in (1023, 2047, 4095)]
    cambios = estudiar_fronteras(mallas[-1])
    informe = crear_informe(mallas, cambios)
    print(informe)
    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "informe.txt").write_text(informe + "\n", encoding="utf-8")
    graficar(mallas[-1], args.salida / "niveles_y_densidades.png", mostrar=not args.sin_mostrar)
    print(f"\nResultados guardados en: {args.salida.resolve()}")


if __name__ == "__main__":
    main()
