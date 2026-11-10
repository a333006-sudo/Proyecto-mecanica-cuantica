"""Controles científicos; las comparaciones de energía usan escalas físicas."""

import unittest

import numpy as np
from scipy.constants import hbar

from oscilador_armonico import resolver_hamiltoniano, resolver_oscilador


class PruebasOscilador(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.omega = 1e15
        cls.escala = hbar * cls.omega
        cls.mallas = [resolver_oscilador(N=N, omega=cls.omega)
                      for N in (99, 199, 399)]

    def test_potencial_constante_con_espectro_discreto_exacto(self):
        # La caja discreta tiene solución exacta, independiente del oscilador.
        N, niveles = 47, 5
        c, U0 = 1.9e-20, -3.0e-21
        energias, vectores = resolver_hamiltoniano(c, np.full(N, U0), niveles)
        k = np.arange(1, niveles + 1)
        exactas = U0 + 4.0 * c * np.sin(np.pi * k / (2.0 * (N + 1)))**2
        np.testing.assert_allclose(energias / c, exactas / c,
                                   rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(vectores.T @ vectores, np.eye(niveles),
                                   rtol=0, atol=1e-12)

    def test_fronteras_normalizacion_y_ortogonalidad(self):
        for solucion in self.mallas:
            with self.subTest(N=solucion.x.size - 2):
                self.assertTrue(np.all(solucion.psi[[0, -1], :] == 0.0))
                gram = solucion.dx * solucion.psi.T @ solucion.psi
                np.testing.assert_allclose(gram, np.eye(4), rtol=0, atol=1e-12)
                self.assertAlmostEqual(
                    solucion.dx / solucion.ell,
                    12.0 / (solucion.x.size - 1), places=14,
                )

    def test_residuo_de_schrodinger_en_unidades_de_hbar_omega(self):
        solucion = self.mallas[-1]
        psi = solucion.psi[1:-1, :]
        Hpsi = (2.0 * solucion.c + solucion.potencial)[:, None] * psi
        Hpsi[1:, :] -= solucion.c * psi[:-1, :]
        Hpsi[:-1, :] -= solucion.c * psi[1:, :]
        residuo = Hpsi - psi * solucion.energias[None, :]
        relativo = (np.linalg.norm(residuo, axis=0)
                    / (self.escala * np.linalg.norm(psi, axis=0)))
        self.assertTrue(np.all(relativo < 1e-9), relativo)

    def test_convergencia_de_segundo_orden(self):
        exactas = np.arange(4) + 0.5
        errores = np.array([exactas - s.energias / self.escala
                            for s in self.mallas])
        # En L=6 ell domina la corrección negativa de diferencias finitas.
        self.assertTrue(np.all(errores > 0.0))
        pasos = np.array([s.dx for s in self.mallas])
        ordenes = (np.log(errores[:-1] / errores[1:])
                   / np.log(pasos[:-1] / pasos[1:])[:, None])
        self.assertTrue(np.all((ordenes > 1.98) & (ordenes < 2.03)), ordenes)

    def test_frontera_a_paso_fijo_y_error_de_discretizacion(self):
        dominios = [resolver_oscilador(N=N, factor_L=L, omega=self.omega)
                    for L, N in ((3.0, 199), (6.0, 399), (9.0, 599))]
        for solucion in dominios:
            self.assertAlmostEqual(solucion.dx / solucion.ell, 0.03, places=14)
        energias = np.array([s.energias / self.escala for s in dominios])
        self.assertTrue(np.all(energias[0] > energias[1]))
        self.assertGreater(np.max(np.abs(energias[0] - energias[2])), 1e-3)
        self.assertLess(np.max(np.abs(energias[1] - energias[2])), 1e-7)
        # Ampliar L no elimina el error de la derivada a paso fijo.
        self.assertGreater(abs(energias[2, 3] - 3.5), 1e-4)


if __name__ == "__main__":
    unittest.main()
