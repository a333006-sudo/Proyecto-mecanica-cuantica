"""Validación física y numérica del pozo Josephson con fronteras de Dirichlet."""

import unittest

import numpy as np
from scipy.constants import h, hbar

from niveles_energia_josephson import ParametrosJosephson, resolver_josephson


class PruebasJosephson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parametros = ParametrosJosephson()
        cls.base = resolver_josephson(N=4095, parametros=cls.parametros)

    def test_referencia_en_ghz_y_dos_candidatos_bajo_barrera(self):
        # Referencias independientes con 4095 puntos, redondeadas a 1 kHz.
        esperadas_ghz = np.array([1.927662, 5.587045, 9.112495])
        np.testing.assert_allclose(self.base.energias[:3] / h / 1e9,
                                   esperadas_ghz, rtol=0, atol=0.00000051)
        self.assertAlmostEqual(self.parametros.barrera / h / 1e9,
                               8.076295329957, places=8)
        self.assertAlmostEqual(self.parametros.omega_p / (2.0 * np.pi) / 1e9,
                               3.964287348456, places=8)
        np.testing.assert_array_equal(self.base.candidatos, np.array([0, 1]))
        self.assertGreater(self.base.energias[2], self.parametros.barrera)

    def test_fronteras_normalizacion_ortogonalidad_y_residuo(self):
        solucion = self.base
        self.assertEqual(solucion.psi.shape[0], 4097)
        self.assertTrue(np.all(solucion.psi[[0, -1], :] == 0.0))
        gram = solucion.dgamma * solucion.psi.T @ solucion.psi
        np.testing.assert_allclose(gram, np.eye(solucion.energias.size),
                                   rtol=0, atol=2e-11)
        psi = solucion.psi[1:-1, :]
        Hpsi = (2.0 * solucion.c + solucion.potencial[1:-1])[:, None] * psi
        Hpsi[1:, :] -= solucion.c * psi[:-1, :]
        Hpsi[:-1, :] -= solucion.c * psi[1:, :]
        residuo = Hpsi - psi * solucion.energias[None, :]
        escala = hbar * self.parametros.omega_p
        relativo = (np.linalg.norm(residuo, axis=0)
                    / (escala * np.linalg.norm(psi, axis=0)))
        self.assertTrue(np.all(relativo < 1e-8), relativo)

    def test_convergencia_cuadratica_sin_solucion_analitica(self):
        energias = np.array([
            resolver_josephson(N=N, parametros=self.parametros).energias[:2]
            for N in (1023, 2047, 4095)
        ]) / (hbar * self.parametros.omega_p)
        diferencias = np.diff(energias, axis=0)
        self.assertTrue(np.all(diferencias > 0.0))
        orden = np.log2(diferencias[0] / diferencias[1])
        self.assertTrue(np.all((orden > 1.98) & (orden < 2.04)), orden)

    def test_sensibilidad_a_la_frontera_derecha_en_region_prohibida(self):
        # La sensibilidad a una pared no desaparece solo porque U(pared)>E.
        referencias_mhz = {
            -0.02: np.array([1.9723052, 43.8719144]),
            0.02: np.array([-0.8237796, -25.2249865]),
        }
        for desplazamiento, referencia in referencias_mhz.items():
            with self.subTest(desplazamiento=desplazamiento):
                derecha = self.parametros.gamma_der + desplazamiento
                N = int(round((derecha - self.parametros.gamma_izq)
                              / self.base.dgamma)) - 1
                nueva = resolver_josephson(N=N, parametros=self.parametros,
                                          gamma_der=derecha)
                self.assertLess(abs(nueva.dgamma / self.base.dgamma - 1.0), 1e-5)
                candidatos = self.base.candidatos
                alturas = nueva.potencial[[0, -1], None]
                self.assertTrue(np.all(alturas > self.base.energias[candidatos]))
                self.assertTrue(np.all(alturas > nueva.energias[candidatos]))
                delta_mhz = ((nueva.energias[candidatos]
                              - self.base.energias[candidatos]) / h / 1e6)
                np.testing.assert_allclose(delta_mhz, referencia,
                                           rtol=1e-6, atol=1e-5)
                self.assertTrue(np.all(np.sign(delta_mhz) == -np.sign(desplazamiento)))


if __name__ == "__main__":
    unittest.main()
