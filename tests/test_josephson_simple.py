"""Pruebas de josephson.py.  Ejecutar desde la carpeta del proyecto:
    python -m unittest discover tests
"""

import unittest

import numpy as np
from scipy.constants import e, hbar

from josephson import (PUNTOS_POZO, PozoCubico, Union, exponente_wkb_numerico,
                       niveles_pozo)

IC, C = 9.489e-6, 6.35e-12


class PruebasJosephson(unittest.TestCase):

    def test_formulas_cerradas(self):
        # omega_p (Ec. 2.26) y Delta U (Ec. 2.20) contra el código.
        s = 0.99
        u = Union(IC, C, s)
        wp = np.sqrt(2 * e * IC / (hbar * C)) * (1 - s**2)**0.25
        dU = 2 * u.E_J * (np.sqrt(1 - s**2) - s * np.arccos(s))
        self.assertAlmostEqual(u.omega_p / wp, 1.0, places=12)
        self.assertAlmostEqual(u.dU / dU, 1.0, places=9)

    def test_pozo_profundo_coincide_con_perturbaciones(self):
        # Con Delta U >> hbar omega_p la teoría de perturbaciones debe funcionar.
        u = Union(IC, C, 0.95)
        E = niveles_pozo(u)["E"]
        hw = hbar * u.omega_p
        for n in range(3):
            self.assertLess(abs(E[n] - u.E_perturbativo(n)) / hw, 0.005)

    def test_niveles_estables_respecto_a_la_pared(self):
        # El nivel base no debe depender de dónde se ponga la pared derecha.
        u = Union(IC, C, 9.40e-6 / IC)
        res = niveles_pozo(u)
        self.assertEqual(len(res["E"]), 2)
        self.assertLess(res["dE"][0] / (hbar * u.omega_p), 1e-3)
        self.assertTrue(np.all(res["E"] < u.dU))

    def test_convergencia_de_segundo_orden(self):
        u = Union(IC, C, 0.99)
        dg = (u.g_max - u.g_min) / PUNTOS_POZO
        E = [niveles_pozo(u, dg / k)["E"][0] for k in (1, 2, 4)]
        orden = np.log2((E[1] - E[0]) / (E[2] - E[1]))
        self.assertAlmostEqual(orden, 2.0, delta=0.1)

    def test_integral_wkb_cubica_coincide_con_formula_cerrada(self):
        # La integral numérica en el pozo cúbico debe dar B = 36 DU_cub/(5 hbar w_p).
        c = PozoCubico(Union(IC, C, 0.99))
        self.assertAlmostEqual(exponente_wkb_numerico(c) / c.exponente_wkb(), 1.0, places=4)

    def test_cubico_sobrestima_la_barrera_y_converge_en_Ic(self):
        cocientes = [PozoCubico(Union(IC, C, s)).dU / Union(IC, C, s).dU
                     for s in (0.95, 0.99, 0.999)]
        self.assertTrue(all(q > 1 for q in cocientes))
        self.assertTrue(cocientes[0] > cocientes[1] > cocientes[2])
        self.assertLess(cocientes[2] - 1, 0.002)


if __name__ == "__main__":
    unittest.main()
