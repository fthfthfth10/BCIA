import unittest

import numpy as np

from potsdam_v2 import PALETTE, exact_labels


class PotsdamV2Test(unittest.TestCase):
    def test_exact_palette_and_legacy_collision_rejected(self):
        labels = exact_labels(np.asarray([PALETTE], dtype=np.uint8))
        self.assertEqual(labels.tolist(), [list(range(7))])
        with self.assertRaisesRegex(ValueError, "Noncanonical"):
            exact_labels(np.asarray([[(127, 0, 1)]], dtype=np.uint8))

    def test_reject_non_rgb_arrays(self):
        with self.assertRaisesRegex(ValueError, "uint8 RGB"):
            exact_labels(np.zeros((2, 2), dtype=np.uint8))
