import unittest

import numpy as np

from tests.label_source_audit import PALETTE, alignment, decode


class LabelSourceAuditTest(unittest.TestCase):
    def test_exact_palette_and_legacy_collisions(self):
        rgb = np.array([(*PALETTE, (127, 0, 1), (1, 1, 1))], dtype=np.uint8)
        exact, legacy = decode(rgb)
        self.assertEqual(exact.tolist(), [[0, 1, 2, 3, 4, 5, 6, 7, 7]])
        self.assertEqual(legacy.tolist(), [[0, 1, 2, 3, 4, 5, 6, 2, 0]])

    def test_alignment_excludes_unknown_and_ignore(self):
        result = alignment(np.array([[1, 2, 3, 0, 1]]), np.array([[1, 3, 7, 2, 0]]))
        self.assertEqual((result["shared"], result["equal"], result["full_only"],
                          result["eroded_only"]), (2, 1, 2, 1))
