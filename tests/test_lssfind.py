import unittest
from lssfind import get_prevalent_interactions, get_sample_interactions
import numpy as np
from collections import OrderedDict
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

class RFtests(unittest.TestCase):
    """
    Tests on scikit-learn Random Forest.
    """

    def setUp(self):
        """
        Create dataset and train Random Forest.
        """
        X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
        self.rf = RandomForestRegressor(random_state=0)
        self.rf.fit(X, y)

    def test_global_interactions(self):
        output = get_prevalent_interactions(self.rf, impurity_decrease_threshold=1., min_weight=0.5)
        expected = OrderedDict([(frozenset({(np.int64(1), '-')}), 1.),
             (frozenset({(np.int64(1), '+')}), 1.),
             (frozenset({(np.int64(0), '+'), (np.int64(1), '-')}),
              0.9975),
             (frozenset({(np.int64(0), '-'), (np.int64(1), '-')}),
              0.9975),
             (frozenset({(np.int64(0), '-')}), 0.99),
             (frozenset({(np.int64(0), '+')}), 0.99),
             (frozenset({(np.int64(0), '-'), (np.int64(1), '+')}),
              0.9825),
             (frozenset({(np.int64(0), '+'), (np.int64(1), '+')}),
              0.9825)])
        self.assertEqual(set(output.keys()), set(expected.keys()))
        for interaction in output:
            with self.subTest(interaction=interaction):
                self.assertAlmostEqual(
                    output.get(interaction, 0.),
                    expected.get(interaction, 0.), 7
                )

    def test_local_interactions(self):
        output = get_sample_interactions(self.rf, impurity_decrease_threshold=1., testpoints=[[0, 0, 0, 0]], min_weight_dwp=0.5, min_weight_pp=0.25)[0]
        expected = OrderedDict([(frozenset({(np.int64(1), '-')}),
               (1., 0.71)),
              (frozenset({(np.int64(0), '+')}),
               (0.99, 0.59)),
              (frozenset({(np.int64(0), '+'), (np.int64(1), '-')}),
               (0.9975, 0.44)),
              (frozenset({(np.int64(0), '-')}),
               (0.99, 0.41)),
              (frozenset({(np.int64(1), '+')}),
               (1., 0.29)),
              (frozenset({(np.int64(0), '-'), (np.int64(1), '-')}),
               (0.9975, 0.27))])
        self.assertEqual(set(output.keys()), set(expected.keys()))
        for interaction in output:
            with self.subTest(interaction=interaction):
                self.assertAlmostEqual(
                    output.get(interaction, (0., 0.))[0],
                    expected.get(interaction, (0., 0.))[0], 7
                )
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()
