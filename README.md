# LSSFind

Python implementation of LSSFind and LocalLSSFind.

## Installation

This package is available at PyPI and can be installed using
```bash
pip install lssfind
```

## Usage

The main algorithms of LSSFind and LocalLSSFind are implemented as `get_prevalent_interactions` and `get_sample_interactions` respectively. They can be used with Random Forest from scikit-learn.

```python
from lssfind import get_prevalent_interactions, get_sample_interactions
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
rf = RandomForestRegressor()
rf.fit(X, y)
get_prevalent_interactions(rf, impurity_decrease_threshold=1., min_weight=0.5)
get_sample_interactions(rf, impurity_decrease_threshold=1., testpoints=[[0, 0, 0, 0]], min_weight_dwp=0.5, min_weight_pp=0.25)
```

## Literature

For theoretical background about LSSFind, see
> M. Behr, Y. Wang, X. Li, & B. Yu, Provable Boolean interaction recovery from tree ensemble obtained via random forests, *Proc. Natl. Acad. Sci. U.S.A.* 119 (22) e2118636119, https://doi.org/10.1073/pnas.2118636119 (2022).

Theoretical background about LocalLSSFind can be found in
> K. Vuk, N. A. Ihlo, & M. Behr, Provable Recovery of Locally Important Signed Features and Interactions from Random Forest, *arXiv*, https://arxiv.org/abs/2512.11081 (2025).

## Acknowledgement

Part of our implementation is based on Python code which was provided by Yu Wang
as part of iterative Random Forest, a project of the group of Bin Yu,
see https://github.com/Yu-Group/iterative-Random-Forest/blob/master/irf/utils.py
