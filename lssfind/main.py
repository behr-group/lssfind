from sklearn.tree import _tree
from collections import OrderedDict
from wfpgrowth import find_frequent_patterns

def compute_impurity_decrease(dtree):
    """Compute the impurity decrease at each node."""
    impurity = dtree.tree_.impurity
    weight = dtree.tree_.n_node_samples
    weight = [x / weight[0] for x in weight] # normalize the weight
    impurity_decrease = []
    n_nodes = len(weight)
    for i in range(n_nodes):
        left_child = dtree.tree_.children_left[i]
        right_child = dtree.tree_.children_right[i]
        if left_child < 0 or right_child < 0:
            impurity_decrease.append(-1)
        else:
            curr_impurity = weight[i] * impurity[i]
            left_impurity = weight[left_child] * impurity[left_child]
            right_impurity = weight[right_child] * impurity[right_child]
            impurity_decrease.append(curr_impurity - left_impurity - right_impurity)
    return impurity_decrease
   
def get_filtered_feature_paths(dtree_or_rf, threshold, signed=True, 
                               weight_scheme='depth'):
    """Get the set of feature paths and their weights filtered by
    the impurity decrease.
    """
    if hasattr(dtree_or_rf, 'tree_'):
        impurity_decrease = compute_impurity_decrease(dtree_or_rf)
        features = dtree_or_rf.tree_.feature
        filtered = [x >= threshold for x in impurity_decrease]
        if signed:
            tree_paths = all_tree_signed_paths(dtree_or_rf)
            feature_paths = []
            for path in tree_paths:
                tmp = [(features[x[0]],x[1]) for x in path if filtered[x[0]]]
                # remove features that appear twice
                cleaned = []
                cache = set()
                for k in tmp:
                    if k[0] not in cache:
                        cleaned.append(k)
                        cache.add(k[0])
                feature_paths.append(cleaned)
            if weight_scheme == 'depth':
                weight = [2 ** (1-len(path)) for path in tree_paths]
            elif weight_scheme == 'samplesize':
                samplesize_per_node = dtree_or_rf.tree_.weighted_n_node_samples
                weight = [samplesize_per_node[path[-1]] for path in tree_paths]
            elif weight_scheme == 'label':
                raise NotImplementedError("this has not been implemented yet.")
            else:
                raise ValueError("weight scheme is not allowed.")
        else:
            tree_paths = all_tree_paths(dtree_or_rf)
            feature_paths = []
            for path in tree_paths:
                feature_paths.append(list(set([features[x] for x in path if filtered[x]])))
            if weight_scheme == 'depth':
                weight = [2 ** (1-len(path)) for path in tree_paths]
            elif weight_scheme == 'samplesize':
                samplesize_per_node = dtree_or_rf.tree_.weighted_n_node_samples
                weight = [samplesize_per_node[path[-1]] for path in tree_paths]
            else:
                raise ValueError("weight scheme is not allowed.")
        # make sure the weight sums up to 1.
        total = sum(weight)
        weight = [w / total for w in weight]
        return feature_paths, weight
    elif hasattr(dtree_or_rf, 'estimators_'):
        all_fs = []
        all_ws = []
        for tree in dtree_or_rf.estimators_:
            feature_paths, weight = get_filtered_feature_paths(tree, threshold, signed, weight_scheme)
            all_fs += feature_paths
            all_ws += [w / dtree_or_rf.n_estimators for w in weight]
        return all_fs, all_ws

def get_prevalent_interactions(
        rf,
        impurity_decrease_threshold,
        min_weight=0.01,
        weight_scheme="depth",
        signed=True,
        mask=None,
        max_interaction_size=6,
    ):
    """Compute the prevalent interactions and their prevalence.

    We use FP-Growth to find a series of interactions with their weighted prevalence.
    For signed interactions, these are adjusted to the length of the interaction.
    
    Parameters
    ----------
    rf : Random Forest
        The random forest model from scikit-learn.
    impurity_decrease_threshold : float
        If a split results in a decrease smaller than this parameter, then it
        will not be considered in the path.
    min_weight : float, default=0.01
        The minimum weighted prevalence a interaction must have to be considered.
    weight_scheme : "depth" or "samplesize", default="depth"
        How to compute the weight of paths for prevalence.
    signed : bool, default=True
        Whether to consider signed interactions.
    mask : dict, optional
        If provided, his stores the name of each feature. Features with the same name
        are treated as the same.
    max_interaction_size: int, default=6
        Maximum size for interactions to be considered.

    Returns
    -------
    dict
        Dictionary with found interactions as keys and their prevalences as values.
        If signed is True, prevalences are adjusted for the length of the interaction.
    """
    feature_paths, weight = get_filtered_feature_paths(
        rf,
        impurity_decrease_threshold,
        signed=signed,
        weight_scheme=weight_scheme,
    )
    if mask is not None:
        if signed:
            feature_paths = [sorted([(mask[elem[0]], elem[1]) for elem in x]) for x in feature_paths]
        else:
            feature_paths = [sorted([mask[elem] for elem in x]) for x in feature_paths]
    patterns = find_frequent_patterns(
        feature_paths, weight,
        min_weight / 2**max_interaction_size if signed else min_weight
    )
    prevalence = OrderedDict(
        sorted(
            filter(
                lambda x: len(x[0]) <= max_interaction_size and x[1] >= min_weight,
                map(lambda x: (frozenset(x[0]), x[1]*2**len(x[0]) if signed else x[1]), patterns.items())
            ),
            key=lambda t: -t[1]
        ),
    )
    return prevalence

def all_tree_signed_paths(dtree, root_node_id=0):
    """ Get all the individual tree signed paths from root node to the leaves
    for a decision tree.

    Parameters
    ----------
    dtree : Decision Tree
        An individual decision tree from scikit-learn.
    root_node_id : int, default=0
        The index of the root node of the tree. Default 0 consideres the entire tree,
        while other values would consider a subtree or lead to an error.

    Returns
    -------
    list of lists of tuples
        List of all paths in tree, starting from root node and denoted as list of tuples
        (node_id, direction) with direction '+' or '-' for inner nodes and (node_id,)
        for final leaf node.
    """
    # Use these lists to parse the tree structure
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right

    if root_node_id is None:
        paths = []

    if root_node_id == _tree.TREE_LEAF:
        raise ValueError(f"Invalid node_id {_tree.TREE_LEAF}")

    # if left/right is None we'll get empty list anyway
    # feature_id = dtree.tree_.feature[root_node_id] 
    if children_left[root_node_id] != _tree.TREE_LEAF:
        paths_left = [[(root_node_id, '-')] + path
                 for path in all_tree_signed_paths(dtree, children_left[root_node_id])]
        paths_right = [[(root_node_id, '+')] + path
                 for path in all_tree_signed_paths(dtree, children_right[root_node_id])]
        paths = paths_left + paths_right
    else:
        paths = [[(root_node_id, )]]
    return paths
    
def all_tree_paths(dtree, root_node_id=0):
    """ Get all the individual tree paths from root node to the leaves
    for a decision tree.

    Parameters
    ----------
    dtree : Decision Tree
        An individual decision tree from scikit-learn.
    root_node_id : int, default=0
        The index of the root node of the tree. Default 0 consideres the entire tree,
        while other values would consider a subtree or lead to an error.

    Returns
    -------
    list of lists
        One list for each path, containing ids of its nodes starting at the root and
        ending at the leaf.
    """
    # Use these lists to parse the tree structure
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right

    if root_node_id is None:
        paths = []

    if root_node_id == _tree.TREE_LEAF:
        raise ValueError(f"Invalid node_id {_tree.TREE_LEAF}")

    # if left/right is None we'll get empty list anyway
    if children_left[root_node_id] != _tree.TREE_LEAF:
        paths = [[root_node_id] + path
                 for path in all_tree_paths(dtree, children_left[root_node_id]) +
                 all_tree_paths(dtree, children_right[root_node_id])]

    else:
        paths = [[root_node_id]]
    return paths

def get_path_prevalence(
        rf,
        impurity_decrease_threshold,
        testpoint,
        min_weight=0.01,
        signed=True,
        mask=None,
    ):
    """ Compute the prevalent interactions and their prevalence for a testpoint.
    
    We use FP-Growth to find a series of interactions with their weighted prevalence,
    based on the paths passed through by the testpoint.
    
    Parameters
    ----------
    rf : Random Forest
        The random forest model from scikit-learn.
    impurity_decrease_threshold : float
        If a split results in a decrease smaller than this parameter, then it
        will not be considered in the path.
    testpoint : array
        Covariates of the point, for which paths should be considered.
    min_weight : float, default=0.01,
        The minimum weighted prevalence a interaction must have to be considered.
    signed : bool, default=True
        Whether interactions should be considered with signs.
    mask : dict, optional
        If provided, stores the name of each feature. Features with the same name are
        treated as the same.
    
    Returns
    -------
    OrderedDict
        Dictionary, where keys correspond to interactions and the values are
        their prevalence.
    """
    feature_paths, weight = filtered_point_paths(
        rf,
        impurity_decrease_threshold,
        testpoint,
        signed=signed,
    )
    feature_paths = [list(path) for path in feature_paths]
    if mask is not None:
        if signed:
            feature_paths = [sorted([(mask[elem[0]], elem[1]) for elem in x]) for x in feature_paths]
        else:
            feature_paths = [sorted([mask[elem] for elem in x]) for x in feature_paths]
    patterns = find_frequent_patterns(feature_paths, weight, min_weight)
    prevalence = OrderedDict(
        sorted(
            map(lambda x: (frozenset(x[0]), x[1]), patterns.items()),
            key=lambda t: -t[1]
        ),
    )
    return prevalence

def filtered_point_paths(dtree_or_rf, threshold, testpoint, signed=True):
    """ Get the set of feature paths and their weights filtered by
    the impurity decrease
    """
    if hasattr(dtree_or_rf, 'tree_'):
        impurity_decrease = compute_impurity_decrease(dtree_or_rf)
        features = dtree_or_rf.tree_.feature
        filtered = [x >= threshold for x in impurity_decrease]
        if signed:
            node = 0
            path = []
            while node != _tree.TREE_LEAF:
                feature = features[node]
                if testpoint[feature] <= dtree_or_rf.tree_.threshold[node]:
                    if filtered[node] and feature not in map(lambda x: x[0], path):
                        path.append((feature, '-'))
                    node = dtree_or_rf.tree_.children_left[node]
                else:
                    if filtered[node] and feature not in map(lambda x: x[0], path):
                        path.append((feature, '+'))
                    node = dtree_or_rf.tree_.children_right[node]
        else:
            node = 0
            path = []
            while node != _tree.TREE_LEAF:
                feature = features[node]
                if filtered[node] and feature not in path:
                    path.append(feature)
                if testpoint[feature] <= dtree_or_rf.tree_.threshold[node]:
                    node = dtree_or_rf.tree_.children_left[node]
                else:
                    node = dtree_or_rf.tree_.children_right[node]
        return [path], [1]
    elif hasattr(dtree_or_rf, 'estimators_'):
        all_fs = []
        all_ws = []
        for tree in dtree_or_rf.estimators_:
            feature_paths, weight = filtered_point_paths(tree, threshold, testpoint, signed)
            all_fs += feature_paths
            all_ws += [w / dtree_or_rf.n_estimators for w in weight]
        return all_fs, all_ws

def get_sample_interactions(
        rf,
        impurity_decrease_threshold,
        testpoints,
        min_weight_dwp=0.01,
        min_weight_pp=0.01,
        weight_scheme="depth",
        signed=True,
        mask=None,
        max_interaction_size=6,
        combining_outputs='intersection',
    ):
    """Find interactions, which are active at the respective testpoints.

    Parameters
    ----------
    rf : Random Forest
        The random forest model from scikit-learn.
    impurity_decrease_threshold : float
        Minimum impurity decrease for each split to be considered.
    testpoints : iterable of arrays
        List with covariates for each considered point.
    min_weight_dwp : float, default=0.01
        Minimum value to be obtained for an interaction to be considered according to
        adjusted DWP.
    min_weight_pp : float, default=0.01
        Minimum value to be obtained for an interaction to be considered according to
        path prevalence.
    weight_scheme : str, default="depth"
        Weight scheme to be used by `get_prevalent_interactions`.
    signed : bool, default=True
        Whether interactions should be considered with signs.
    mask : dict, optional
        If provided, stores the name of each feature. Features with the same name are
        treated as the same.
    max_interaction_size : int, default=6
        Largest size for interactions to be considered.
    combining_outputs : 'intersection' or 'union', default='intersection'
        How to combine the results of DWP and Path Prevalence. For 'union' a default value
        of 0.0 will be used for interactions, which where in only one of the two results.

    Returns
    -------
    list of OrderedDict
        For each point in testpoints, the interactions which were found sorted by the
        product of the prevalences.
    """
    if combining_outputs not in ['intersection', 'union']:
        ValueError(f"Combination method {combining_outputs} not allowed.")
    prevalent_interact = get_prevalent_interactions(rf, impurity_decrease_threshold, min_weight_dwp, weight_scheme, signed, mask, max_interaction_size)
    outputs = []
    for testpoint in testpoints:
        path_prevalence = get_path_prevalence(rf, impurity_decrease_threshold, testpoint, min_weight_pp, signed, mask)
        if combining_outputs == 'intersection':
            outputs.append(OrderedDict(sorted(
                [(key, (prevalent_interact[key], path_prevalence[key])) for key in prevalent_interact.keys() & path_prevalence.keys()],
                key=lambda x: -x[1][0] * x[1][1]
            )))
        elif combining_outputs == 'union':
            outputs.append(OrderedDict(sorted(
                [(key, (prevalent_interact.get(key, 0.), path_prevalence.get(key, 0.))) for key in prevalent_interact.keys() | path_prevalence.keys()],
                key=lambda x: -x[1][0] * x[1][1]
            )))
    return outputs
