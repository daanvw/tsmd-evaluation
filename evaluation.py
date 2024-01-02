import numpy as np
import time
import pandas as pd
import scipy.optimize
import sklearn.metrics as skmetrics

# _SUPPORTED_METRICS = ["F1", "Score"]

def evaluate(gt, discovered_sets):
    mm = match_matrix(discovered_sets, gt)
    return f1_score(mm)
    

def overlap_union_ratio(s, e, s_gt, e_gt):
    return max(0, (min(e, e_gt) - max(s, s_gt)) / (max(e, e_gt) - min(s, s_gt)))


def match_matrix(gt, discovered_sets, threshold=0.5):
    assert (threshold >= 0.5)
    assert gt
        
    n, m = len(gt), len(discovered_sets)
    column_names = np.arange(1, m+1)
    if type(gt) is list:
        row_names = np.arange(1, n+1)
        gt_sets   = gt
    elif type(gt) is dict:
        row_names = np.array(list(gt.keys())) 
        gt_sets   = list(gt.values())
    
    if not discovered_sets:
        # only return missed discovery column
        match_matrix = np.zeros((n + 1, 1), dtype=int)
        match_matrix[:-1, 0] = [len(gt_set) for gt_set in gt_sets]
        match_matrix[-1, 0]  = np.nan
        return match_matrix, row_names, []
    
    # inner match matrix
    mm = np.zeros((n, m), dtype=int)
    for i, gt_set in enumerate(gt_sets):
        for (s_gt, e_gt) in gt_set:
            # find the best match in any motif set, greater than threshold
            best     = None
            best_our = 0.0 
            for j, discovered_set in enumerate(discovered_sets):
                for (s, e) in discovered_set:
                    our = overlap_union_ratio(s, e, s_gt, e_gt)
                    if our > threshold and our > best_our:
                        best = j
                        best_our = our
            
            if best is not None:
                mm[i, best] += 1
    
    # r and c are of length min(n, m)
    r, c = scipy.optimize.linear_sum_assignment(mm, maximize=True)
    
    r_perm = np.hstack((r, np.setdiff1d(range(n), r))) 
    c_perm = np.hstack((c, np.setdiff1d(range(m), c)))
    
    # compute number of missed detections
    md = np.array([len(gt_set) for gt_set in gt_sets])
    md = md - np.sum(mm, axis=1)
    
    # compute number of false discoveries 
    fd = np.array([len(discovered_set) for discovered_set in discovered_sets])
    fd = fd - np.sum(mm, axis=0)
    
    # permute M
    mm = mm[:, c_perm]
    mm = mm[r_perm, :]
    
    # permute false discoveries and missed detections
    fd = fd[c_perm]
    md = md[r_perm]
    
    md = np.expand_dims(md, axis=0).T
    match_matrix = np.block([
        [mm, md],
        [fd, 0 ] 
    ])
    return match_matrix, row_names[r_perm], column_names[c_perm]

def recall(match_matrix):
    n, m = match_matrix.shape[0] - 1, match_matrix.shape[1] - 1
    diag = np.diag(match_matrix[:min(n, m),:min(n, m)])
    return np.sum(diag) / np.sum(match_matrix[:n, :])

def precision(match_matrix):
    n, m = match_matrix.shape[0] - 1, match_matrix.shape[1] - 1
    # no motif sets found
    if m == 0:
        return 0.0
    diag = np.diag(match_matrix[:min(n, m),:min(n, m)])
    return np.sum(diag) / np.sum(match_matrix[:, :min(n, m)])

def f1_score(match_matrix):
    prec = precision(match_matrix)
    rec  = recall(match_matrix)
    if not(prec == 0 and rec == 0):
        return 2 * rec * prec / (prec + rec)
    else:
        return 0

def pretty_print_match_matrix(match_matrix, row_names, col_names):
    from tabulate import tabulate
    table, row_names, col_names = match_matrix.tolist(), row_names.tolist(), col_names.tolist()
    table.insert(0, [""] + col_names + ["MD"])
    
    for i, row_name in enumerate(row_names):
        table[i+1].insert(0, row_name)
        
    table[-1].insert(0, "FD")
    table[-1][-1] = "-"
    return tabulate(table, [], tablefmt="grid")

def score_metric(discovered_sets, gt_sets, l):
    kappa, kappa_gt = len(discovered_sets), len(gt_sets)
    kappa_max = max(kappa, kappa_gt)
    
    matrix = np.zeros((kappa_max, kappa_max))
    for i in range(kappa_max):
        for j in range(kappa_max):
            gt_set = gt_sets[i] if i < kappa_gt else []
            discovered_set = discovered_sets[j] if j < kappa else [] 
            matrix[i, j] = optimal_score(gt_set, discovered_set, l)
            
    # comment these lines if you want additional discovered motif sets to be penalized
    # if kappa > kappa_gt:
        # matrix = matrix[:, :kappa_gt]
    
    r, c = scipy.optimize.linear_sum_assignment(matrix, maximize=False)
    score = np.sum(matrix[r, c])
    return score
    

def optimal_score(discovered_set, gt_set, l):    
    k, k_gt = len(discovered_set), len(gt_set)
    
    matrix = np.full((k_gt + k, k + k_gt), np.inf)
    for i in range(k_gt):
        (s_gt, e_gt) = gt_set[i]
        for j in range(k):
            (s, e) = discovered_set[j]
            # if overlapping
            if s_gt < e and s < e_gt:
                matrix[i, j] = abs(s_gt - s) 
        
    matrix[:k_gt, k:] = l
    matrix[k_gt:, :k] = l
    matrix[k_gt:, k:] = 0
            
    r, c = scipy.optimize.linear_sum_assignment(matrix, maximize=False)
    score = np.sum(matrix[r, c])
    return score