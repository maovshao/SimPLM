import math
import warnings
import numpy as np
import scipy.sparse as ssp
from pathlib import Path
from sklearn.metrics import average_precision_score as aupr
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as area_under_curve

from itertools import chain

from downstream.util import get_pid_go, get_pid_go_sc, get_pid_go_mat, get_pid_go_sc_mat

__all__ = ['fmax', 'aupr', 'pair_aupr', 'smin', 'ROOT_GO_TERMS']
#ROOT_GO_TERMS = set()
ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}

def fmax(targets: ssp.csr_matrix, scores: np.ndarray):
    fmax_ = 0.0, 0.0
    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, r = correct / cut_sc.sum(axis=1), correct / targets.sum(axis=1)
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
        if np.isnan(p):
            continue
        try:
            fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, cut))
        except ZeroDivisionError:
            pass
    return fmax_


def pair_aupr(targets: ssp.csr_matrix, scores: np.ndarray, top=200):
    scores[np.arange(scores.shape[0])[:, None],
           scores.argpartition(scores.shape[1] - top)[:, :-top]] = -1e100
    return aupr(targets.toarray().flatten(), scores.flatten())

def smin(pro_anno, term_scores, type_list, ic_file):
    ic = {}
    with open(ic_file) as fp:
        for line in fp:
            go_term, ic_score = line.split()
            ic[go_term] = float(ic_score)

    smin = float('inf'), 1.0
    for cut in (c / 100 for c in range(101)):
        num, ru, mi = 0, 0.0, 0.0
        for seq in type_list:
            if seq in pro_anno and len(pro_anno[seq]) > 1:
                num += 1
                for func in pro_anno[seq]:
                    if seq not in term_scores or func not in term_scores[seq] or term_scores[seq][func] < cut:
                        ru += ic.get(func, 0)
                if seq in term_scores:
                    for func in term_scores[seq]:
                        if term_scores[seq][func] >= cut and func not in pro_anno[seq]:
                            mi += ic.get(func, 0)
        smin = min(smin, (math.sqrt(ru * ru + mi * mi) / num, cut))
    return smin[0]

def get_y(pro_anno, term_scores, pid_list, label_list):
    y_true, y_pred = [], []
    for pid in pid_list:
        if pid not in pro_anno or len(pro_anno[pid]) <= 1:
            continue
        for go_term in label_list:
            y_true.append(1.0 if go_term in pro_anno[pid] else 0.0)
            y_pred.append(term_scores[pid][go_term] if pid in term_scores and go_term in term_scores[pid] else 0.0)
    return y_true, y_pred

def term_aupr(pro_anno, term_scores, pid_list, label_list, res_file=None):
    if res_file is not None:
        res_file = open(res_file, 'w')
    auprs = []
    for label in label_list:
        y_true, y_pred = get_y(pro_anno, term_scores, pid_list, {label})
        if sum(y_true) >= 3:
            aupr = 0
            if len(y_pred) - y_pred.count(0) >= 3:
                p, r, _ = precision_recall_curve(y_true, y_pred)
                aupr = area_under_curve(r, p)
            auprs.append(aupr)
            if res_file is not None:
                print(label, aupr, file=res_file)
    res = sum(auprs) / len(auprs) if len(auprs) > 0 else 0.0
    if res_file is not None:
        res_file.close()
    return res

def evaluate_metrics(pid_go, pid_go_sc, ic_file = None, if_m_aupr = True):
    if (if_m_aupr == True):
        maupr_path = ''.join([x+'/' for x in pid_go_sc.split('/')[:-1]]) + 'maupr/'
        maupr_res_file = maupr_path + pid_go_sc.split('/')[-1].split('.txt')[0] + '_maupr'
        maupr_path = Path(maupr_path)
        maupr_path.mkdir(parents=True, exist_ok=True)
    pid_go_sc, pid_go = get_pid_go_sc(pid_go_sc), get_pid_go(pid_go)
    pid_list = list(pid_go.keys())
    go_list = sorted(set(list(chain(*([pid_go[p_] for p_ in pid_list] +
                                      [pid_go_sc[p_] for p_ in pid_list if p_ in pid_go_sc])))) - ROOT_GO_TERMS)
    go_mat, score_mat = get_pid_go_mat(pid_go, pid_list, go_list), get_pid_go_sc_mat(pid_go_sc, pid_list, go_list)
    smin_ = None
    m_aupr_ = None
    if (ic_file != None):
        smin_ = smin(pid_go, pid_go_sc, pid_list, ic_file)
    if (if_m_aupr == True):
        m_aupr_ = term_aupr(pid_go, pid_go_sc, pid_list, go_list, maupr_res_file)
    return fmax(go_mat, score_mat), pair_aupr(go_mat, score_mat), smin_, m_aupr_
    