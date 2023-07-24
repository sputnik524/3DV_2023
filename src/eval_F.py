# select the data
import numpy as np
# import matplotlib.pyplot as plt
from utils_rlsac import *
from tqdm import tqdm
import os, time
from metrics import *

import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def evaluate_results(submission, split = 'val', all=False):
    ang_errors = {}
    r_errors = {}
    t_errors = {}
    DIR = split
    seqs = os.listdir(DIR)
    for seq in seqs:
        matches = load_h5(f'{DIR}/{seq}/matches.h5')
        K1_K2 = load_h5(f'{DIR}/{seq}/K1_K2.h5')
        R = load_h5(f'{DIR}/{seq}/R.h5')
        T = load_h5(f'{DIR}/{seq}/T.h5')
        F_pred, inl_mask = submission[0][seq], submission[1][seq]
        ang_errors[seq] = {}
        r_errors[seq] = {}
        t_errors[seq] = {}
        total_cnt = 0
        valid_cnt = 0
        for k, m in tqdm(matches.items()):
            total_cnt += 1
            if k not in F_pred:
                continue
            valid_cnt += 1

            if F_pred[k] is None:
                ang_errors[seq][k] = 3.14
                r_errors[seq][k] = 3.14
                t_errors[seq][k] = 3.14
                continue
            img_id1 = k.split('-')[0]
            img_id2 = k.split('-')[1]
            K1 = K1_K2[k][0][0]
            K2 = K1_K2[k][0][1]
            try:
                E_cv_from_F = get_E_from_F(F_pred[k], K1, K2)
            except:
                print ("Fail")
                E = np.eye(3)
            R1 = R[img_id1]
            R2 = R[img_id2]
            T1 = T[img_id1]
            T2 = T[img_id2]
            dR = np.dot(R2, R1.T)
            dT = T2 - np.dot(dR, T1)
            if all==False:
                pts1 = inl_mask[k][:,:2] # coordinates in image 1
                pts2 = inl_mask[k][:,2:4]  # coordinates in image 2
            else:
                pts1 = m[inl_mask[k],:2] # coordinates in image 1
                pts2 = m[inl_mask[k],2:]  # coordinates in image 2
            p1n = normalize_keypoints(pts1, K1)
            p2n = normalize_keypoints(pts2, K2)
            err_q, err_t = eval_essential_matrix(p1n, p2n, E_cv_from_F, dR, dT)
            ang_errors[seq][k] = max(err_q, err_t)
            r_errors[seq][k] = err_q
            t_errors[seq][k] = err_t
        print(f'{seq} total: {total_cnt}, valid: {valid_cnt}')
    return ang_errors, r_errors, t_errors

def f_error_mask(pts1, pts2, F, threshold=0.001):
    """Compute multiple evaluaton measures for a fundamental matrix.

    Return (False, 0, 0, 0) if the evaluation fails due to not finding inliers for the ground truth model, 
    else return() True, F1 score, % inliers, mean epipolar error of inliers).

    Follows the evaluation procedure in:
    "Deep Fundamental Matrix Estimation"
    Ranftl and Koltun
    ECCV 2018

    Keyword arguments:
    pts1 -- 3D numpy array containing the feature coordinates in image 1, dim 1: x and y coordinate, dim 2: correspondences, dim 3: dummy dimension
    pts2 -- 3D numpy array containing the feature coordinates in image 2, dim 1: x and y coordinate, dim 2: correspondences, dim 3: dummy dimension
    F -- 2D numpy array containing an estimated fundamental matrix
    gt_F -- 2D numpy array containing the corresponding ground truth fundamental matrix
    threshold -- inlier threshold for the epipolar error in pixels
    """
    EPS = 0.00000000001
    num_pts = pts1.shape[1]

    # 2D coordinates to 3D homogeneous coordinates
    hom_pts1 = np.concatenate((pts1, np.ones((1, num_pts))), axis=0)
    hom_pts2 = np.concatenate((pts2, np.ones((1, num_pts))), axis=0)

    def epipolar_error(hom_pts1, hom_pts2, F):
        """Compute the symmetric epipolar error."""
        res  = 1 / (np.linalg.norm(F.T.dot(hom_pts2)[0:2], axis=0)+EPS)
        res += 1 / (np.linalg.norm(F.dot(hom_pts1)[0:2], axis=0)+EPS)
        res *= abs(np.sum(hom_pts2 * np.matmul(F, hom_pts1), axis=0))
        return res

    # determine inliers based on the epipolar error
    est_res = epipolar_error(hom_pts1, hom_pts2, F)

    inlier_mask = (est_res < threshold)

    return inlier_mask

