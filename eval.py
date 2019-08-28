#!/usr/bin/env python3

import sys, os
from os import path
import shutil
import cv2 as cv
import numpy as np
import pandas as pd

# pixel accumulator for different classes
def pixel_accumulator():
    return pd.DataFrame([[0]*8], columns = [
            'err_bg', 'px_bg', 'err_bg_re', 'px_bg_re',
            'err_fg', 'px_fg', 'err_fg_re', 'px_fg_re',
        ])

log_colors = [
        [ 0,     np.array([ 49,  54, 149], dtype = np.uint8) ],
        [ 2**-4, np.array([ 69, 117, 180], dtype = np.uint8) ],
        [ 2**-3, np.array([116, 173, 209], dtype = np.uint8) ],
        [ 2**-2, np.array([171, 217, 233], dtype = np.uint8) ],
        [ 2**-1, np.array([224, 243, 248], dtype = np.uint8) ],
        [ 2** 0, np.array([254, 224, 144], dtype = np.uint8) ],
        [ 2** 1, np.array([253, 174,  97], dtype = np.uint8) ],
        [ 2** 2, np.array([244, 109,  67], dtype = np.uint8) ],
        [ 2** 3, np.array([215,  48,  39], dtype = np.uint8) ],
        [ 2** 4, np.array([165,   0,  38], dtype = np.uint8) ]
        ]

num_test_images = 200
num_error_images = 20
abs_error_threshold = 3#px
rel_error_threshold = 0.05

def eval(name):
    results_dir = 'results/{}/'.format(name)
    for fldr in ['errors_disp_noc_0', 'errors_disp_occ_0', 'errors_disp_img_0', 'result_disp_img_0']:
        os.makedirs(results_dir+fldr, exist_ok = True)
    # _tp for template
    obj_map_tp = 'data/scene_flow/obj_map/{:0>6}_10.png'
    #  disp_gt_tp = 'data/scene_flow/disp_noc_0/{:0>6}_10.png'
    #  run_on(disp_gt_tp, obj_map_tp, results_dir)
    disp_gt_tp = 'data/scene_flow/disp_occ_0/{:0>6}_10.png'
    run_on(disp_gt_tp, obj_map_tp, results_dir)

def run_on (
        obj_map_dir,
        disp_gt_tp,
        disp_tp,
        err_img_tp,
        err_stats_tp = ""
        ):
    accumulator = pixel_accumulator()
    for index in range(num_test_images):
        obj_map = cv.imread(obj_map_tp.format(i), cv.IMREAD_ANYDEPTH)
        disp_gt = cv.imread(ground_truth_tp.format(i), cv.IMREAD_ANYDEPTH)
        disp = cv.imread(ground_truth_tp.format(i), cv.IMREAD_ANYDEPTH)
        if disp.dtype != np.uint16:
            disp = disp.astype(np.uint16) * 256
        errors, err_img = evaluate( obj_map, disp_gt, disp )
        if index < num_error_images:
            cv.imwrite(err_img, err_img_tp.format(i))
        if err_stats_tp != "":
            log_stats(err_stats_tp.format(i), errors)
        #TODO add `errors` to `accumulator`
        #TODO add `err^2` to `acc2` (for the STD)
    log_stats(results_dir+'stats_disp_occ_0.txt', accumulator)

def evaluate( obj_map, disp_gt, disp ):
    errors = pixel_accumulator()
    results_mask = disp > 0
    bg_mask = disp_gt > 0 & obj_map == 0
    fg_mask = disp_gt > 0 & obj_map > 0
    disp = interpolate(disp)
    abs_err = np.abs(disp - disp_gt)
    rel_err = abs_err / disp_gt # division by zero yields infinity
    wrong_px = (abs_err > abs_error_threshold) | (rel_err > rel_error_threshold)
    errors['px_bg'] = np.sum( bg_mask )
    errors['err_bg'] = np.sum( bg_mask & wrong_px )
    errors['px_bg_re'] = np.sum( bg_mask & results_mask )
    errors['err_bg_re'] = np.sum( bg_mask & results_mask & wrong_px )
    errors['px_fg'] = np.sum( fg_mask )
    errors['err_fg'] = np.sum( fg_mask & wrong_px )
    errors['px_fg_re'] = np.sum( fg_mask & results_mask )
    errors['err_fg_re'] = np.sum( fg_mask & results_mask & wrong_px )
    err = np.minimum( abs_err / 3.0, rel_err * 20.0 )
    # ^^^ TODO: why min and not max?
    err_img = np.zeros((*disp.shape, 3), dtype = np.uint8)
    for threshold, color in log_colors:
        err_img[ err > threshold ] = color
    err_img[ results_mask == False ] /= 2
    err_img[ disp_gt == 0 ] *= 0
    return errors, err_img

def interpolate(disp):
    # TODO
    return disp.copy()

def log_stats(fn, stats, with_headers = True):
    with open(fn, 'w') as f:
        f.write(stats.to_string(header = False, index = False))
        # TODO

if __name__ == '__main__':
    assert len(sys.argv) == 2
    name = sys.argv[1]
    success = eval(name)
    sys.exit(success)
