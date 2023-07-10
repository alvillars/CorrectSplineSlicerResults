from typing import List, Optional, Tuple
import math

import numpy as np
from napari.layers import Layer, Image
from napari.types import LayerDataTuple, ImageData

from skimage.transform import rotate

import scipy
import matplotlib.pyplot as plt


from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from geomdl import exchange, operations
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from skimage.measure import label, regionprops

import splineslicer 

import matplotlib.pyplot as plt
import seaborn as sns

"""This set of functions are made by Kevin Yamauchi and modified by Alexis Villars 
for the image analysis pipeline made for the neural tube segmentation"""
class BoundaryModes(Enum):
    DERIVATIVE = 'derivative'
    PERCENT_MAX = 'percent_max'
    TANH = 'tanh'

def calculate_slice_rotations(im_stack: np.ndarray, im_stack_chan: np.ndarray, max_rotation:float = 45, p=0.5) -> List[float]:
    """Calculate the rotation angle to align each slice so the
    objects long axis is aligned with the horizontal axis.

    Parameters
    ----------
    im_stack : np.ndarray
        A stack of images. The images should be binary
        or label iamges. The regions are found and processed
        with the scikit-image label and regionprops functions.
        The stack should have shape (z, y, x) for z images
        with shape (y, x).
    max_rotation : float
        The maximum allowed rotation between slices in degrees.
        If this value is exceeded, it is assumed that the
        opposite rotation was found and 180 is added to the rotation.
        The default value is 45.

    Returns
    -------
    rotations : List[float]
        The rotation for each slice in degrees.
    """
    # get the rotations of the images
    rotations = []
    rotations_raw = []
    prev_rot = 0
    previous_values = []
    line_scans = []
    orientations = []
    pos = []
    adapted_rotations = []
    for i, im in enumerate(im_stack):
        previous_values.append(prev_rot)
        rp = regionprops(im.astype(int))
        if len(rp) > 0:
            orientation = rp[0].orientation
            y0, x0 = rp[0].centroid

            # compute boundary points of the NT 
            x1 = x0 - math.sin(orientation) * p * rp[0].axis_major_length
            y1 = y0 - math.cos(orientation) * p * rp[0].axis_major_length
            x2 = x0 + math.sin(orientation) * p * rp[0].axis_major_length
            y2 = y0 + math.cos(orientation) * p * rp[0].axis_major_length

            length = int(np.hypot(x1-x2, y1-y2))
            x, y = np.linspace(x1, x2, length), np.linspace(y1, y2, length)
            pos.append([x1,y1,x2,y2])
            # Extract the values along the line
            linescan = im_stack_chan[i, y.astype(int), x.astype(int)]
            line_scans.append(linescan)
            orientations.append(orientation)
            angle = orientation * (180 / np.pi)

            if linescan[0]:
                if angle <0:
                    adapted_angle = angle+90
                else:
                    adapted_angle = angle+90
            else:
                if angle <0:
                    adapted_angle = angle-90
                else:
                    adapted_angle = angle-90
    
            angle_in_degrees = orientation * (180 / np.pi) + 90
            adapted_rotations.append(adapted_angle)

        else:
            angle_in_degrees = 0

        rotations_raw.append(angle_in_degrees)

        if i > 0:
            # check if we should flip the rotation
            if abs(prev_rot - angle_in_degrees) > max_rotation:
                angle_in_degrees = -1 * (180 - angle_in_degrees)

        prev_rot = angle_in_degrees

        rotations.append(angle_in_degrees+360)

    # method to check for switches
    check_angles = abs(np.diff(adapted_rotations)) # get the diff to get 
    a = np.where(check_angles > 90, 1, 0) # find the abrupt changes in the differential of angles
    switches = a[:-1]+a[1:] # sum consequent indices to find changes which are sudden and not maitained
    switching_ind = np.asarray(np.where(switches == 2)[0]+1) # these flipped would be equal to 2, these are the flipped indices
    if len(switching_ind)>0:
        np.array(adapted_rotations)[switching_ind] = np.array(adapted_rotations)[switching_ind]+180

    return rotations, line_scans, orientations, pos, np.asarray(adapted_rotations)

def align(at, to_align):
    delta_x, delta_y = at
    shifted_im = np.zeros(np.shape(to_align[:, :]))
    if delta_x >0 and delta_y>0:
        shifted_im[delta_x:,delta_y:] = to_align[0:-delta_x, 0:-delta_y]
    elif delta_x <0 and delta_y<0:
        shifted_im[:delta_x,:delta_y] = to_align[-delta_x:, -delta_y:]
    elif delta_x ==0 and delta_y<0:
        shifted_im[:,:delta_y] = to_align[:, -delta_y:]
    elif delta_x <0 and delta_y==0:
        shifted_im[:delta_x,:] = to_align[-delta_x:, :]
    elif delta_x ==0 and delta_y>0:
        shifted_im[:,delta_y:] = to_align[:, 0:-delta_y]
    elif delta_x >0 and delta_y==0:
        shifted_im[delta_x:,:] = to_align[0:-delta_x, :]
    elif delta_x >0 and delta_y<0:
        shifted_im[delta_x:,:delta_y] = to_align[0:-delta_x, -delta_y:]
    elif delta_x <0 and delta_y>0:
        shifted_im[:delta_x,delta_y:] = to_align[-delta_x:, 0:-delta_y]

    return shifted_im

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def method_1(raw_slice, nt_seg_slice, half_width):

    nt_label_im = label(nt_seg_slice)
    rp = regionprops(nt_label_im)

    nt_centroid = rp[0].centroid
    nt_bbox = rp[0].bbox

    # get the crop region
    row_min = int(nt_centroid[0] - half_width)
    row_max = int(nt_centroid[0] + half_width + 1)
    col_min = int(nt_bbox[1])
    col_max = int(nt_bbox[3])

    raw_crop = raw_slice[nt_bbox[0]:nt_bbox[2], nt_bbox[1]:nt_bbox[3]]
    sample_region = raw_slice[row_min:row_max, col_min:col_max]
    summed_profile = sample_region.sum(axis=0)

    # correct for columns with no nt labels
    n_sample_rows = nt_seg_slice[row_min:row_max, col_min:col_max].sum(axis=0)
    no_nt_indices = np.argwhere(n_sample_rows == 0)
    n_sample_rows[no_nt_indices] = 1
    summed_profile[no_nt_indices] = 0

    # normalize the intensity by the number of nt pixels in the column
    raw_profile = summed_profile / n_sample_rows
    return raw_profile, raw_crop, col_min, col_max

def method_2(raw_slice, nt_seg_slice, half_width):
    intermediate_im = nt_seg_slice*raw_slice
    nt_label_im = label(nt_seg_slice)
    rp = regionprops(nt_label_im)

    nt_centroid = rp[0].centroid
    nt_bbox = rp[0].bbox

    restrict_width = half_width*(rp[0].axis_minor_length/2)
    # get the crop region
    row_min = int(nt_centroid[0] - restrict_width)
    row_max = int(nt_centroid[0] + restrict_width + 1)
    col_min = int(nt_bbox[1])
    col_max = int(nt_bbox[3])

    raw_crop = intermediate_im[nt_bbox[0]:nt_bbox[2], nt_bbox[1]:nt_bbox[3]]
    sample_region = intermediate_im[row_min:row_max, col_min:col_max]

    summed_profile = sample_region.sum(axis=0)


    # correct for columns with no nt labels
    n_sample_rows = nt_seg_slice[row_min:row_max, col_min:col_max].sum(axis=0)
    no_nt_indices = np.argwhere(n_sample_rows == 0)
    n_sample_rows[no_nt_indices] = 1
    summed_profile[no_nt_indices] = 0

    # normalize the intensity by the number of nt pixels in the column
    raw_profile = summed_profile / n_sample_rows
    return raw_profile, raw_crop, col_min, col_max
    
def find_boundaries(
        seg_im: np.ndarray,
        stain_im: np.ndarray,
        half_width: int = 10,
        bg_sample_pos: float = 0.7,
        bg_half_width: int = 2,
        edge_method: BoundaryModes = BoundaryModes.DERIVATIVE,
        edge_value: float = 0.1
):
    ventral_boundary = []
    dorsal_boundary = []
    nt_length = []
    nt_col_start = []
    nt_col_end = []
    bg_sub_profiles = []
    cropped_ims = []
    raw_profiles = []
    for nt_seg_slice, raw_slice in zip(seg_im, stain_im):

        # normalize the intensity by the number of nt pixels in the column
        raw_profile, raw_crop, col_min, col_max = method_2(raw_slice, nt_seg_slice, half_width, )

        bg_sample_center = int(bg_sample_pos * (col_max - col_min))
        bg_sample_min = bg_sample_center - bg_half_width
        bg_sample_max = bg_sample_center + bg_half_width
        bg_value = np.mean(raw_profile[bg_sample_min:bg_sample_max])
        bg_sub_profile = (raw_profile - bg_value).clip(min=0, max=1)

        if isinstance(edge_method, str):
            edge_method = BoundaryModes(edge_method)
        if edge_method == BoundaryModes.DERIVATIVE:
            rising_edge, falling_edge = splineslicer.measure.measure_boundaries.find_edge_derivative(bg_sub_profile)
        elif edge_method == BoundaryModes.PERCENT_MAX:
            rising_edge, falling_edge = splineslicer.measure.measure_boundaries.find_edge_percent_max(bg_sub_profile, threshold=edge_value)
        elif edge_method == BoundaryModes.TANH:
            rising_edge, falling_edge = splineslicer.measure.measure_boundaries.find_edge_fit(bg_sub_profile, threshold=edge_value)
        else:
            raise ValueError('unknown boundary mode')

        # store the values
        nt_length.append(col_max - col_min)
        ventral_boundary.append(rising_edge)
        dorsal_boundary.append(falling_edge)
        cropped_ims.append(raw_crop)
        bg_sub_profiles.append(bg_sub_profile)
        raw_profiles.append(raw_profile)
        nt_col_start.append(col_min)
        nt_col_end.append(col_max - 1)

    return nt_length, ventral_boundary, dorsal_boundary, bg_sub_profiles, raw_profiles, cropped_ims, nt_col_start, nt_col_end

def find_edge_percent_max(bg_sub_profile: np.ndarray, threshold: float) -> Tuple[int, int]:
    # find the peak index
    peaks, _ = find_peaks(bg_sub_profile, distance=10, width=2)
    if len(peaks > 0):
        peak_index = np.min(peaks)

        # find the peak value
        peak_value = bg_sub_profile[peak_index]

        # calculate the threshold value
        threshold_value = threshold * peak_value

        # find rising edge
        rising_edge = splineslicer.measure.measure_boundaries._detect_threshold_crossing(
            bg_sub_profile,
            threshold=threshold_value,
            iloc=peak_index,
            increment=-1
        )

        # find trailing edge
        falling_edge = splineslicer.measure.measure_boundaries._detect_threshold_crossing(
            bg_sub_profile,
            threshold=threshold_value,
            iloc=peak_index,
            increment=1
        )
    else:
        rising_edge = np.nan
        falling_edge = np.nan

    return rising_edge, falling_edge

def find_boundaries_method2( 
        seg_im: np.ndarray,
        stain_im: np.ndarray,
        half_width: float = 0.5,
        edge_method: BoundaryModes = BoundaryModes.PERCENT_MAX,
        edge_value: float = 0.1,
        upper_range: float = 0.7,
        lower_range: float = 0,
):
    print('edge method from function :', edge_method)
    ventral_boundary = []
    dorsal_boundary = []
    nt_length = []
    nt_col_start = []
    nt_col_end = []
    bg_sub_profiles = []
    cropped_ims = []
    raw_profiles = []

    filtered_im = seg_im*stain_im
    bg_value = np.median(filtered_im[np.nonzero(filtered_im)])

    for nt_seg_slice, raw_slice in zip(seg_im, stain_im):

        # normalize the intensity by the number of nt pixels in the column
        raw_profile, raw_crop, col_min, col_max = method_2(raw_slice, nt_seg_slice, half_width)
        bg_sub_profile = (raw_profile - bg_value).clip(min=0, max=1)

        lim_up = round(upper_range*len(bg_sub_profile))
        lim_down = round(lower_range*len(bg_sub_profile))
        search_profile = bg_sub_profile[lim_down:lim_up]

        if isinstance(edge_method, str):
            edge_method = BoundaryModes(edge_method)
        if edge_method == BoundaryModes.DERIVATIVE:
            rising_edge, falling_edge = splineslicer.measure.measure_boundaries.find_edge_derivative(search_profile)
        elif edge_method == BoundaryModes.PERCENT_MAX:
            print('here')
            rising_edge, falling_edge = find_edge_percent_max(search_profile, threshold=edge_value)
        elif edge_method == BoundaryModes.TANH:
            rising_edge, falling_edge = splineslicer.measure.measure_boundaries.find_edge_fit(search_profile, threshold=edge_value)
        else:
            raise ValueError('unknown boundary mode')

        # store the values
        nt_length.append(col_max - col_min)
        ventral_boundary.append(rising_edge)
        dorsal_boundary.append(falling_edge)
        cropped_ims.append(raw_crop)
        bg_sub_profiles.append(bg_sub_profile)
        raw_profiles.append(raw_profile)
        nt_col_start.append(col_min)
        nt_col_end.append(col_max - 1)

    return nt_length, ventral_boundary, dorsal_boundary, bg_sub_profiles, raw_profiles, cropped_ims, nt_col_start, nt_col_end

def boundary_plot(all_data, xparameter, yparameter, hue_parameter, fsize):
    l = len(all_data.target.unique())
    n = len(np.unique(all_data['somites']))
    fig, axes = plt.subplots(l, n, figsize=fsize)
    
    pallettes  = ['Reds', 'Greens', 'Purples']
    for k, target in enumerate(all_data.target.unique()):
        for i, somite in enumerate(np.unique(all_data['somites'])):
            if n > 1:
                sns.scatterplot(data=all_data.loc[(all_data['somites'] == somite) & (all_data['target'] == target)], 
                                x=xparameter, y=yparameter, ax = axes[k, i], hue=hue_parameter, palette= pallettes[k], edgecolor="gray")
                axes[k, i].set_box_aspect(1)
                axes[k, i].set_xlabel(str(xparameter))
                axes[k, i].set_ylabel(str(yparameter)+'DV position \u03B6')
                axes[k, i].set_title(str(somite)+'ss'+' '+str(target)+' '+str(yparameter))
                axes[k, i].set_ylim([0, 1])
            else:
                sns.scatterplot(data=all_data.loc[(all_data['somites'] == somite) & (all_data['target'] == target)], 
                                x=xparameter, y=yparameter, ax = axes[k], hue=hue_parameter, palette= pallettes[k],  edgecolor="gray")
                axes[k].set_box_aspect(1)
                axes[k].set_xlabel(str(xparameter))
                axes[k].set_ylabel(str(yparameter)+'DV position \u03B6')
                axes[k].set_title(str(somite)+'ss'+' '+str(target)+' '+str(yparameter))
                axes[k].set_ylim([0, 1])
    fig.tight_layout()

def boundary_plot_single_target(all_data, target, xparameter, yparameter, hue_parameter, fsize):
    all_data = all_data.loc[all_data['target'] == target]
    l = len(all_data.target.unique())
    n = len(np.unique(all_data['somites']))
    pallettes  = ['Reds', 'Greens', 'Purples']
    fig, axes = plt.subplots(l, n, figsize=fsize)
    if n >1:
        for i, somite in enumerate(np.unique(all_data['somites'])):  
            sns.scatterplot(data=all_data.loc[(all_data['somites'] == somite)], 
                            x=xparameter, y=yparameter, ax = axes[i], hue=hue_parameter, palette= pallettes[k], edgecolor="gray")
            axes[i].set_box_aspect(1)
            axes[i].set_xlabel('slice_index (a.u)')
            axes[i].set_ylabel('DV position \u03B6')
            axes[i].set_title(str(somite)+'ss'+' '+str(target)+' '+str(yparameter))
            axes[i].set_ylim([0, 1])
    else:
        somite = np.unique(all_data['somites'])[0]
        sns.scatterplot(data=all_data, x=xparameter, y=yparameter, hue=hue_parameter, palette= pallettes[k], edgecolor="gray")
        axes.set_box_aspect(1)
        axes.set_xlabel('slice_index (a.u)')
        axes.set_ylabel('DV position \u03B6')
        axes.set_title(str(somite)+'ss'+' '+str(target)+' '+str(yparameter))
        axes.set_ylim([0, 1])       

    fig.tight_layout() 