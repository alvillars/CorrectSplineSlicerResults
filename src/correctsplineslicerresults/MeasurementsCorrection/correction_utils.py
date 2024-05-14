from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple
from napari.layers import Image
import numpy as np 
from enum import Enum
from geomdl import exchange, operations, BSpline
import pandas as pd
import splineslicer
from skimage.measure import label, regionprops
from scipy.signal import find_peaks

if TYPE_CHECKING:
    import napari

class BoundaryModes(Enum):
    DERIVATIVE = 'derivative'
    PERCENT_MAX = 'percent_max'
    TANH = 'tanh'

def update_metadata(
    image_layer: Image,
    channel_0: str,
    channel_1: str,
    channel_2: str,
    channel_3: Optional[str] = None
):
    if channel_3 is None:
        stain_channel_names = [channel_0, channel_1, channel_2, 'segmentation']
    else: 
        stain_channel_names = [channel_0, channel_1, channel_2, channel_3, 'segmentation']

    image_metadata =  image_layer.metadata
    image_metadata.update({"channel_names": stain_channel_names})
    print(image_metadata)

def measure_boundaries(
        image_layer: Image,
        spline_file_path: str,
        table_output_path: str,
        half_width: float = 0.5,
        edge_method: BoundaryModes = BoundaryModes.PERCENT_MAX,
        edge_value: float = 0.5,
        pixel_size_um: float = 5.79,
        upper_range: float = 1,
        lower_range: float = 0,
        start_slice: int = 0,
        end_slice: int = 99):
    all_cropped_images = []
    channel_data = []
    bg_sub_profiles = []
    raw_profiles = []
    
    im_channels = image_layer.data
    # seg_im= image_layer.data[-1,...]

    targets =  image_layer.metadata.get(
            "channel_names", {}
    )

    # run through the 3 first channels: I should make this automatically adapt to the channel number. 
    # restricts the measurements to the area that are correctly rotated and correclty segmented
    seg_im = np.asarray(im_channels[-1, start_slice:end_slice, ...])
    
    channel_data = []
    bg_sub_profiles = []
    raw_profiles = []

    # run the measurement for each channels except the last one (the segmentation one)
    for k, chan in enumerate(im_channels[:-1]):
        
        # restrict measurement to current channels
        stain_im = np.asarray(im_channels[k, start_slice:end_slice, ...])

        #measure
        
        nt_length, ventral_boundary_px, dorsal_boundary_px, bg_sub_profile, raw_profile, cropped_ims, nt_col_start, nt_col_end = find_boundaries_method2(
                seg_im= seg_im,
                stain_im= stain_im,
                half_width = half_width,
                edge_method = edge_method,
                edge_value = edge_value, 
                upper_range = upper_range,
                lower_range = lower_range
        )
        
        # set the result array
        bg_sub_profiles.append(bg_sub_profile)
        raw_profiles.append(raw_profile)
        nt_length = np.asarray(nt_length)
        nt_length_um = nt_length * pixel_size_um
        ventral_boundary_um = np.asarray(ventral_boundary_px) * pixel_size_um
        dorsal_boundary_um = np.asarray(dorsal_boundary_px) * pixel_size_um
        ventral_boundary_rel = ventral_boundary_um / nt_length_um
        dorsal_boundary_rel = dorsal_boundary_um / nt_length_um

        # get spline for length measurement
        spline = exchange.import_json(spline_file_path)[0]
        spline_length = operations.length_curve(spline) * pixel_size_um
        spline_increment = spline_length / stain_im.shape[0]

        
        n_slices = len(nt_length_um)
        target_name = targets[k]
        target = [target_name] * n_slices
        threshold_list = [edge_value] * n_slices
        slice_index = np.arange(start_slice, end_slice)
        slice_position_um = [i * spline_increment for i in slice_index]
        slice_position_rel_um = [i * spline_increment for i in range(n_slices)]
        param_measure_half_width = [half_width] * n_slices
        param_measure_edge_method = [edge_method] * n_slices
        param_measure_edge_value = [edge_value] * n_slices

        # put into a dataframe to save
        df = pd.DataFrame(
                {
                    'slice_index': slice_index,
                    'slice_position_um': slice_position_um,
                    'slice_position_rel_um': slice_position_rel_um,
                    'nt_length_um': nt_length_um,
                    'target': target,
                    'ventral_boundary_um': ventral_boundary_um,
                    'dorsal_boundary_um': dorsal_boundary_um,
                    'ventral_boundary_rel': ventral_boundary_rel,
                    'dorsal_boundary_rel': dorsal_boundary_rel,
                    'domain_edge_threshold': threshold_list,
                    'nt_start_column_px': nt_col_start,
                    'nt_end_column_px': nt_col_end,
                    'ventral_boundary_px': ventral_boundary_px,
                    'dorsal_boundary_px': dorsal_boundary_px,
                    'param_measure_half_width': param_measure_half_width,
                    'param_measure_edge_method': param_measure_edge_method,
                    'param_measure_edge_value': param_measure_edge_value
                }
            )
        channel_data.append(df)

        # if target_name == 'Olig2':
        #     for j, crop_im in enumerate(cropped_ims):
        #         file_name = row['root_path']+'/profiles/NT_crops/'+row['File'].replace('.h5','_')+target_name+'_slice_'+str(j+start_slice)+'.jpg'
        #         plt.ioff()
        #         fig, axes = plt.subplots()
        #         axes.imshow(crop_im)
        #         ventral = ventral_boundary_px[j]
        #         dorsal = dorsal_boundary_px[j]
        #         axes.plot([ventral,ventral], [0,crop_im.shape[0]], 'orange')
        #         axes.plot([dorsal,dorsal], [0,crop_im.shape[0]], 'green')
        #         plt.savefig(file_name)
        #         # tifffile.imwrite(file_name, ((crop_im/crop_im.max())*255).astype('uint8'))

    # concatenate the data frames for the channels and save
    all_data = pd.concat(channel_data, ignore_index=True)
    all_data.to_csv(table_output_path)

    # here add the saving for the aligned slices

def find_boundaries_method2( 
        seg_im: np.ndarray,
        stain_im: np.ndarray,
        half_width: float = 0.5,
        edge_method: BoundaryModes = BoundaryModes.PERCENT_MAX,
        edge_value: float = 0.1,
        upper_range: float = 1,
        lower_range: float = 0,):
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

def find_edge_percent_max(bg_sub_profile: np.ndarray, threshold: float) -> Tuple[int, int]:
    # find the peak index
    peaks, _ = find_peaks(bg_sub_profile, distance=10)
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
        rising_edge = -1
        falling_edge = -1

    return rising_edge, falling_edge
