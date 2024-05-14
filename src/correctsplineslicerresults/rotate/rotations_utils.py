from typing import Any, Dict, List, Optional, TYPE_CHECKING
from ..NT_splineslicer_functions import calculate_slice_rotations
from napari.layers import Image
import numpy as np 
import splineslicer
import napari

def new_align_rotate(
        mask_layer: Image,
        stain_layer: Image,
        start_slice: Optional[int] = None,
        end_slice: Optional[int] = None,
        NT_segmentation_index: int=0,
        background_index: int=0,
        invert_rotation: bool = False
        ) -> "napari.types.LayerDataTuple":
    
    print(mask_layer.data.shape)
    if start_slice is None:
        start_slice = 0
    if end_slice is None:
        end_slice = mask_layer.data.shape[0]

    rotations, line_scans, orientations, pos, adapted_rotations = calculate_slice_rotations(mask_layer.data[NT_segmentation_index,start_slice:end_slice,...]
                                                                        ,mask_layer.data[background_index,start_slice:end_slice,...], p=0.6)
    
    print(invert_rotation)
    if invert_rotation is True:
        adapted_rotations = np.asarray(adapted_rotations) + 180
    else:
        adapted_rotations = np.asarray(adapted_rotations)

    # rotate the segmentation
    rotated_seg = splineslicer.measure.align_slices.rotate_stack(mask_layer.data[NT_segmentation_index,start_slice:end_slice,...], adapted_rotations)
    rotated_stack_adapted = []
    for i, chan in enumerate(stain_layer.data[:,start_slice:end_slice,...]):
        rotated_stack_adapted.append(splineslicer.measure.align_slices.rotate_stack(np.asarray(chan), adapted_rotations))
    
    # if no NT segmented or multiple segmented elements are found then the slice is not taken and axis 1 (nb of slices)
    # may become <100. needs to account for this 
    m = np.shape(rotated_stack_adapted[0:3])[1]
    rotated_stack_adapted.append(rotated_seg[0:m])

    # transform to np array for saving as h5
    rotated_stack_adapted = np.asarray(rotated_stack_adapted)
    return (rotated_stack_adapted, {'name': 'output_rotation', 'colormap': 'gray'}, 'image')
