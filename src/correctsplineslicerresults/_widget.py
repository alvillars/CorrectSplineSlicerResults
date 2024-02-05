"""
This module has been written by Alexis Villars, mostly based on plugins from Kevin Yamauchi
This is meant as a way to provide correction pluggin for Kevin's splineslicer pluggin
As well as to provide plugin to do the measurements updated with changes I made in Kevin's original function. 
This provides a quick way to test these functions into napari and for other to use during development. 
This piece of code is for development and temporary use and is not meant as a standalone and should be installed with Kevin's splineslicer
"""
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from enum import Enum
import h5py
from magicgui import magicgui
import napari
from napari.layers import Layer, Image
import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget, QPushButton
from skimage.transform import rotate
from superqt.sliders import QLabeledSlider
from skimage.transform import rotate
from .NT_splineslicer_functions import calculate_slice_rotations,  method_2, find_edge_percent_max
# from splineslicer.skeleton.binarize import _binarize_image_mg
import splineslicer
from superqt.collapsible import QCollapsible
from geomdl import exchange, operations
import re
from splineslicer._reader import napari_get_reader
if TYPE_CHECKING:
    import napari

class BoundaryModes(Enum):
    DERIVATIVE = 'derivative'
    PERCENT_MAX = 'percent_max'
    TANH = 'tanh'


class QtImageSliceWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.image_slices: Optional[np.ndarray] = None
        self.results_table: Optional[pd.DataFrame] = None
        self.pixel_size_um: float = 5.79
        self.stain_channeL_names: List[str] = []
        self.min_slice: int = 0
        self.max_slice: int = 0
        self.draw_domain_boundaries = True
        self.current_channel_index: int = 0

        # create the slider
        self.slice_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 99)
        self.slice_slider.setSliderPosition(50)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.setTickInterval(1)
        self.slice_slider.valueChanged.connect(self._on_slider_moved)

        # create the stain channel selection box
        self.image_selector = QComboBox()
        self.image_selector.currentIndexChanged.connect(self._update_current_channel)

        # create the image
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.image_widget = pg.ImageView(parent=self)
        self.nt_ventral_position = pg.InfiniteLine(
            pen={'color': "#1b9e77", "width": 2},
            angle=90,
            movable=False
        )
        self.image_widget.addItem(self.nt_ventral_position)
        self.nt_dorsal_position = pg.InfiniteLine(
            pen={'color': "#1b9e77", "width": 2},
            angle=90,
            movable=False
        )
        self.image_widget.addItem(self.nt_dorsal_position)

        # create the plot
        # self.plot_column_selector = QComboBox()
        # self.plot_column_selector.currentIndexChanged.connect(self._update_plot)
        self.plot_widget = pg.PlotWidget(parent=self)
        self.plot_slice_line = pg.InfiniteLine(
            angle=90,
            movable=False
        )
        self.plot_widget.addItem(self.plot_slice_line)

        # create ventral slider
        self.ventral_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.ventral_slider.setRange(0, 99)
        self.ventral_slider.setSliderPosition(50)
        self.ventral_slider.setSingleStep(1)
        self.ventral_slider.setTickInterval(1)
        self.ventral_slider.valueChanged.connect(self._move_boundary_with_slider)

        # create dorsal slider
        self.dorsal_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.dorsal_slider.setRange(0, 99)
        self.dorsal_slider.setSliderPosition(50)
        self.dorsal_slider.setSingleStep(1)
        self.dorsal_slider.setTickInterval(1)
        self.dorsal_slider.valueChanged.connect(self._move_boundary_with_slider)

        # create None push button 
        self.Null_widget = QPushButton(text='No boundary')
        self.Null_widget.clicked.connect(self._remove_boundary_on_click)

        # create update push button
        self.update_widget = QPushButton(text='update')
        self.update_widget.clicked.connect(self._update_on_click)

        # create save push button
        self.save_widget = QPushButton(text='save')
        self.save_widget.clicked.connect(self._save_on_click)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.slice_slider)
        self.layout().addWidget(self.image_selector)
        self.layout().addWidget(self.image_widget)
        # self.layout().addWidget(self.plot_column_selector)
        # self.layout().addWidget(self.plot_widget)
        self.layout().addWidget(self.ventral_slider)
        self.layout().addWidget(self.dorsal_slider)
        self.layout().addWidget(self.update_widget)
        self.layout().addWidget(self.Null_widget)
        self.layout().addWidget(self.save_widget)

    def _remove_boundary_on_click(self, event=None):
        # get index values from updated sliders
        current_slice_index = int(self.slice_slider.value())

        # select line of interest in the result file
        res,ind = np.unique(self.results_table["target"], return_index=True)
        target_names_result_table = res[np.argsort(ind)]
        target_name = target_names_result_table[self.current_channel_index]
        self.slice_row = self.results_table.loc[
            (self.results_table["target"] == target_name) &
            (self.results_table["slice_index"] == current_slice_index)
        ]
        
        # update column of that row according to slider position
        self.slice_row["ventral_boundary_px"].values[0] = -1
        self.slice_row["dorsal_boundary_px"].values[0] = -1
        self.slice_row["ventral_boundary_um"].values[0] =  -1
        self.slice_row["dorsal_boundary_um"].values[0] =  -1
        self.slice_row["ventral_boundary_rel"].values[0] =  -1
        self.slice_row["dorsal_boundary_rel"].values[0] =  -1

        self.results_table.loc[
            (self.results_table["target"] == target_name) &
            (self.results_table["slice_index"] == current_slice_index)
        ] = self.slice_row

    def _save_on_click(self, event=None):
        res_path = str(self.results_table_path)
        print('saving')
        self.results_table.to_csv(res_path[0:-4]+'_corrected.csv')
        print('saved')

    def _update_on_click(self, event=None):
        # get index values from updated sliders
        current_ventral_index = int(self.ventral_slider.value()) #+ self.start_nt
        current_dorsal_index = int(self.dorsal_slider.value()) #+ self.start_nt
        current_slice_index = int(self.slice_slider.value())

        # select line of interest in the result file
        res,ind = np.unique(self.results_table["target"], return_index=True)
        target_names_result_table = res[np.argsort(ind)]
        target_name = target_names_result_table[self.current_channel_index]
        self.slice_row = self.results_table.loc[
            (self.results_table["target"] == target_name) &
            (self.results_table["slice_index"] == current_slice_index)
        ]
        
        # update column of that row according to slider position
        self.slice_row["ventral_boundary_px"].values[0] = current_ventral_index
        self.slice_row["dorsal_boundary_px"].values[0] = current_dorsal_index
        self.slice_row["ventral_boundary_um"].values[0] =  current_ventral_index * self.pixel_size_um
        self.slice_row["dorsal_boundary_um"].values[0] =  current_dorsal_index * self.pixel_size_um
        self.slice_row["ventral_boundary_rel"].values[0] =  self.slice_row["ventral_boundary_um"].values[0] / self.slice_row["nt_length_um"].values[0]
        self.slice_row["dorsal_boundary_rel"].values[0] =  self.slice_row["dorsal_boundary_um"].values[0] / self.slice_row["nt_length_um"].values[0]

        self.results_table.loc[
            (self.results_table["target"] == target_name) &
            (self.results_table["slice_index"] == current_slice_index)
        ] = self.slice_row
        
    def _move_boundary_with_slider(self, event=None):
        current_ventral_index = int(self.ventral_slider.value()) #+ self.start_nt
        current_dorsal_index = int(self.dorsal_slider.value()) #+ self.start_nt
        self.nt_ventral_position.setValue(current_ventral_index)
        self.nt_dorsal_position.setValue(current_dorsal_index)   

    def _on_slider_moved(self, event=None):
        self.draw_at_current_slice_index()
        self.ventral_slider.setSliderPosition(self.neural_tube_ventral_boundary)
        self.dorsal_slider.setSliderPosition(self.neural_tube_dorsal_boundary)

    def draw_at_current_slice_index(self):
        current_slice_index = int(self.slice_slider.value())
        self.draw_at_slice_index(current_slice_index)
        self._update_plot_slice_line(current_slice_index)

    def draw_at_slice_index(self, slice_index: int):
        self._get_boundaries(slice_index)
        self._update_image(slice_index)

    def _get_boundaries(self, slice_index: int):
        # update the vertical lines
        res,ind = np.unique(self.results_table["target"], return_index=True)

        # Sorting indices
        target_names_result_table = res[np.argsort(ind)]
        target_name = target_names_result_table[self.current_channel_index]

        #target_name = self.stain_channeL_names[self.current_channel_index]
        slice_row = self.results_table.loc[
            (self.results_table["target"] == target_name) &
            (self.results_table["slice_index"] == slice_index)
        ]

        # set the neural tube boundaries
        if self.draw_domain_boundaries:
            self.start_nt = slice_row["nt_start_column_px"].values[0]
            self.end_nt = slice_row["nt_end_column_px"].values[0]
            self.neural_tube_ventral_boundary =  slice_row["ventral_boundary_px"].values[0] #+ self.start_nt
            self.neural_tube_dorsal_boundary =  slice_row["dorsal_boundary_px"].values[0] #+ self.start_nt
            self.nt_ventral_position.setValue(self.neural_tube_ventral_boundary)
            self.nt_dorsal_position.setValue(self.neural_tube_dorsal_boundary)
        else:
            self.nt_ventral_position.setVisible(False)
            self.nt_dorsal_position.setVisible(False)

    def _update_image(self, slice_index: int):
        # offset the slice index since we only have a subset of the slices
        if self.image_slices is None:
            # if images haven't been set yet, do nothing
            return
        # self.min_slice = min(self.results_table["slice_index"].values)
        offset_slice_index = slice_index #- self.min_slice
        upper_bound = np.shape(self.image_slices)[2]//2 + 30
        lower_bound = np.shape(self.image_slices)[2]//2 - 30
        image_slice = self.image_slices[self.current_channel_index, offset_slice_index, lower_bound:upper_bound, self.start_nt:self.end_nt]

        # update the image slice
        self.image_widget.setImage(image_slice)

    def _update_plot(self, event=None):

        if len(self.stain_channeL_names) == 0:
            # if no data loaded - just return
            return

        self.plot_widget.clear()

        # get the column
        current_target = self.stain_channeL_names[self.current_channel_index]
        target_measurements = self.results_table.loc[self.results_table["target"] == current_target]
        column_to_plot = self.plot_column_selector.currentText()
        y_data = target_measurements[column_to_plot].values
        x_data = target_measurements["slice_index"].values

        self.plot_widget.plot(x_data, y_data)

        self.plot_widget.addItem(self.plot_slice_line)
        current_slice_index = int(self.slice_slider.value())
        self.plot_slice_line.setValue(current_slice_index)

        # set the labels
        axis_parameters = {
            "bottom": pg.AxisItem(orientation="bottom", text="slice index"),
            "left": pg.AxisItem(orientation="left", text=column_to_plot)
        }
        self.plot_widget.setAxisItems(axis_parameters)

    def _update_plot_slice_line(self, slice_index):
        self.plot_slice_line.setValue(slice_index)

    def set_data(
            self,
            stain_image: np.ndarray,
            results_table: pd.DataFrame,
            pixel_size_um: float,
            results_table_path: str,
            stain_channel_names: Optional[List[str]]=None,
    ):
        if stain_image.ndim == 3:
            # make sure the image is 4D
            # (channel, slice, y, x)
            stain_image = np.expand_dims(stain_image, axis=0)
        # set the range slider range
        self.min_slice = results_table["slice_index"].min()
        self.max_slice = results_table["slice_index"].max()
        self.slice_slider.setRange(self.min_slice, self.max_slice)

        self.pixel_size_um = pixel_size_um
        self.image_slices = stain_image
        self.results_table = results_table
        self.results_table_path = results_table_path

        # update the plot-able columns
        # column_names = list(self.results_table.columns.values)
        # self.plot_column_selector.clear()
        # self.plot_column_selector.addItems(column_names)

        # add the image channels
        if stain_channel_names is not None:
            self.stain_channeL_names = stain_channel_names

        else:
            n_channels = stain_image.shape[0]
            self.stain_channeL_names = [
                f"channel {channel_index}" for channel_index in range(n_channels)
            ]

        # check if all stain channels are in the results table
        contains_channel = [
            np.any(results_table["target"] == channel)
            for channel in self.stain_channeL_names
        ]
        all_channels_in_results_table = np.all(contains_channel)

        self.draw_domain_boundaries = True    
        # if ("nt_start_column_px" in self.results_table.columns) and all_channels_in_results_table:
        #     self.draw_domain_boundaries = True
        # else:
        #     self.draw_domain_boundaries = False

        self.image_selector.clear()
        self.image_selector.addItems(self.stain_channeL_names)

        # refresh the selected channel index and redraw
        self._update_current_channel()

        self.setVisible(True)

    def _update_current_channel(self, event=None):
        if len(self.stain_channeL_names) == 0:
            # don't do anything if there aren't any channels
            return
        self.current_channel_index = self.image_selector.currentIndex()
        self.draw_at_current_slice_index()

class QtResultsViewer(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()

        # store the viewer
        self._viewer = napari_viewer

        # make the load data widget
        self.load_data_widget = magicgui(
            self.load_data,
            sliced_image_path={
                'label': 'sliced image path',
                'widget_type': 'FileEdit', 'mode': 'r',
                'filter': '*.h5'
            },
            results_table_path={
                'label': 'results table path',
                'widget_type': 'FileEdit', 'mode': 'r',
                'filter': '*.csv'
            },
            spline_path={
                'label': 'spline path',
                'widget_type': 'FileEdit',
                'mode': 'r',
                'filter': '*.json'
            },
            raw_image_path={
                'label': 'raw image path',
                'widget_type': 'FileEdit', 'mode': 'r',
                'filter': '*.h5'
            }
        )
        self.image_slice_widget = QtImageSliceWidget()
        self.image_slice_widget.setVisible(False)

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.load_data_widget.native)
        self.layout().addWidget(self.image_slice_widget)

    def load_data(
        self,
        segmentation_channel: int = 3,
        sliced_image_path: str = "",
        results_table_path: str = "",
        spline_path: str = "",
        raw_image_path: str = "",
        pixel_size_um: float = 5.79
    ):
        self._load_slices(
            image_path=sliced_image_path,
            results_table_path=results_table_path,
            pixel_size_um=pixel_size_um
        )
        self._load_spline(spline_path=spline_path)
        self._load_raw_image(image_path=raw_image_path)

    def _load_slices(self, image_path: str, results_table_path: str, pixel_size_um: float):
        # load the data
        results_table = pd.read_csv(results_table_path)

        stain_image, stain_channels = self._prepare_slices(
            image_path=image_path,
            results_table=results_table
        )

        self.image_slice_widget.set_data(
            stain_image=stain_image,
            stain_channel_names=stain_channels,
            results_table=results_table,
            results_table_path=results_table_path,
            pixel_size_um=pixel_size_um
        )

    def _prepare_slices(self, image_path: str, results_table: pd.DataFrame):
        hdf5_file = h5py.File(image_path)
        stain_image = hdf5_file[list(hdf5_file.keys())[0]]

        if stain_image.ndim == 3:
            # make sure the image is 4D
            # (channel, slice, y, x)
            stain_image = np.expand_dims(stain_image, axis=0)

        # if no channels are present, make names with the channel index
        n_channels = stain_image.shape[0]
        

        stain_channel_names = [
            f"channel {channel_index}" for channel_index in range(n_channels)
        ]
        return stain_image, stain_channel_names

    def _load_spline(self, spline_path: str):
        # get the reader
        reader = napari_get_reader(spline_path)
        if reader is None:
            raise ValueError(f"no reader found for {spline_path}")

        # load the layer data
        layer_data = reader(spline_path)[0]

        # add the layer to the viewer
        self._viewer.add_layer(Layer.create(*layer_data))

        # add the slicing plane
        spline_model = self._viewer.layers["spline"].metadata["spline"]
        plane_coords, faces, _, _ = splineslicer.view.results_viewer_utils.get_plane_coords(
            spline_model, 0.5, 10
        )
        values = np.ones(4)
        self._viewer.add_surface(data=(plane_coords, faces, values), name="slice plane")
        self.image_slice_widget.slice_slider.valueChanged.connect(self._update_slice_plane)

        # add the slicing point
        self._viewer.add_points(data=[[0, 0, 0]], name="slice point", shading="spherical")

    def _update_slice_plane(self, slice_coordinate):
        spline_model = self._viewer.layers["spline"].metadata["spline"]
        plane_coords, faces, center_position, plane_normal = splineslicer.view.results_viewer_utils.get_plane_coords(
            spline_model, slice_coordinate / 100, 10
        )
        values = np.ones(4)
        self._viewer.layers["slice plane"].data = (plane_coords, faces, values)
        self._viewer.layers["slice point"].data = center_position

    def _load_raw_image(self, image_path: str):
        # load the image
        with h5py.File(image_path) as f:
            image = f[list(f.keys())[0]][:]

        # add the layer to the viewer
        self._viewer.add_image(
            image,
            name="raw image"
        )

class QtUpdatedRotation(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self._viewer = napari_viewer

        self.load_data_widget = magicgui(
            self.load_data,
            sliced_image_path={
                'label': 'sliced image path',
                'widget_type': 'FileEdit', 'mode': 'r',
                'filter': '*.h5'
            },
            segmented_image_path={
                'label': 'segmented image path',
                'widget_type': 'FileEdit', 'mode': 'r',
                'filter': '*.h5'
            }
        )

        self.image_slices: Optional[np.ndarray] = None
        self.results_table: Optional[pd.DataFrame] = None
        self.pixel_size_um: float = 5.79
        self.stain_channeL_names: List[str] = []
        self.min_slice: int = 0
        self.max_slice: int = 0
        self.draw_domain_boundaries = True
        self.current_channel_index: int = 0

        # make the binarize section
        self._binarize_section = QCollapsible(title='1. binarize', parent=self)
        self._binarize_widget = magicgui(
            self._binarize_segmentation,
            call_button='binarize image'
        )
        self._binarize_section.addWidget(self._binarize_widget.native)
        self._viewer.layers.events.inserted.connect(
            self._binarize_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._binarize_widget.reset_choices
        )
        self._binarize_widget.reset_choices()

        # make rotation widget
        self._rotate_section = QCollapsible(title='2. align images', parent=self)
        self._rotate_widget = magicgui(
            new_align_rotate,
            mask_layer={'choices': self._get_image_layers},
            stain_layer={'choices': self._get_image_layers},
            NT_segmentation_index={"choices": [0, 1, 2]},
            background_index={"choices": [0, 1, 2]},
            call_button='rotate and align layers'
        )
        self._rotate_section.addWidget(self._rotate_widget.native)

        self._viewer.layers.events.inserted.connect(
            self._rotate_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._rotate_widget.reset_choices
        )
        self._rotate_widget.reset_choices()

        # create the saving button to save rotation
        self.save_rotation_widget = magicgui(
            self._save_rotation,
            output_path={
                'label': 'select saving path',
                'widget_type': 'FileEdit', 'mode': 'd',
                'filter': ''
            },
            call_button='save rotation'
        )

        # set the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.load_data_widget.native)
        self.layout().addWidget(self._binarize_section)
        self.layout().addWidget(self._rotate_section)
        self.layout().addWidget(self.save_rotation_widget.native)

    def _binarize_segmentation(self, threshold: float = 0.5, closing_size: int = 3) -> "napari.types.LayerDataTuple":
        im_seg = self._viewer.layers.selection.active.data
        binarized_im = np.zeros((im_seg.shape))
        for i, chan in enumerate(im_seg):
            binarized_im[i,...] = splineslicer.skeleton.binarize.binarize_image(im = im_seg, channel=i, threshold=threshold)
        layer_type = 'image'
        metadata = {
            'name': 'binarized_image',
            'colormap': 'blue'
        }
        # self._viewer._add_layer_from_data(binarized_im, metadata, binarized_im)
        return (binarized_im, metadata, layer_type)

    def load_data(
        self,
        sliced_image_path: str = "",
        segmented_image_path: str = ""
    ) -> List[napari.types.LayerDataTuple]: 
        # load the image containing the data
        hdf5_file = h5py.File(sliced_image_path, 'r+')
        image_slices = hdf5_file[list(hdf5_file.keys())[0]]
        layer_type ='image'
        metadata = {
            'name': 'sliced_image',
            'colormap': 'gray'
        }

        hdf5_file = h5py.File(segmented_image_path, 'r+')
        segmented_image_slices = hdf5_file[list(hdf5_file.keys())[0]]
        segmented_image_layer_type ='image'
        segmented_image_metadata = {
            'name': 'segmented_image',
            'colormap': 'gray'
        }

        return [(image_slices, metadata, layer_type), (segmented_image_slices, segmented_image_metadata, segmented_image_layer_type)]

    def _get_image_layers(self, combo_widget) -> List[Image]:
        """Get a list of Image layers in the viewer"""
        return [layer for layer in self._viewer.layers if isinstance(layer, Image)]
    
    def _save_rotation(self,
        output_path: str = ""):
        loaded_file = self.load_data_widget.sliced_image_path
        # print(re.match("([\w\d_\.]+\.[\w\d]+)[^\\]", str(loaded_file.value)))
        fname = str(loaded_file.value).split('\\')[-1]
        print(fname)
        output_path=str(output_path)+'/'+fname.replace('.h5','_after_rotation.h5')
        print(output_path)
        with h5py.File(output_path,'w') as f_out:
            f_out.create_dataset(
            'sliced_stack_rotated',
            self._viewer.layers.selection.active.data.shape,
            data=self._viewer.layers.selection.active.data,
            compression='gzip'
        )
        
class QtRotationWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self._viewer = napari_viewer

        self.check_angle = 0
        self.make_copy = 0

        # create the rotation slider
        self.rotate_slider = QLabeledSlider(Qt.Orientation.Vertical)
        self.rotate_slider.setRange(-180, 180)
        self.rotate_slider.setSliderPosition(0)
        self.rotate_slider.setSingleStep(1)
        self.rotate_slider.setTickInterval(1)
        self.rotate_slider.valueChanged.connect(self._on_rotate_slider_moved)

        # reset the slider to 0 upon changing the slice to avoid unwanted rotations
        self.current_step = self._viewer.dims.current_step[1]
        self._viewer.dims.events.current_step.connect(self._update_slider)

        # create apply push button 
        self.apply_button = QPushButton(text='apply rotation')
        self.apply_button.clicked.connect(self._apply_button)

        

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.rotate_slider)
        self.layout().addWidget(self.apply_button)

    def _update_slider(self, event=None):
        self.rotate_slider.setSliderPosition(0)
        self.make_copy = 0
        self.check_angle = 0

    def _on_rotate_slider_moved(self, event=None):
        self.check_angle = int(self.rotate_slider.value())
        #is of form (2,50,150,150) where (channel, slice, x, y) and I want to change the rotation for all channel at one slice
        if self.make_copy == 0:
            self.ind = self._viewer.dims.current_step[1]
            self.current_slice = self._viewer.layers.selection.active.data[:,self.ind,...].copy()
            self.make_copy = 1
        
        for i, chan in enumerate(self._viewer.layers.selection.active.data[:,self.ind,...]):
            if i == self._viewer.layers.selection.active.data[:,self.ind,...].shape[0]-1:
                self._viewer.layers.selection.active.data[i,self.ind,...] = rotate(self.current_slice[i,...], self.check_angle, order = 0, resize=False, preserve_range=True)
            else:
                self._viewer.layers.selection.active.data[i,self.ind,...] = rotate(self.current_slice[i,...], self.check_angle, order = 1, resize=False, preserve_range=True)
        self._viewer.layers.selection.active.refresh()

    def _apply_button(self, event=None):
        self.current_slice = self._viewer.layers.selection.active.data[:,self.ind,...].copy()
        for i, chan in enumerate(self._viewer.layers.selection.active.data[:,self.ind,...]):
            if i == self._viewer.layers.selection.active.data[:,self.ind,...].shape[0]-1:
                self._viewer.layers.selection.active.data[i,self.ind,...] = rotate(self.current_slice[i,...], self.check_angle, order = 0, resize=False, preserve_range=True)
            else:
                self._viewer.layers.selection.active.data[i,self.ind,...] = rotate(self.current_slice[i,...], self.check_angle, order = 1, resize=False, preserve_range=True)

class QtUpdatedMeasurements(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self._viewer = napari_viewer

        self.load_data_widget = magicgui(
            self.load_data,
            final_output_path={
                'label': 'sliced image path',
                'widget_type': 'FileEdit', 'mode': 'r',
                'filter': '*.h5'
            }
        )

        # make the binarize section
        self._update_section = QCollapsible(title='1. update metadata', parent=self)
        self._update_widget = magicgui(
            update_metadata,
            image_layer={'choices': self._get_image_layers},
            channel_0={"choices": ['Olig2', 'Nkx2_2', 'Pax6', 'Laminin', 'Sox2', 'Pax7', 'Shh', 'Arx1', 'Unknown']},
            channel_1={"choices": ['Olig2', 'Nkx2_2', 'Pax6', 'Laminin', 'Sox2', 'Pax7', 'Shh', 'Arx1', 'Unknown']},
            channel_2={"choices": ['Olig2', 'Nkx2_2', 'Pax6', 'Laminin', 'Sox2', 'Pax7', 'Shh', 'Arx1', 'Unknown']},
            channel_3={"choices": ['Olig2', 'Nkx2_2', 'Pax6', 'Laminin', 'Sox2', 'Pax7', 'Shh', 'Arx1', 'Unknown']},
            call_button='update metadata with channel names'
        )
        self._update_section.addWidget(self._update_widget.native)
        self._viewer.layers.events.inserted.connect(
            self._update_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._update_widget.reset_choices
        )
        self._update_widget.reset_choices()

        # make the binarize section
        self._measure_section = QCollapsible(title='2. measure', parent=self)
        self._measure_widget = magicgui(
            measure_boundaries,
            image_layer={'choices': self._get_image_layers},
            segmentation_layer={"choices": [0, 1, 2, 3, 4]},
            spline_file_path={'widget_type': 'FileEdit', 'mode': 'r', 'filter': '*.json'},
            table_output_path={'widget_type': 'FileEdit', 'mode': 'w', 'filter': '*.csv'},
            aligned_slices_output_path={'widget_type': 'FileEdit', 'mode': 'w', 'filter': '*.h5'},
            call_button='measure image'
        )
        self._measure_section.addWidget(self._measure_widget.native)
        self._viewer.layers.events.inserted.connect(
            self._measure_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._measure_widget.reset_choices
        )
        self._measure_widget.reset_choices()

        # set the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.load_data_widget.native)
        self.layout().addWidget(self._update_section)
        self.layout().addWidget(self._measure_section)

    def load_data(
        self,
        final_output_path: str = ""
    ) -> "napari.types.LayerDataTuple": 
        # load the image containing the data
        hdf5_file = h5py.File(final_output_path, 'r+')
        image_slices = hdf5_file[list(hdf5_file.keys())[0]]
        layer_type ='image'
        metadata = {
            'name': 'sliced_image',
            'colormap': 'gray'
        }
        return (image_slices, metadata, layer_type)

    def _get_image_layers(self, combo_widget) -> List[Image]:
        """Get a list of Image layers in the viewer"""
        return [layer for layer in self._viewer.layers if isinstance(layer, Image)]


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

def measure_boundaries(
        image_layer: Image,
        segmentation_layer,
        spline_file_path: str,
        table_output_path: str,
        aligned_slices_output_path: str,
        half_width: float = 0.5,
        bg_sample_pos: float = 0.7,
        bg_half_width: int = 2,
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
    seg_im= image_layer.data[segmentation_layer,...]

    targets =  image_layer.metadata.get(
            "channel_names", {}
    )
    # run through the 3 first channels: I should make this automatically adapt to the channel number. 
    for k in range(0, 3):
        target_name = targets[k]
        print(target_name)
        stain_im = image_layer.data[k,...]
        nt_length, ventral_boundary_px, dorsal_boundary_px, bg_sub_profile, raw_profile, cropped_ims, nt_col_start, nt_col_end = find_boundaries_method2(
                    seg_im= seg_im,
                    stain_im= stain_im,
                    half_width = half_width,
                    edge_method = edge_method,
                    edge_value = edge_value, 
                    upper_range = upper_range,
                    lower_range = lower_range
            )

        all_cropped_images.append(cropped_ims)
        bg_sub_profiles.append(bg_sub_profile)
        raw_profiles.append(raw_profile)

        pixel_size_um = 5.79

        nt_length = np.asarray(nt_length)
        nt_length_um = nt_length * pixel_size_um
        ventral_boundary_um = np.asarray(ventral_boundary_px) * pixel_size_um
        dorsal_boundary_um = np.asarray(dorsal_boundary_px) * pixel_size_um

        ventral_boundary_rel = ventral_boundary_um / nt_length_um
        dorsal_boundary_rel = dorsal_boundary_um / nt_length_um

        spline = exchange.import_json(spline_file_path)[0]
        spline_length = operations.length_curve(spline) * pixel_size_um
        spline_increment = spline_length / stain_im.shape[0]
        
        n_slices = len(nt_length_um)
        ###################################target_name = c[i][k] ###################################
        target = [target_name] * n_slices
        threshold_list = [edge_value] * n_slices
        slice_index = np.arange(start_slice, end_slice+1)
        slice_position_um = [i * spline_increment for i in slice_index]
        slice_position_rel_um = [i * spline_increment for i in range(n_slices)]
        
        param_measure_half_width = [half_width] * n_slices
        param_measure_bg_sample_pos = [bg_sample_pos] * n_slices
        param_measure_bg_half_width = [bg_half_width] * n_slices
        param_measure_edge_method = [edge_method] * n_slices
        param_measure_edge_value = [edge_value] * n_slices

        print(len(slice_index),
              len(slice_position_um), 
              len(slice_position_rel_um),
              len(nt_length_um),
              len(target),
              len(ventral_boundary_um),
              len(dorsal_boundary_um),
              len(ventral_boundary_rel),
              len(dorsal_boundary_rel),
              len(threshold_list),
              len(nt_col_start),
              len(nt_col_end),
              len(ventral_boundary_px),
              len(dorsal_boundary_px),
              len(param_measure_half_width),
              len(param_measure_bg_sample_pos),
              len(param_measure_bg_half_width),
              len(param_measure_edge_method),
              len(param_measure_edge_value)
              )
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
                    'param_measure_bg_sample_pos': param_measure_bg_sample_pos,
                    'param_measure_bg_half_width': param_measure_bg_half_width,
                    'param_measure_edge_method': param_measure_edge_method,
                    'param_measure_edge_value': param_measure_edge_value
                }
            )
        channel_data.append(df)

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
