from typing import Any, Dict, List, Optional, TYPE_CHECKING
import h5py
from magicgui import magicgui
from napari.layers import Layer, Image, Shapes
import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget, QPushButton
from superqt.sliders import QLabeledSlider
import splineslicer
from splineslicer._reader import napari_get_reader
import napari

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
        plane_coords, faces, center_position, plane_normal = splineslicer.view.results_viewer_utils.get_plane_coords(
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

class QtNormalsViewer(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()

        # store the viewer
        self._viewer = napari_viewer

        # make the load data widget
        self.load_data_widget = magicgui(
            self.load_data,
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
        
        # create a slider to go through all slices and check the normal of the plane
        self.slice_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 99)
        self.slice_slider.setSliderPosition(50)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.setTickInterval(1)
        self.slice_slider.valueChanged.connect(self._update_slice_plane)

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.load_data_widget.native)
        self.layout().addWidget(self.slice_slider)

    def load_data(
        self,
        spline_path: str = "",
        raw_image_path: str = ""
    ):
        self._load_raw_image(image_path=raw_image_path)
        self._load_spline(spline_path=spline_path)
        #self._load_segmentation(image_path=olig_seg_path)

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
        plane_coords, faces, center_position, plane_normal = splineslicer.view.results_viewer_utils.get_plane_coords(
            spline_model, 0.5, 10
        )
        values = np.ones(4)
        self._viewer.add_surface(data=(plane_coords, faces, values), name="slice plane")

        self.vector = [center_position, plane_normal]
        self._viewer.add_vectors(self.vector, edge_width=1, length=100, edge_color='green', name="normal_vector")

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

        self.vector = [center_position, plane_normal]
        self._viewer.layers["normal_vector"].data = self.vector


        plane_parameters = {
            'position': (center_position[0], center_position[1], center_position[2]),
            'normal': (plane_normal[0], plane_normal[1], plane_normal[2]),
            'thickness': 10}

        self._viewer.layers["plane"].plane = plane_parameters
        
    def _load_raw_image(self, image_path: str):
        # load the image
        with h5py.File(image_path) as f:
            image = f[list(f.keys())[0]][:]
        
        # add the layer to the viewer
        self._viewer.add_image(
            image,
            name="raw image"
        )

        plane_parameters = {
            'position': (32, 32, 32),
            'normal': (1, 0, 0),
            'thickness': 10}

        self._viewer.add_image(
            data = self._viewer.layers["raw image"].data,
            rendering='average',
            name='plane',
            colormap='bop orange',
            blending='additive',
            opacity=0.5,
            depiction="plane",
            plane=plane_parameters)

    def _load_segmentation(self, image_path: str):
        # load the image
        with h5py.File(image_path) as f:
            seg_im = f[list(f.keys())[0]][0,...]

        # add the layer to the viewer
        self._viewer.add_image(
            seg_im,
            name="Olig2 segmentation"
        )
