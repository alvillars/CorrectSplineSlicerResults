from typing import Any, Dict, List, Optional, TYPE_CHECKING
import h5py
from magicgui import magicgui
import napari
from napari.layers import Layer, Image, Shapes
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget, QPushButton
from superqt.sliders import QLabeledSlider
from skimage.transform import rotate
import splineslicer
from superqt.collapsible import QCollapsible
from splineslicer._reader import napari_get_reader
from geomdl import exchange, operations, BSpline
import geomdl

from .slicer_utils import slice_it


class QtDoubleSlider(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()

        # store the viewer
        self._viewer = napari_viewer

        # set the resolution 
        self.slider_resolution: int = 1000
        self.count_splines = 0

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


        # create the rostral slider
        self.rostral_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.rostral_slider.setRange(0, self.slider_resolution)
        self.rostral_slider.setSliderPosition(0)
        self.rostral_slider.setSingleStep(1)
        self.rostral_slider.setTickInterval(1)

        # create the caudal slider
        self.caudal_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.caudal_slider.setRange(0, self.slider_resolution)
        self.caudal_slider.setSliderPosition(self.slider_resolution)
        self.caudal_slider.setSingleStep(1)
        self.caudal_slider.setTickInterval(1)

        # subspline button
        self.subpline_button = QPushButton(text='create subspline')
        self.subpline_button.clicked.connect(self._subspline)

        self._slice_section = QCollapsible(title='slicing', parent=self)
        self._slice_widget = magicgui(slice_it,
            im_layer={'choices': self._get_image_layers},
            spline_layer={'choices': self._get_spline_layers},
            call_button='slice it')
        self._slice_section.addWidget(self._slice_widget.native)
        self._slice_section.addWidget(self._slice_widget.native)
        self._viewer.layers.events.inserted.connect(
            self._slice_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._slice_widget.reset_choices
        )
        self._slice_widget.reset_choices()

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.load_data_widget.native)
        self.layout().addWidget(self.rostral_slider)
        self.layout().addWidget(self.caudal_slider)
        self.layout().addWidget(self.subpline_button)
        self.layout().addWidget(self._slice_section)

    def load_data(
        self,
        spline_path: str = "",
        raw_image_path: str = "",
        pixel_size_um: float = 5.79
    ):
        self._load_spline(spline_path=spline_path)
        self._load_raw_image(image_path=raw_image_path)

    def _load_spline(self, spline_path: str):
        # get the reader
        reader = napari_get_reader(spline_path)
        if reader is None:
            raise ValueError(f"no reader found for {spline_path}")

        # load the layer data
        layer_data = reader(spline_path)[0]

        self.spline_path = spline_path
        # add the layer to the viewer
        self._viewer.add_layer(Layer.create(*layer_data))

        # add the slicing plane
        spline_model = self._viewer.layers["spline"].metadata["spline"]
        plane_coords, faces, _, _ = splineslicer.view.results_viewer_utils.get_plane_coords(
            spline_model, 0.5, 10
        )
        values = np.ones(4)
        self._viewer.add_surface(data=(plane_coords, faces, values), name="slice plane rostral", colormap='bop blue')
        self.rostral_slider.valueChanged.connect(self._update_slice_plane_rostral)
        self.rostral_slider.setSliderPosition(0)
        
        self._viewer.add_surface(data=(plane_coords, faces, values), name="slice plane caudal", colormap='bop orange')
        self.caudal_slider.valueChanged.connect(self._update_slice_plane_caudal)
        self.caudal_slider.setSliderPosition(1000)

        # add the slicing point
        self._viewer.add_points(data=[[0, 0, 0]], name="slice point rostral", shading="spherical", face_colormap='bop blue')

        # add the slicing point
        self._viewer.add_points(data=[[0, 0, 0]], name="slice point caudal", shading="spherical", face_colormap='bop orange')

    def _update_slice_plane_rostral(self, slice_coordinate):
        spline_model = self._viewer.layers["spline"].metadata["spline"]
        plane_coords, faces, center_position, plane_normal = splineslicer.view.results_viewer_utils.get_plane_coords(
            spline_model, slice_coordinate / self.slider_resolution, 10
        )
        values = np.ones(4)
        self._viewer.layers["slice plane rostral"].data = (plane_coords, faces, values)
        self._viewer.layers["slice point rostral"].data = center_position

    def _update_slice_plane_caudal(self, slice_coordinate):
        spline_model = self._viewer.layers["spline"].metadata["spline"]
        plane_coords, faces, center_position, plane_normal = splineslicer.view.results_viewer_utils.get_plane_coords(
            spline_model, slice_coordinate / self.slider_resolution, 10
        )
        values = np.ones(4)
        self._viewer.layers["slice plane caudal"].data = (plane_coords, faces, values)
        self._viewer.layers["slice point caudal"].data = center_position

    def _load_raw_image(self, image_path: str):
        # load the image
        with h5py.File(image_path) as f:
            image = f[list(f.keys())[0]][:]

        # add the layer to the viewer
        self._viewer.add_image(
            image,
            name="raw image"
        )

    def _subspline(self, event=None):
        spline_model = self._viewer.layers["spline"].metadata["spline"]
        start = int(self.rostral_slider.value()) / self.slider_resolution
        stop = int(self.caudal_slider.value()) / self.slider_resolution
        spline_model.evaluate(start=start, stop=stop)
        curve2 = BSpline.Curve()

        # Set degree
        curve2.degree = 3

        # Set control points
        curve2.ctrlpts = spline_model.evalpts # here we get 100 evalpts from the original spline, so I guess knotvector should be set to 100

        # set knotvector
        curve2.knotvector = geomdl.knotvector.generate(3,100)

        curve2.delta = 0.01

        self.count_splines = self.count_splines + 1

        self.curve2 = curve2
        curve2_layer_data = (
                curve2.ctrlpts,
                {   'name' : 'Spline_n_'+str(self.count_splines),
                    'shape_type': 'path',
                    'edge_width': 0.5,
                    'edge_color': 'green',
                    'metadata': {'spline': curve2}
                },
                'shapes'
            )
        
        output_path = str(self.spline_path)
        output_path = output_path.replace(".json","_subspline.json")
        exchange.export_json(curve2, output_path)
        self._viewer.add_layer(Layer.create(*curve2_layer_data))

    def _get_image_layers(self, combo_widget) -> List[Image]:
        """Get a list of Image layers in the viewer"""
        return [layer for layer in self._viewer.layers if isinstance(layer, Image)]
    
    def _get_spline_layers(self, combo_widget) -> List[Shapes]:
        """Get a list of Spline/shape layers in the viewer"""
        return [layer for layer in self._viewer.layers if isinstance(layer, Shapes)]
