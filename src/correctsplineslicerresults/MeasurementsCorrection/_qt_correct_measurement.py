from typing import Any, Dict, List, Optional, TYPE_CHECKING
import h5py
from magicgui import magicgui
import napari
from napari.layers import Layer, Image, Shapes
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget, QPushButton
from skimage.transform import rotate
from superqt.collapsible import QCollapsible
from .correction_utils import update_metadata, measure_boundaries

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

        List_target = ['Olig2', 'Nkx2_2', 'Pax6', 'Laminin', 'Sox2', 
                       'Pax7', 'Shh', 'Arx1', 'DAPI', 'Fgf8', 'Mesp2', 'Ngn1',
                        'Ngn2', 'Ascl1', 'Unknown']
        # make the binarize section
        self._update_section = QCollapsible(title='1. update metadata', parent=self)
        self._update_widget = magicgui(
            update_metadata,
            image_layer={'choices': self._get_image_layers},
            channel_0={"choices": List_target},
            channel_1={"choices": List_target},
            channel_2={"choices": List_target},
            channel_3={"choices": List_target},
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
            spline_file_path={'widget_type': 'FileEdit', 'mode': 'r', 'filter': '*.json'},
            table_output_path={'widget_type': 'FileEdit', 'mode': 'w', 'filter': '*.csv'},
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
