"""
This module has been written by Alexis Villars, mostly based on plugins from Kevin Yamauchi
This is meant as a way to provide correction pluggin for Kevin's splineslicer pluggin
As well as to provide plugin to do the measurements updated with changes I made in Kevin's original function. 
This provides a quick way to test these functions into napari and for other to use during development. 
This piece of code is for development and temporary use and is not meant as a standalone and should be installed with Kevin's splineslicer
"""


"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""

from napari_plugin_engine import napari_hook_implementation

from .rotate._qt_rotator import QtUpdatedRotation, QtRotationWidget
from .view._qt_viewer import QtResultsViewer, QtNormalsViewer
from .sub_slicing._qt_sub_slice import QtDoubleSlider
from .MeasurementsCorrection._qt_correct_measurement import QtUpdatedMeasurements


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [
        (QtUpdatedRotation, {"name": "Updated rotation tools"}),
        (QtRotationWidget, {"name": "Rotation Slider"}),
        (QtUpdatedMeasurements, {"name": "Update Measurements tools"}),
        (QtDoubleSlider, {"name": "sub slicer"}),
        (QtResultsViewer, {"name": "view results"})
    ]