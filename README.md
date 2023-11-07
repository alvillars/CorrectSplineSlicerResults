# CorrectSplineSlicerResults

[![License MIT](https://img.shields.io/pypi/l/CorrectSplineSlicerResults.svg?color=green)](https://github.com/alvillars/CorrectSplineSlicerResults/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/CorrectSplineSlicerResults.svg?color=green)](https://pypi.org/project/CorrectSplineSlicerResults)
[![Python Version](https://img.shields.io/pypi/pyversions/CorrectSplineSlicerResults.svg?color=green)](https://python.org)
[![tests](https://github.com/alvillars/CorrectSplineSlicerResults/workflows/tests/badge.svg)](https://github.com/alvillars/CorrectSplineSlicerResults/actions)
[![codecov](https://codecov.io/gh/alvillars/CorrectSplineSlicerResults/branch/main/graph/badge.svg)](https://codecov.io/gh/alvillars/CorrectSplineSlicerResults)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/CorrectSplineSlicerResults)](https://napari-hub.org/plugins/CorrectSplineSlicerResults)

allows to correct the boundary detection from the SplineSlicer

----------------------------------
This napari pluggin was developped as an add-on to splineslicer https://github.com/kevinyamauchi/splineslicer/tree/main from Kevin Yamauchi. 
It is meant to help correcting the results from the boundary detection of splineslicer. Also it improves the rotation during the alignment of the slices and provide an additional way to further correct the rotation before calling the measurements. 



This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

First create an environment `conda create -n splineslicer python=3.8`

then:
    `conda activate splineslicer`
    
    pip install "napari[all]"
    pip install git+https://github.com/kevinyamauchi/splineslicer.git

if you need to update:

    conda activate splineslicer
    
    pip install --upgrade git+https://github.com/kevinyamauchi/splineslicer.git


Then You can install `CorrectSplineSlicerResults` via [pip]:

    pip install CorrectSplineSlicerResults



To install latest development version :

    pip install git+https://github.com/alvillars/CorrectSplineSlicerResults.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"CorrectSplineSlicerResults" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/alvillars/CorrectSplineSlicerResults/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
