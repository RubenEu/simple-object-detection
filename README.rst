=======================
Simple Object Detection
=======================


.. image:: https://img.shields.io/pypi/v/simple_object_detection.svg
        :target: https://pypi.python.org/pypi/simple_object_detection

.. image:: https://img.shields.io/travis/rubeneu/simple_object_detection.svg
        :target: https://travis-ci.com/rubeneu/simple_object_detection

.. image:: https://readthedocs.org/projects/simple-object-detection/badge/?version=latest
        :target: https://simple-object-detection.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



Toolkit and models for object detection.

* Free software: MIT license
* Documentation: https://simple-object-detection.readthedocs.io.


Features
--------

* Wrapper for object detections neural networks.
* Class for store object information (center, bounding box, width, height, score, label, ...)
* Abstract class for PyTorcuh hub models easy implementation.
* Implemented neuronal networks: YOLOv5 with PyTorch.
* Image tools to draw bounding boxes on it.
* Video tools to load and easily generate detections.
* File geneneration with objects detections.
* Objects filters based on its label, score, region, etc.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
