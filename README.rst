========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/pyfimtdd/badge/?style=flat
    :target: https://pyfimtdd.readthedocs.io/
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.com/MLRichter/pyfimtdd.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.com/github/MLRichter/pyfimtdd

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/MLRichter/pyfimtdd?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/MLRichter/pyfimtdd

.. |requires| image:: https://requires.io/github/MLRichter/pyfimtdd/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/MLRichter/pyfimtdd/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/MLRichter/pyfimtdd/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/MLRichter/pyfimtdd

.. |version| image:: https://img.shields.io/pypi/v/pyfimtdd.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/pyfimtdd

.. |wheel| image:: https://img.shields.io/pypi/wheel/pyfimtdd.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/pyfimtdd

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/pyfimtdd.svg
    :alt: Supported versions
    :target: https://pypi.org/project/pyfimtdd

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pyfimtdd.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/pyfimtdd

.. |commits-since| image:: https://img.shields.io/github/commits-since/MLRichter/pyfimtdd/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/MLRichter/pyfimtdd/compare/v0.0.0...master



.. end-badges

Python implementation for the Fast Incremental Model Tree for Drift Detection algorithm.

* Free software: MIT license

Installation
============

::

    pip install pyfimtdd

You can also install the in-development version with::

    pip install https://github.com/MLRichter/pyfimtdd/archive/master.zip


Documentation
=============


https://pyfimtdd.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
