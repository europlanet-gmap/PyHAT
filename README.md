
## Status

| OS    | Status |
|-------|--------|
| OSX   | [![Build Status](https://travis-ci.org/USGS-Astrogeology/PyHAT.svg?branch=dev)](https://travis-ci.org/USGS-Astrogeology/PyHAT) |
|Linux  | [![Build Status](https://travis-ci.org/USGS-Astrogeology/PyHAT.svg?branch=dev)](https://travis-ci.org/USGS-Astrogeology/PyHAT) |
|Windows| [![Build status](https://ci.appveyor.com/api/projects/status/orfb1txhicspo7ap/branch/dev?svg=true)](https://ci.appveyor.com/project/jlaura/pyhat/branch/dev)|


[![Coverage Status](https://coveralls.io/repos/github/USGS-Astrogeology/PyHAT/badge.svg?branch=dev)](https://coveralls.io/github/USGS-Astrogeology/PyHAT?branch=dev)
[![Join the chat at https://gitter.im/USGS-Astrogeology/pyhat](https://badges.gitter.im/USGS-Astrogeology/pyhat.svg)](https://gitter.im/USGS-Astrogeology/pyhat?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# Installation - For Users
  - Install [Anaconda Python](https://www.continuum.io/downloads).  Be sure to get Python 3.x
  - Create a conda env for PyHAT: `conda create -n pyhat`.
  - Add conda forge to your channels list: `conda config --add channels conda-forge`
  - Activate the PyHAT environment: `conda activate pyhat` (for windows: `conda pyhat`)
  - To install: `conda install -c usgs-astrogeology pyhat`

# Installation - For Developers
  - Install [Anaconda Python](https://www.continuum.io/downloads).  Be sure to get Python 3.x
  - Add conda forge to your channels list: `conda config --add channels conda-forge`.
  - Clone this repo: `git clone https://github.com/USGS-Astrogeology/PyHAT`.
  - Enter the cloned repo: `cd PyHAT`.
  - Pull the `dev` branch: `git fetch && git checkout dev`.
  - Install the dependencies: `conda env create -f environment.yml`.
  - For Ubuntu/Linux: Update your `$PYTHONPATH' to include the PyHAT directory.

# Jupyter Notebook Demo
  - For Ubuntu/Linux: Update your $JUPYTERPATH to include the installed PyHAT dependencies.
      - It will look something like this: `/path-to-anaconda/anaconda3/envs/pyhat/lib/python3.9/site-packages/.`
      - Note that as Anaconda and Python versions change, so might this path.
  - Activate the PyHAT environment: `conda activate PyHAT`.
  - Execute the `jupyter notebook` that will open a new browser tab with the Jupyter homepage.
  - Navigate to the `notebooks' folder in the PyHAT directory.
  - Launch (click) the `Horgan Example.ipynb` notebook.
      - Note: Not all notebooks are working at the moment.
  
# QGIS Plugin
  - A QGIS plugin is available [here](https://github.com/USGS-Astrogeology/pyhat_qgis).  This plugin allows users to create and view derived products using QGIS rather than a native Python environment.
