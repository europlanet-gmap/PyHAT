
## Status

[![](http://github-actions.40ants.com/USGS-Astrogeology/PyHAT/matrix.svg)](https://github.com/USGS-Astrogeology/PyHAT)

[![codecov](https://codecov.io/gh/USGS-Astrogeology/PyHAT/branch/master/graph/badge.svg?token=chD2TczRZn)](https://codecov.io/gh/USGS-Astrogeology/PyHAT)

# Installation - For Users
  - Install [Anaconda Python](https://www.continuum.io/downloads).  Be sure to get Python 3.x
  - Create a conda env for PyHAT: `conda create -n pyhat`.
  - Activate the PyHAT environment: `conda activate pyhat` (for windows: `conda pyhat`)
  - Add conda forge to your channels list: `conda config --env --add channels conda-forge`
  - To install: `conda install -c usgs-astrogeology pyhat`

# Installation - For Developers
  - Install [Anaconda Python](https://www.continuum.io/downloads).  Be sure to get Python 3.x
  - Create a conda env for PyHAT: `conda create -n pyhat`.  
  - Activate the PyHAT environment: `conda activate pyhat` (for windows: `conda pyhat`)
  - Add conda forge to your channels list: `conda config --env --add channels conda-forge` 
  - Install the dependencies: `conda env update -f environment.yml`.  
  - Clone this repo: `git clone https://github.com/USGS-Astrogeology/PyHAT`.
  - Enter the cloned repo: `cd PyHAT`.
  - Pull the `dev` branch: `git fetch && git checkout dev`.
  - If you plan to test locally before sending a pull request for testing remotely, you'll need to install some prerequisites
      - `conda install --quiet pytest pytest-cov nbsphinx pytest --cov=libpyhat`
      - When you want to run a test: `pytest --cov=libpyhat`
  - For Ubuntu/Linux: Update your `$PYTHONPATH` to include the PyHAT directory.

# Jupyter Notebook Demo
  - Activate the PyHAT environment: `conda activate PyHAT`.
  - Install Jupyter: `conda install jupyter'.
  - Jupyter notebook likely won't have the path to libpyhat set up out of the box.
      - For Ubuntu/Linux: Update your `$JUPYTER_PATH` to include the installed PyHAT dependencies.
          - It will look something like this: `/path-to-anaconda/anaconda3/envs/pyhat/lib/python3.9/site-packages/.`
          - You may also need to add the PyHAT install directory that you cloned from GitHub to your path.
          - Note that as Anaconda and Python versions change, so this path might change.
      - For Mac: It seems you only need to add the PyHAT install directory.
          - You can just add this to the top of your notebooks:
              - `import sys && sys.path.append('/path-to-cloned-repo/')`
  - Execute the `jupyter notebook` that will open a new browser tab with the Jupyter homepage.
  - Navigate to the `notebooks' folder in the PyHAT directory.
  - Launch (click) the `Horgan Example.ipynb` notebook.
      - Note: Not all notebooks are working at the moment.
  
# QGIS Plugin
  - A QGIS plugin is available [here](https://github.com/USGS-Astrogeology/pyhat_qgis).  This plugin allows users to create and view derived products using QGIS rather than a native Python environment.
