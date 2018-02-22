# surface_state_viz

Python script that uses the module glumpy for 3d representations of millions of datapoints, produced by the TOF momentum microscope.
For analysis purposes, different approaches for visualizing the data (marching cubes, density, raw data, ...) were used and can be found in the different branches.

### Requirements
Python, numpy, matplotlib, PIL(low), scipy, glumpy

Glumpy requires the installation of the packages a backend (e.g. pyopengl) and the modules cython & triangles
**Important:** Due to some reasons, the glumpy module does not work properly with newest numpy versions, as these are more strictly regarding implicit type casting.
Latest numpy version which is known to work without any problem: numpy 1.10.4

## Installation guide for Windows users
1. Download Anaconda from the [Continuum Download Page](https://www.anaconda.com/download/) and install it.
2. Start the Anaconda Prompt and create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) for glumpy:
```
conda create --name GlumpyEnv numpy=1.10.4
```
This already installs a certain numpy version.
The name of the environment (here: GlumpyEnv) can be chosen freely. All environments can be listed by using the command `conda env list`.

3. Install glumpy using pip: `pip install glumpy`

4. Install the mandatory requirements for glumpy:
```
pip install cython
pip install pyopengl
pip install triangle
```
5. Install freetype:
  - Download a precompiled x64 version from [here](https://github.com/ubawurinna/freetype-windows-binaries)
  - Extract the zip & copy either of the freetype.dlls in the win64 folder to your python3 folder (probably `...\Anaconda3\Library\bin`).
  - Rename the freetype dll file to just "freetype.dll"

6. Install GLFW:
  - Download the x64 bit version from [here](http://www.glfw.org/download.html).
  - Extract the zip & copy one of the glfw.dll files from one of the "lib-xxxxx" folders to your python3 folder (probably `...\Anaconda3\Library\bin`).
  The preferred file is probably from "lib-mingw-w64," you do not need to rename it.
