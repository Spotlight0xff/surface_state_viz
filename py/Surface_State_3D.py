#!/usr/bin/env python3
import numpy as np
# Masked arrays
import numpy.ma as ma
# Three-dimensional scatter plots
from mpl_toolkits.mplot3d import Axes3D
# Matplotlib
import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
# Glumpy package for GPU-rendering
from glumpy import app, gl, gloo, glm, transforms
import itertools
# Find fit paraboloid
import scipy.optimize as opt
from glumpy.graphics.collections import GlyphCollection
from glumpy.graphics.collections import PathCollection
from glumpy.graphics.collections import SegmentCollection

from glumpy.transforms import Trackball, Position

# Used for vertices and indices of a cube
from glumpy.geometry import colorcube, primitives 

from data import load_tiff, preload

def paraBolEqn(data,b,curv,d, zcenter):
    ''' Equation for the paraboloid used to fit the surface state at the Fermi edge.'''
    x,y = data
    return curv*((x-b)**2+(y-d)**2)+zcenter

###################### SHADERS ######################

# Vertex shader for all data points
vertex = """
#version 120

uniform float uBoxsize;


uniform float uTransfer1;
uniform float uTransfer2;

attribute float aDensity;

attribute vec4  aColor;
attribute float aRadius;
attribute vec3  position;

varying vec4 vColor;
varying float vRadius;

vec4 densityTransfer(float aDensity) {
    if (aDensity < uTransfer1) {
        return vec4(0.1, 0.1, 0.1, aDensity);
    } else if (aDensity > uTransfer2) {
        return vec4(0.8, 0.1, 0.1, aDensity);
    }

    // 0.1 -> (uTransfer1) -> smooth -> (uTransfer2) -> 0.8
    float r = 0.1 + 0.7 * smoothstep(uTransfer1, uTransfer2, aDensity);
    return vec4(r, 0.1, 0.1, aDensity);
}

void main (void)
{
    vRadius = aRadius;
    vColor = densityTransfer(aDensity);

    gl_Position = <transform>;
    gl_PointSize = vRadius * 5*aDensity;
}
"""

# Fragment shader for all data points
fragment = """
#version 120

uniform float uBoxSize;

varying float vRadius;
varying vec4  vColor;

void main()
{
    float dist = length(gl_PointCoord.xy - vec2(0.5,0.5)) * vRadius;
    float alpha = exp(-dist);

    if (dist > vRadius) {
        discard;
    } else {
        gl_FragColor = vec4(vColor.rgb, vColor.a*alpha);
    }
}
"""

# Vertex shader for the grid, colors are currently not used in the rendering
vertex_box = """
uniform vec4 uColor;

attribute vec3 position;
attribute vec4 aColor;

varying vec4 vColor;

void main()
{
    vColor = uColor * aColor;
    gl_Position = <transform>;
}
"""

# Fragment shader for the grid

fragment_box = """
varying vec4 vColor;
void main()
{
    gl_FragColor = vColor;
}
"""

# Show print output? Useful for debugging purposes
verbose = True

# Defines if the Grid is shown in three-dimensional space
draw_box = True # Show the grid? Can be toggled by using the "#" key

# Use precomputed numpy data array instead of translating it each time
preload_data = False

# Mouse sensitivity for rotating
sensitivity = 0.5

# Mouse scroll sensitivity for zooming
zoom_sensitivity = 0.5

# Where to find the paraboloid?
param_guess = [660, -0.0018, 600, 35]

# Estimation of the relevant space (where to search for data points relevant for the fit?)
border_guess = (np.sqrt(abs((param_guess[3])/param_guess[1])))*1.08 # also used here: magnification factor 1.08
if verbose: print (border_guess, 'Border guess')

if preload_data:
    data, frames = preload(verbose)
else:
    data, frames = load_tiff(verbose)
################################ PARABOLOID FIT
if verbose: print("Calculating fit")
#data[-2] *= -1
coords_masked = np.copy(data)
mask1 = np.tile(np.array(coords_masked[2] > param_guess[3]+0.05*frames), (3,1))
mask2 = np.tile(np.array(np.sqrt((coords_masked[0]-param_guess[0])**2 + (coords_masked[1]-param_guess[2])**2)  > border_guess), (3,1))
coords_masked = ma.masked_where(np.logical_or(mask2,mask1), coords_masked)
coords_masked = coords_masked.compressed().reshape(3, int(coords_masked.count()/3)) # convert back into np array

if verbose: print(coords_masked.shape, 'Shape of masked coords')

popt,pcov=opt.curve_fit(paraBolEqn,np.vstack((coords_masked[0, :],coords_masked[1, :])),coords_masked[2, :], p0=param_guess , maxfev=1000000)#, diag = (np.vstack((coords[0, :],coords[1, :])).mean(),coords[2, :].mean()) )
if verbose: print ("Param guess was:", param_guess)
if verbose: print (popt, 'opt Values found' )
residuals = coords_masked[2,:]- paraBolEqn(np.vstack((coords_masked[0, :],coords_masked[1, :])), *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((coords_masked[2,:]-np.mean(coords_masked[2,:]))**2)
r_squared = 1 - (ss_res / ss_tot)
if verbose: print ("r squared:", r_squared)
################################ END OF PARABOLOID FIT


#data = coords_masked # THATS ONLY A TEST!!!

# counter = data.shape[1]


################### REDUCTION OF DATA POINTS
#### Reduce the data shown for performance reasons, use step as a divider
step = 10
data = data[:, ::step]
data[0] /= 1401.
data[1] /= 1401.
data[2] /= 1.*frames
data -= 0.5
if verbose: print (data[:,0], data[:,-1])

# counter = int(counter/step)+1
################### END OF REDUCTION OF DATA POINTS

################### VOLUME DATA
# bounding box
bb_min = data.min(axis=1)
bb_max = data.max(axis=1)
box_size = 200
volume_data, edges = np.histogramdd(data.T, bins=box_size)

density_idx = np.transpose(np.nonzero(volume_data))
# add
density = np.zeros((density_idx.shape[0],4), np.float32)
density[:,:-1] = density_idx
density[:,3] = volume_data[density_idx[:,0], density_idx[:,1], density_idx[:,2]]
max_density = density[:,3].max()

# re-normalize
density[:,:-1] /= float(box_size)
density[:,:-1] -= 0.5
density[:,3] /= max_density

import seaborn as sns
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
for i in range(50):
    plt.clf()
    b = np.where(density[:,2] == (i/float(box_size)-0.5))
    print(b)
    den = density[b]
    if den.shape[1] == 0 or len(den)==0: continue
    print('step %i, den: %s' % (i, den.shape))
    sns.kdeplot(den[:,0], den[:,1], shade=True, n_levels=60)
    plt.savefig('density_%i.png' % i)
print('max density: ', max_density)

counter = density.shape[0]
if verbose: print ('Counts in total', counter)
################### END VOLUME DATA

################### GLUMPY INITIALIZATION
theta, phi, zeta = 0, 0, 0
window = app.Window(width=800, height=800, color=(1,1,1,1))


program = gloo.Program(vertex, fragment, count=counter)
view = np.eye(4, dtype=np.float32)
glm.translate(view, 0, 0, -2)

########## GRID WITH TICKS, COPIED FROM LORENZ.PY, CURRENTLY NOT WORKING

#transform = transforms.Trackball(transforms.Position())
#viewport = transforms.Viewport()
#ticks = SegmentCollection(mode="agg++",viewport = viewport, transform = transform, linewidth='local', color='local')
#window.attach(ticks["transform"])
#window.attach(ticks["viewport"])
#xmin,xmax = -1,1
#ymin,ymax = -1,1
#z = 0.5
#
## Frame at z = 1
## -------------------------------------
#P0 = [(xmin,ymin,z), (xmin,ymax,z), (xmax,ymax,z), (xmax,ymin,z)]
#P1 = [(xmin,ymax,z), (xmax,ymax,z), (xmax,ymin,z), (xmin,ymin,z)]
#ticks.append(P0, P1, linewidth=2)
#z = -0.5
#
## Frame at z = -0.5
## -------------------------------------
#P0 = [(xmin,ymin,z), (xmin,ymax,z), (xmax,ymax,z), (xmax,ymin,z)]
#P1 = [(xmin,ymax,z), (xmax,ymax,z), (xmax,ymin,z), (xmin,ymin,z)]
#ticks.append(P0, P1, linewidth=2)

##################################################### INITIALIZATION OF GRID:
vertices, faces, outline = colorcube(.1)
#if verbose: print (outline, outline.shape, type(outline))
#outline = np.concatenate((outline,[20, 0], outline+24, [40,24]), axis=0)
#if verbose: print (outline, outline.shape, type(outline))

if verbose: print (vertices, vertices.shape, type(vertices))
# Copy and translate cube so that we get 1000 cells in total
vertices_n = vertices
for i in range(11):
    for j in range(11):
        for k in range(11):
            vertices2 = np.copy(vertices)
            vertices2['position']+= [(i-5)/10.,(j-5)/10., (k-5)/10.]
            vertices_n = np.concatenate((vertices_n, vertices2), axis = 0)
vertices = vertices_n.view(gloo.VertexBuffer)

box = gloo.Program(vertex_box, fragment_box, count=48)

if verbose: print (vertices, vertices.shape, type(vertices))
box.bind(vertices)
if verbose: print (box['position'].shape)

# data is now (n,4) where [:,:-1] are the coords and [:,-1] is the density
program['position'] = np.transpose(density[:,:-1].T)
program['uBoxsize'] = box_size
program['aRadius']   = 10
program['aColor'] = 1,0,0,1
# some sane defaults for the transfer function to color the densities
program['uTransfer1'] = 0.3
program['uTransfer2'] = 0.5
#colors = np.random.uniform(0.75, 1.00, (counter, 4))
colors = np.random.uniform(0.01, 0.02, (counter, 4))
# Change the red part of the color of data points to give an impression of the z value
colors[:,0] = 0.5
#colors[:,1] = data[2, :]
#colors[:,2] = data[2, :]
######################################################### TRANSPARENCY
# colors[:,3] = density[:,3] # 1 equals no transparency at all
program['aDensity'] = density[:,3]
# while using transparency, higher radii are recommended
######################################################### END TRANSPARENCY

transform = Trackball(Position('position'), distance=3)
program['transform'] = transform
box['transform'] = transform
window.attach(transform)


@window.event
def on_draw(dt):
    global theta, phi, zeta, translate
    window.clear()
    program.draw(gl.GL_POINTS)
    #ticks.draw()
    if draw_box:
        # Outlined cube
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glDepthMask(gl.GL_FALSE)
        # Grid color is BLACK
        box['uColor'] = 0., 0., 0., 1
        box.draw(gl.GL_LINES)#, outline)
        # gl.glDepthMask(gl.GL_TRUE) 


@window.event
def on_init():
    # gl.glEnable(gl.GL_DEPTH_TEST)
    # # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
    gl.glPolygonOffset(1, 1)
    gl.glEnable(gl.GL_LINE_SMOOTH)
    gl.glLineWidth(0.02)

@window.event
def on_key_press(key, modifiers):
    global transform
    # this only works roughly, but not very well
    if key == app.window.key.UP:
        transform.theta += 2 if modifiers & app.window.key.MOD_SHIFT else 0.5
    elif key == app.window.key.DOWN:
        transform.theta -= 2 if modifiers & app.window.key.MOD_SHIFT else 0.5
    elif key == app.window.key.LEFT:
        transform.phi -= 2 if modifiers & app.window.key.MOD_SHIFT else 0.5
    elif key == app.window.key.RIGHT:
        transform.phi += 2 if modifiers & app.window.key.MOD_SHIFT else 0.5
    elif  key == app.window.key.SPACE:
        transform.phi, theta, zeta = 0, 0, 0


@window.event
def on_character(character):
    global draw_box
    if character == '#':
        draw_box = not draw_box # toggle Grid drawing
    # z translations of all objects:
    if character == 'm':
        box['position'] += [0, 0, +0.02]
        program['position'] += [0, 0, +0.02]
    if character == 'n':
        box['position'] -= [0, 0, +0.02]
        program['position'] -= [0, 0, +0.02]
    if character == 's':
        program['uTransfer1'] -= 0.05
        print("transfer1: ", program['uTransfer1'])
    if character == 'w':
        program['uTransfer1'] += 0.05
        print("transfer1: ", program['uTransfer1'])
    if character == 'a':
        program['uTransfer2'] -= 0.05
        print("transfer2: ", program['uTransfer2'])
    if character == 'd':
        program['uTransfer2'] += 0.05
        print("transfer1: ", program['uTransfer2'])



app.run(framerate=144)
