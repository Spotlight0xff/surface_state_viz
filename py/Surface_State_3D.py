#!/usr/bin/env python3
import numpy as np
# Masked arrays
import numpy.ma as ma
# Three-dimensional scatter plots
from mpl_toolkits.mplot3d import Axes3D
# Matplotlib
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

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float linewidth;
uniform float antialias;

attribute vec4  fg_color;
attribute vec4  bg_color;
attribute float radius;
attribute vec3  position;

varying float v_pointsize;
varying float v_radius;
varying vec4  v_fg_color;
varying vec4  v_bg_color;
void main (void)
{
    v_radius = radius;
    v_fg_color = fg_color;
    v_bg_color = bg_color;

    gl_Position = <transform>;
    gl_PointSize = 2 * (v_radius + linewidth + 1.5*antialias);
}
"""
    # // gl_Position = projection * view * model * vec4(position,1.0);

# Fragment shader for all data points
fragment = """
#version 120

uniform float linewidth;
uniform float antialias;

varying float v_radius;
varying vec4  v_fg_color;
varying vec4  v_bg_color;

float marker(vec2 P, float size)
{
   const float SQRT_2 = 1.4142135623730951;
   float x = SQRT_2/2 * (P.x - P.y);
   float y = SQRT_2/2 * (P.x + P.y);

   float r1 = max(abs(x)- size/2, abs(y)- size/10);
   float r2 = max(abs(y)- size/2, abs(x)- size/10);
   float r3 = max(abs(P.x)- size/2, abs(P.y)- size/10);
   float r4 = max(abs(P.y)- size/2, abs(P.x)- size/10);
   return min( min(r1,r2), min(r3,r4));
}


void main()
{
    float r = (v_radius + linewidth + 1.5*antialias);
    float t = linewidth/2.0 - antialias;
    float signed_distance = length(gl_PointCoord.xy - vec2(0.5,0.5)) * 2 * r - v_radius;
//    float signed_distance = marker((gl_PointCoord.xy - vec2(0.5,0.5))*r*2, 2*v_radius);
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);

    // Inside shape
    if( signed_distance < 0 ) {
        // Fully within linestroke
        if( border_distance < 0 ) {
            gl_FragColor = v_fg_color;
        } else {
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
        }
    // Outside shape
    } else {
        // Fully within linestroke
        if( border_distance < 0 ) {
            gl_FragColor = v_fg_color;
        } else if( abs(signed_distance) < (linewidth/2.0 + antialias) ) {
            gl_FragColor = vec4(v_fg_color.rgb, v_fg_color.a * alpha);
        } else {
            discard;
        }
    }
}
"""

# Vertex shader for the grid, colors are currently not used in the rendering
vertex_box = """
uniform vec4 ucolor;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
attribute vec3 position;
attribute vec4 color;

varying vec4 v_color;

void main()
{
    v_color = ucolor * color;
    gl_Position = <transform>;
}
"""

# Fragment shader for the grid

fragment_box = """
varying vec4 v_color;
void main()
{
    gl_FragColor = v_color;
}
"""

# Show print output? Useful for debugging purposes
verbose = True

# Defines if the Grid is shown in three-dimensional space
draw_box = True # Show the grid? Can be toggled by using the "#" key

# Use precomputed numpy data array instead of translating it each time
preload_data =True

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
    load_tiff(verbose)
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

counter = data.shape[1]

if verbose: print ('Counts in total', counter)

################### REDUCTION OF DATA POINTS
#### Reduce the data shown for performance reasons, use step as a divider
step = 10
data = data[:, ::step]
data[0] /= 1401.
data[1] /= 1401.
data[2] /= 1.*frames
data -= 0.5
if verbose: print (data[:,0], data[:,-1])
for i in range(int(counter/100000)-1):
	if verbose: print (data[:, i*100])

counter = int(counter/step)+1
################### END OF REDUCTION OF DATA POINTS

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
if verbose: print (box['position'])
if verbose: print (box['position'].shape)


program['position'] = np.transpose(data)
program['radius']   = 1.#np.random.uniform(5,10,counter)
program['fg_color'] = 1,0,0,1
#colors = np.random.uniform(0.75, 1.00, (counter, 4))
colors = np.random.uniform(0.01, 0.02, (counter, 4))
# Change the red part of the color of data points to give an impression of the z value
colors[:,0] = data[2, :]+0.5
#colors[:,1] = data[2, :]
#colors[:,2] = data[2, :]
######################################################### TRANSPARENCY
colors[:,3] = 0.02 # 1 equals no transparency at all
# while using transparency, higher radii are recommended
######################################################### END TRANSPARENCY
if verbose: print(colors[:10, :])
program['bg_color'] = colors
program['fg_color'] = colors
program['linewidth'] = 0#1.0#1.0
program['antialias'] = 0#1.0#1.0
program['model'] = np.eye(4, dtype=np.float32) # Model will be at the origin
program['projection'] = np.eye(4, dtype=np.float32)
program['view'] = view

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
        box['ucolor'] = 0., 0., 0., 1
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

app.run(framerate=144)
