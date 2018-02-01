import numpy as np

# Masked arrays
import numpy.ma as ma

from PIL import Image

#Read the Tags of the TIFF Image
from PIL.TiffTags import TAGS


tiff_file = '../data/014_HHGAU111.tif'
preload_file = '../data/particle_data.npy'


def preload(verbose=False):
    if verbose: print('Loading preloaded numpy array at "%s"' % preload_file)
    data = np.load(preload_file)
    if verbose: print('Loaded array, shape: ', data.shape)

    im = Image.open(tiff_file)
    if verbose: print (im.info, im.size)
    im.load()
    if verbose: print('Loaded TIFF file information')
    return data, im.n_frames


def load_tiff(verbose=False):
    im = Image.open('../data/014_HHGAU111.tif')
    if verbose: print (im.info, im.size)
    im.load()
    if verbose: print (im.tag.get(0xa300))
    #if verbose: print im.tag[270]
    #for i in im.tag:
    #	if not i == 50341:
    #		if verbose: print TAGS2[i], im.tag[i]
    #if verbose: print im.tag.tags
    if verbose: print (dict(im.tag))
    tifarray = np.zeros((im.size[0],im.size[1],im.n_frames))
    if verbose: print (tifarray.shape)


############## READ IN THE TIFF DATA INTO NUMPY ARRAY
    np.sum(tifarray)
    frames = im.n_frames
    if verbose: print (frames)
    for i in range( frames):
            im.seek(i)
            tifarray[:,:,i] = np.array(im)
        #	if verbose: print im.tell()
        #	for j in range(1400):
        #		if im.getpixel((j,j)):
        #			if verbose: print im.getpixel((j,j))
        #	#im.show()
        #

############# TRANSLATE DATA INTO FORMAT WHICH IS EASIER TO HANDLE
    maxcounts = np.amax(tifarray[:,:,:])
    if verbose: print ('amax', maxcounts)
    # If a certain voxel has a count number higher than 1, it will be drawn several times
    # Useful if transparency is used
    # Not useful at all if there is no transparency!
    mask_temp = [int(maxcounts)]
    for i in range(int(maxcounts)):
            #mask_temp = np.transpose(np.transpose(tifarray.nonzero())) # first run, ugly solution
            mask_temp = np.transpose(np.transpose(ma.nonzero(tifarray > i))) 
            if verbose: print ('Voxels found with', i+1, 'Counts:', mask_temp.shape)
            if not i == 0:
                    data = np.concatenate((data, mask_temp), axis = 1)
            else:
                    data = mask_temp
                    # TODO: Different colors for different count rates
                    #data_allinone = mask_temp
                    #voxelcount = np.array(np.zeros(data_allinone.shape[1]))
            if verbose: print (data.shape, 'New data tensor')
    data = data.astype(np.float64) # convert to float!
    if verbose: print (data[:,0], data[:,-1])
    return data
