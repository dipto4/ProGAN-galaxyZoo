import numpy as np
import os
from PIL import Image
from astropy.modeling.models import Gaussian2D
from photutils.datasets import make_noise_image
import matplotlib.pyplot as plt

data = np.loadtxt("trimmed_and_selected_elliptical",usecols=(0),dtype=int)
imageids = data
for i in range(0,5000):
    img_file = Image.open("images/{}.jpg".format(imageids[i]))
    img_gray = img_file.convert('L')
    img_g = np.asarray(img_gray)
    #cropping
    width, height = img_gray.size
    center_w = width/2
    center_h = height/2
    left = center_w - 32
    right = center_w + 32
    top = center_h - 32
    bottom = center_h + 32

    im1 = img_gray.crop((left, top, right, bottom))
    #img_g1 = np.asarray(im1)

    #adding noise for proper gradient calculation (?)
    #noise = make_noise_image(np.shape(img_g1), distribution='gaussian', mean=0.0,
    #                     stddev=2.0, seed=1234)

    #img_g1 = noise + img_g1
    #img_g1 = img_g1.astype(np.uint8)
    #img_to_save = Image.fromarray(img_g1)
    print(imageids[i])
    im1.save("selected_elliptical_zoom/{}.png".format(imageids[i]))
    #plt.clf()
    #plt.imsave('selected_elliptical/{}.png'.format(imageids[i]),img_g1)
    #plt.savefig()
    #print(imageids[i])
    #os.system("cp images/{}.jpg selected_elliptical/".format(imageids[i]))
