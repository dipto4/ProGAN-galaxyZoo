import numpy as np
from matplotlib import pyplot
import glob
import matplotlib.image as mpimg
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid


rng = np.random.default_rng()
sample = rng.integers(5000, size=50)
fig, axs = pyplot.subplots(10,5)

images = glob.glob("images_gray_spiral/spiral/*.png")
print(len(images))
imgs = []
for i in range(0,50):
    j = sample[i]
    imgs.append(read_image(images[j]))

grid = make_grid(imgs,nrow=10)
img = torchvision.transforms.ToPILImage()(grid)
img.show()
#torchvision.utils.save_image(img,fp='originals_elliptical.png')
    #pyplot.tight_layout()
#fig.savefig("originals_elliptical.pdf")


