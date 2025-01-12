import numpy as np
from astropy.modeling.models import Gaussian2D
from photutils.datasets import make_noise_image
import matplotlib.pyplot as plt
from photutils.isophote import EllipseGeometry
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse
from PIL import Image
import glob
import natsort
import threading
from functools import wraps
import signal
import multiprocessing
import warnings
warnings.filterwarnings("ignore")



#required for some bizarre multiprocessing bug that prevents the code from finishing. Crude solution but it works!
def stop_function():
    os.kill(os.getpid(), signal.SIGINT)


def stopit_after_timeout(s, raise_exception=True):
    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = threading.Timer(s, stop_function)
            try:
                timer.start()
                result = func(*args, **kwargs)
            except KeyboardInterrupt:
                msg = f'function {func.__name__} took longer than {s} s.'
                if raise_exception:
                    raise TimeoutError(msg)
                result = msg
            finally:
                timer.cancel()
            return result

        return wrapper

    return actual_decorator




def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

@stopit_after_timeout(120, raise_exception=True)
def get_ellipticty(args):
    fname = args[0]
    img_id = args[1]
    #ellipticity pipeline successful (1) or not (0)
    result = 0
    img_file = Image.open(fname)
    #converting to grayscale
    img_gray = img_file.convert('L')
    img_g = np.asarray(img_gray)
    #cropping
    width, height = img_gray.size
    center_w = width/2
    center_h = height/2
    #left = center_w - 128
    #right = center_w + 128
    #top = center_h - 128
    #bottom = center_h + 128

    #im1 = img_gray.crop((left, top, right, bottom))
    #img_g1 = np.asarray(im1)
    img_g1 = img_g
    #adding noise for proper gradient calculation (?)
    noise = make_noise_image(np.shape(img_g1), distribution='gaussian', mean=0.0,
                         stddev=2.0, seed=1234)

    img_g1 = noise + img_g1
    geometry = EllipseGeometry(x0=64, y0=64, sma=5, eps=0.0,
                           pa=20.0 * np.pi / 180.0,linear_growth=False)
    geometry.find_center(img_g1)
    geometry.find_center(img_g1)
    geometry.find_center(img_g1)
    aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                          geometry.sma * (1 - geometry.eps),
                          geometry.pa)
    ellipse = Ellipse(img_g1,geometry)

    isolist = ellipse.fit_image(integrmode='bilinear')

    eps = 0.0
    eps_err = 0.0
    if(len(isolist.sma)>0):
        result = 1

        idx = find_nearest(isolist.sma,30)
        eps = isolist.eps[idx]
        eps_err = isolist.ellip_err[idx]

    return (result,img_id, eps, eps_err)


if __name__ == "__main__":
    #for testing
    #args = ['/hildafs/home/diptajym/hildafs2/10-701/depth/pics/dumped_elliptical/dumped_261.png','261']
    #result = get_ellipticty(args)
    #images = glob.glob("/hildafs/home/diptajym/hildafs2/10-701/depth/pics/dumped_elliptical/*.png")
    #print(result)

    #for img in images[:5]:
    #    img_id = img.split('/')[-1].split('.')[0]
    #    args.append((img,img_id))
    #print(args)
    images = glob.glob("/hildafs/home/diptajym/hildafs2/10-701/depth/pics/dumped_spiral/*.png")
    args = []
    for img in images:
        img_id = img.split('/')[-1].split('.')[0]
        args.append((img,img_id))


    total = len(images)
    count_success = 0
    count = 0
    UP = "\x1B[3A"
    CLR = "\x1B[0K"
    print("\n\n")

    #for debug purposes only
    rrr=[]
    rrre=[]

    processed_ids = []
    eps = []
    eps_err = []
    with multiprocessing.Pool() as pool:
        iterator = pool.imap_unordered(get_ellipticty, args)

        while True:
            try:

                count +=1
                results = iterator.next()
                processed_percent = count/total * 100
                if(results[0]>0):
                    count_success += 1
                print(f"{UP}processed: {processed_percent}%{CLR}\nellipticity found: {count_success}{CLR}\n",flush=True)

                if(results[0]>0):
                    processed_ids.append(results[1])
                    #print("ell=",results[1])
                    eps.append(results[2])
                    eps_err.append(results[3])

                #print(len(result))
                #rrr.append(result)

            except StopIteration:
                break
            except Exception as e:
                # do something
                rrre.append(e)
        pool.terminate()

    f = open("processed_ellipticity_dumped_spiral",'w')
    for i in range(0,len(processed_ids)):
        f.write("{} {} {}\n".format(processed_ids[i],eps[i],eps_err[i]))
    f.close()
    #np.savetxt("processed_ellipticity",np.c_[processed_ids,eps,eps_err])


