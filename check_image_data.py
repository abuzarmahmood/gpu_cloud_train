from glob import glob
from skimage import io
from os import path
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed, cpu_count

data_dir = '/home/abuzarmahmood/Desktop/img_conv_net/test_data'
file_list = glob(path.join(data_dir,"*/*.jpg"))
file_list_split = np.array_split(file_list, cpu_count()-2)

def verify_image(img_file):
    try:
        img = io.imread(img_file)
        return True
    except:
        return False

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def verify_image_par(file_list):
    return [verify_image(x) for x in file_list]

dat_out = parallelize(verify_image_par, file_list_split) 
bad_imgs = [np.where(x==False)[0] for x in dat_out]

#dat_out = [verify_image(x) for x in tqdm(file_list)]

from PIL import Image

test = file_list[0]
v_image = Image.open(test)
v_image.verify()
