import matplotlib.pyplot as plt
import torch

from colorization import colorizers
from colorization.colorizers.util import load_img, preprocess_img, postprocess_tens

from os import listdir
from os.path import isfile, join

colorizer_eccv16 = colorizers.eccv16().eval()
# colorizer_eccv16.cuda() # uncomment this if you're using GPU

# images' paths
bnw_input_dir = '../img/original/BlackAndWhite/'
bnw_output_dir = '../img/colorized/zhang/BlackAndWhite/'

# BLACK AND WHITE
onlyfiles = [f for f in listdir(bnw_input_dir) if isfile(join(bnw_input_dir, f))]

for i in onlyfiles:
    img = load_img(join(bnw_input_dir, i))
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig, 0*tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_path = join(bnw_output_dir, i)
    plt.imsave(out_path, out_img_eccv16)

