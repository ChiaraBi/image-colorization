import matplotlib.pyplot as plt
import torch

from colorization import colorizers
from colorization.colorizers.util import load_img, preprocess_img, postprocess_tens

colorizer_eccv16 = colorizers.eccv16().eval()
# colorizer_eccv16.cuda() # uncomment this if you're using GPU

img_path = '../img/ansel_adams3.jpeg'
img = load_img(img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
# tens_l_rs = tens_l_rs.cuda() # uncomment this if you're using GPU

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig, 0*tens_l_orig), dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

out_path = '../img/ansel_adams3-color.jpeg'
plt.imsave(out_path, out_img_eccv16)
