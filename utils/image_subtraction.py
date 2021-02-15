from PIL import Image
from PIL import ImageChops
import argparse
 

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', required=True, metavar='/path/to/img1', help="Path to first image")
    parser.add_argument('--img2', required=True, metavar='/path/to/img2', help="Path to second image")
    parser.add_argument('--diff', required=True, metavar='/path/to/img_diff', help="Path to save image difference")
    return parser.parse_args()

args = parser()

im1 = Image.open(args.img1) 
im2 = Image.open(args.img2)

diff = ImageChops.difference(im2, im1)

diff.save(args.diff)