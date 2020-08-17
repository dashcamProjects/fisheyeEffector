import argparse
import io
import os
import sys
import numpy as np
from PIL import Image
from math import sqrt

class FisheyeEffector:
    def __init__(self, height=720, width=1280, distortion=0.5):
        float_height, float_width = float(height), float(width)
        self.height, self.width = height, width

        self.filter = np.full((height, width, 2), -1)
        self.crop = distortion > 0
        self.left, self.upper, self.right, self.lower = 0, 0, width, height

        # calculate filter
        for h in range(height):
            for w in range(width):
                norm_h, norm_w = float((2*h - float_height) / float_height), float((2*w - float_width) / float_width)
                diagonal = norm_h - norm_w == 0
                norm_h = norm_h * float_height / float_width

                radius = sqrt(norm_h**2 + norm_w**2)
                org_norm_h, org_norm_w = calc_points_of_original_image(norm_h, norm_w, radius, distortion)

                org_norm_h = org_norm_h * float_width / float_height
                org_h, org_w = int((org_norm_h + 1) * float_height / 2), int((org_norm_w + 1) * float_width / 2)

                if org_h in range(height) and org_w in range(width):
                    self.filter[h][w] = [org_h, org_w]

                    # remember coordinates for cropping result images
                    if diagonal:
                        if self.left == 0 and self.upper == 0:
                            self.left, self.upper = w, h
                        self.right, self.lower = w, h

    def apply(self, image_bytes):
        image = np.array(Image.open(io.BytesIO(image_bytes)))
        fish_image = np.zeros_like(image)

        # apply filter
        for h in range(len(fish_image)):
            for w in range(len(fish_image[0])):
                org_h, org_w = self.filter[h][w]
                if org_h >= 0 and org_w >= 0:
                    fish_image[h][w] = image[org_h][org_w]

        fish_image = Image.fromarray(fish_image)
        if self.crop:
            fish_image = fish_image.crop((self.left, self.upper, self.right, self.lower))
            fish_image = fish_image.resize((self.width, self.height), Image.LANCZOS)

        fish_image_bytes = io.BytesIO()
        fish_image.save(fish_image_bytes, 'png')

        return fish_image_bytes.getvalue()

def calc_points_of_original_image(x, y, r, distortion):
    if distortion > 1:
        distortion = 1
    elif distortion < -1:
        distortion = -1

    if 1 - distortion*(r**2) == 0:
        return x, y

    return x / (1 - distortion*(r**2)), y / (1 - distortion*(r**2))

def execDir(effector, path):
    for file in os.listdir(path):
        joined_path = os.path.join(path, file)

        if os.path.isfile(joined_path):
            _, ext = os.path.splitext(joined_path)
            output_path = joined_path.replace(ext, '_fish.png')
            execFile(effector, joined_path, output_path=output_path)

        else:
            execDir(effector, joined_path)

def execFile(effector, input_path, output_path='output.png'):
    if input_path.endswith('_fish.png'):
        print('skip', input_path)
        return

    if os.path.exists(output_path):
        print('skip', input_path)
        return

    with open(input_path, 'rb') as image_bin:
        print(input_path)
        output = open(output_path, 'wb')
        output.write(effector.apply(image_bin.read()))
        output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help='path to the input image or directory')
    parser.add_argument('-d', '--distortion', type=float, default=0.1, help='amount of distortion between -1 to 1 (0.1 as default)')
    parser.add_argument('--width', type=int, default=1280, help='input image width (1280 as default)')
    parser.add_argument('--height', type=int, default=720, help='input image height (720 as default)')

    args       = parser.parse_args()
    input_path = args.input
    distortion = args.distortion
    width      = args.width
    height     = args.height

    if not os.path.exists(input_path):
        print('No such file or directory: {}'.format(input_path))
        exit(1)

    effector = FisheyeEffector(height=height, width=width, distortion=distortion)

    if os.path.isfile(input_path):
        execFile(effector, input_path)

    else:
        execDir(effector, input_path)
