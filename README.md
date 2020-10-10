# fisheyeEffector
Python class to apply fish-eye effects to the images.

### Usage
There are two ways for using this.
- Import the class into your program.
- Run independently.

#### Import the class into your program.
- Copy fisheye_effector.py into your project directory.
- Use it like as below.
```
from PIL import Image
from fisheye_effector import FisheyeEffector

effector = FisheyeEffector(height=720, width=1280, distortion=0.2) # distortion is between -1 and 1.

## either one is fine.
# with open('PATH/TO/YOUR/INPUT/IMAGE', 'rb') as image_bin:
#   output = open('PATH/TO/YOUR/OUTPUT/IMAGE', 'wb')
#   output.write(effector.apply(
#       image_bin.read()
#   ))
#   output.close()

image = Image.open('PATH/TO/YOUR/INPUT/IMAGE')
image = effector(image)
image.save('PATH/TO/YOUR/OUTPUT/IMAGE')
```

#### Run independently
- Run as follows.
```
python3 fisheye_effector.py PATH/TO/YOUR/INPUT -d 0.2 --height 720 --width 1280

# show help
python3 fisheye_effector.py -h
```
