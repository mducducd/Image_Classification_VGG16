from lib import *
from config import *
from utils import *
from image_transform import ImageTransform
from pred import *

path = './test/honey-bee.jpg'
img = Image.open(path)
 
img = img.convert("RGB")
label = predict(img)

print(label)