from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
  rotation_range = 10,
  width_shift_range = .2,
  height_shift_range = .2,
  rescale = 1./255,
  shear_range = .2,
  zoom_range = .2,
  horizontal_flip = True,
  fill_mode = 'nearest'
)

i = 0
img = load_img('./CanBeer1.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

for batch in datagen.flow(x, batch_size = 32, save_to_dir = './temp', save_prefix = '', save_format = '.jpg'):
  i += 1
  if i > 400:
    break

