#To use#

get_data(): return the matrix images with shape of (50, 32, 32, 3), which is 50 images, each image has a dimension of 32x32, and 3 channels of RGB values.

get_labels(): return the matrix of labels with shape of (50, 10), which is 50 labels, each label is a vector with 10 length.

show_image(part3_data, i): show the image with index i

#Code Demo#
```
import assign4_part3_images_reader as img_reader
part3_data = img_reader.get_data()
part3_labels = img_reader.get_labels()
print("part3_data.shape: ", part3_data.shape)
print("part3_labels.shape: ", part3_labels.shape)
img_reader.show_image(part3_data, 3) #show the image with index 3
```

![Demo](./demo.png?)
