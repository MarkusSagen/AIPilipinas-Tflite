import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import configs
from tflite_model_maker import ExportFormat
from tflite_model_maker import image_classifier
from tflite_model_maker import ImageClassifierDataLoader
from tflite_model_maker import model_spec

import matplotlib.pyplot as plt
image_path = tf.keras.utils.get_file(
      'flower_photos',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      cache_dir=".",
      untar=True)
data = ImageClassifierDataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
# Create a custom image classifier model based on the loaded data. The default model is EfficientNet-Lite0.
#model = image_classifier.create(train_data, validation_data=model_spec.mobilenet_v2_spec(), epochs=10)
model = image_classifier.create(train_data)
print(model.summary())
loss, accuracy = model.evaluate(test_data)
# print(accuracy)
# A helper function that returns 'red'/'black' depending on if its two input
# parameter matches or not.


def get_label_color(val1, val2):
  if val1 == val2:
    return 'black'
  else:
    return 'red'

# Then plot 100 test images and their predicted labels.
# If a prediction result is different from the label provided label in "test"
# dataset, we will highlight it in red color.


plt.figure(figsize=(20, 20))
predicts = model.predict_top_k(test_data)
for i, (image, label) in enumerate(test_data.dataset.take(100)):
  ax = plt.subplot(10, 10, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image.numpy(), cmap=plt.cm.gray)

  predict_label = predicts[i][0][0]
  color = get_label_color(predict_label,
                          test_data.index_to_label[label.numpy()])
  ax.xaxis.label.set_color(color)
  plt.xlabel('Predicted: %s' % predict_label)
plt.show()

model.export(export_dir='.', export_format=ExportFormat.TFLITE, with_metadata=False)
#model.evaluate_tflite('model.tflite', test_data)