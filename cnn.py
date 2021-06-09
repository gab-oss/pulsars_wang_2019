import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

train_images = np.load("../data/sdss_training_images.npy")
print("train_images.shape = {}, train_images.min = {}, train_images.max = {}".format(train_images.shape, train_images.min(), train_images.max()))

train_labels = np.load("../data/sdss_training_labels.npy")
print("train_labels.shape = {}, train_labels.min = {}, train_labels.max = {}".format(train_labels.shape, train_labels.min(), train_labels.max()))

test_images = np.load("../data/sdss_test_images.npy")
test_labels = np.load("../data/sdss_test_labels.npy")

train_images = np.transpose(train_images, (0, 2, 3, 1))
test_images = np.transpose(test_images, (0, 2, 3, 1))
# train_images = tf.image.rot90(train_images, k=np.random.randint(4))
# if np.random.uniform() > 0.5:
#     train_images = tf.image.flip_left_right(train_images)
# train_images = tf.image.crop_to_bounding_box(
#     train_images, np.random.randint(4 + 1), np.random.randint(4 + 1), 44, 44)

print("train_images.shape = {}".format(train_images.shape))

def renormalize(array):
    return (array - array.min()) / (array.max() - array.min())

for i in range(5):
    train_images[:, i, :, :] = renormalize(train_images[:, i, :, :])

train_labels = renormalize(train_labels).astype(np.int32)

for i in range(5):
    test_images[:, i, :, :] = renormalize(test_images[:, i, :, :])

test_labels = renormalize(test_labels).astype(np.int32)

class DataAugmentation(tf.keras.layers.Layer):
    def __init__(self, flip_chance=0.5, 
                 pixels_to_crop=4, **kwargs):
        super(DataAugmentation, self).__init__(**kwargs)
        self.flip_chance = flip_chance
        self.pixels_to_crop = pixels_to_crop

    def call(self, images, training=True):
            if not training:
                return images

            images = tf.image.rot90(images, k=np.random.randint(4))
            if np.random.uniform() > self.flip_chance:
                images = tf.image.flip_left_right(images)
            images = tf.image.crop_to_bounding_box(
                images, np.random.randint(self.pixels_to_crop + 1), np.random.randint(self.pixels_to_crop + 1), 44, 44)
            return images


k_initializer = tf.keras.initializers.Orthogonal(np.sqrt(2 / (1 + 0.01**2)))

model = models.Sequential()

model.add(tf.keras.layers.InputLayer(
    input_shape=(48, 48, 5), batch_size=10))
model.add(DataAugmentation())
model.add(layers.GaussianNoise(stddev=0.2))

model.add(layers.Conv2D(32, (5, 5), activation=tf.keras.layers.LeakyReLU(), kernel_initializer=k_initializer, bias_initializer=tf.keras.initializers.Constant(0.1)))
model.add(layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(), padding="same", kernel_initializer=k_initializer, bias_initializer=tf.keras.initializers.Constant(0.1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(), padding="same", kernel_initializer=k_initializer, bias_initializer=tf.keras.initializers.Constant(0.1)))
model.add(layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(), padding="same", kernel_initializer=k_initializer, bias_initializer=tf.keras.initializers.Constant(0.1)))
model.add(layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(), padding="same", kernel_initializer=k_initializer, bias_initializer=tf.keras.initializers.Constant(0.1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(), padding="same", kernel_initializer=k_initializer, bias_initializer=tf.keras.initializers.Constant(0.1)))
model.add(layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(), padding="same", kernel_initializer=k_initializer, bias_initializer=tf.keras.initializers.Constant(0.1)))
model.add(layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(), padding="same", kernel_initializer=k_initializer, bias_initializer=tf.keras.initializers.Constant(0.1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(2048, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=k_initializer, bias_initializer=tf.keras.initializers.Constant(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2048, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=k_initializer, bias_initializer=tf.keras.initializers.Constant(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation=tf.keras.activations.softmax, kernel_initializer=k_initializer, bias_initializer=tf.keras.initializers.Constant(0.01)))

model.summary()

starter_learning_rate = 0.003
end_learning_rate = 0.0001
decay_steps = 50
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=1)

model.compile(optimizer=tf.keras.optimizers.SGD(
                  learning_rate=learning_rate_fn),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            #   nesterov=True,
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
