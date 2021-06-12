import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

def renormalize(array):
    return (array - array.min()) / (array.max() - array.min())

X = np.load("../data/sdss_training_images.npy")
print("X.shape = {}, X.min = {}, X.max = {}".format(X.shape, X.min(), X.max()))

y = np.load("../data/sdss_training_labels.npy")
print("y.shape = {}, y.min = {}, y.max = {}".format(y.shape, y.min(), y.max()))

train_images = X[:80]
train_labels = y[:80]

for i in range(5):
    train_images[:, i, :, :] = renormalize(train_images[:, i, :, :])

train_images = np.transpose(train_images, (0, 2, 3, 1))
train_labels = renormalize(train_labels).astype(np.int32)

valid_images = X[-20:]
valid_labels = y[-20:]

for i in range(5):
    valid_images[:, i, :, :] = renormalize(valid_images[:, i, :, :])

valid_images = np.transpose(valid_images, (0, 2, 3, 1))
valid_labels = renormalize(valid_labels).astype(np.int32)

test_images = np.load("../data/sdss_test_images.npy")
test_labels = np.load("../data/sdss_test_labels.npy")

for i in range(5):
    test_images[:, i, :, :] = renormalize(test_images[:, i, :, :])

test_images = np.transpose(test_images, (0, 2, 3, 1))
test_labels = renormalize(test_labels).astype(np.int32)

print("train_images.shape = {}".format(train_images.shape))
print("valid_images.shape = {}".format(valid_images.shape))
print("test_images.shape = {}".format(test_images.shape))

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

# rozszerz dane przeksztalceniami
# przelosuj obrazki i labelki
# dopiero potem rozdziel na trening i walidacje
# 64 zbiory na *testy*


k_initializer = tf.keras.initializers.Orthogonal(np.sqrt(2 / (1 + 0.01**2)))

model = models.Sequential()

model.add(tf.keras.layers.InputLayer(input_shape=(48, 48, 5), batch_size=10))
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
decay_steps = 20
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

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=360,
    width_shift_range=4,
    height_shift_range=4,
    horizontal_flip=True,)


history = model.fit(train_datagen.flow(train_images, train_labels,
                                       batch_size=10, 
                                       seed=27, shuffle=False), 
                    epochs=20, 
                    validation_data=(valid_images, valid_labels))

history.history.keys()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

def data_augmentation(images):
    for startx in range(4):
        for starty in range(4):
            for rotate in range(4):
                aug_images = np.ndarray.copy(images)
                aug_images = tf.image.rot90(aug_images, k=rotate)
                aug_images = tf.image.crop_to_bounding_box(
                    aug_images, startx, starty, 44, 44)

                y_pred = model.predict(aug_images, batch_size=10)
                print(y_pred)

data_augmentation(test_images)


