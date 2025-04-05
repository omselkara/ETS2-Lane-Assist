from keras import models,layers,regularizers
import tensorflow as tf
import numpy as np
import cv2,time
from PIL import Image
from tqdm import tqdm
from colorama import Fore, Style, init


images = []
outputs = []

file = open("dataset/data.txt","r")
data = file.readlines()
file.close()

interval = len(data)//500
if interval==0:
    interval = 1
for i in range(len(data)):
    if i%interval==0:
        print(f"%{100*i/len(data)}")
    img = cv2.imread(f"dataset/images/image{i}.png", cv2.IMREAD_GRAYSCALE)
    #img = advanced_edge_detection(img)
    #img = cv2.resize(img,(335,95))
    images.append(img)
    outs = data[i].rstrip(" \n").split(" ")
    outputs.append(np.array([100*min(1,max(-1,float(outs[4])))]))
images = np.array(images)/255.0
outputs = np.array(outputs)

# Shuffle the data
indices = np.arange(len(images))
np.random.shuffle(indices)

images = images[indices]
outputs = outputs[indices]

train_images = images[1000:]
train_outputs = outputs[1000:]

test_images = images[:1000]
test_outputs = outputs[:1000]


image_input = layers.Input(shape=(95, 335, 1), name='image_input')

x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.4)(x)

x = layers.Conv2D(256, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.5)(x)

x = layers.Conv2D(512, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.5)(x)

x = layers.Flatten()(x)

combined = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
combined = layers.Dropout(0.5)(combined)
combined = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined)
combined = layers.Dropout(0.5)(combined)
combined = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined)

output = layers.Dense(1, activation="linear")(combined)

model = models.Model(inputs=image_input, outputs=output)

model = models.load_model("trained_model.keras")

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
loss_fn = tf.keras.losses.MeanSquaredError(name='mse_loss')
mae_metric = tf.keras.metrics.MeanAbsoluteError(name='mae_metric')

model.compile(optimizer=optimizer, loss=loss_fn, metrics=[mae_metric])
model.summary()

@tf.function
def train_step(image_batch, label_batch):
    with tf.GradientTape() as tape:
        predictions = model(image_batch, training=True)
        loss = loss_fn(label_batch, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    mae_metric.update_state(label_batch, predictions)
    
    return loss, mae_metric.result()

batch_size = 32
num_samples = len(train_images)
num_batches = num_samples // batch_size
epochs = 25
##
##for epoch in range(epochs):
##    epoch_loss = 0
##    mae_metric.reset_state()
##    start_time = time.time()
##    
##    for i in range(0, num_samples, batch_size):
##        batch_images = train_images[i:i + batch_size]
##        batch_labels = train_outputs[i:i + batch_size]
##
##        loss, mae = train_step(batch_images, batch_labels)
##        epoch_loss += loss
##
##    epoch_loss /= num_batches
##    epoch_mae = mae_metric.result()
##    
##    # Test verisi üzerinde değerlendirme
##    test_results = model.evaluate(test_images, test_outputs, batch_size=batch_size, verbose=0)
##    test_loss, test_mae = test_results[0], test_results[1]
##    
##    print(f"Epoch: {epoch + 1}/{epochs} Time: {time.time()-start_time:.4f}")
##    print(f"Train Loss (MSE): {epoch_loss:.10f} Train MAE: {epoch_mae:.10f}")
##    print(f"Test Loss (MSE): {test_loss:.10f} Test MAE: {test_mae:.10f}")
##    print("-" * 50)

init()

for epoch in range(epochs):
    with tqdm(total=num_batches+1,desc=f"Epoch: {epoch}", ncols=200, bar_format="{desc} {percentage:3.0f}%|{bar}| Batch: {n_fmt}/{total_fmt} TPS: {rate_fmt} Elapsed: {elapsed} Remaining: {remaining} {postfix}") as pbar:
        pbar.set_description(f"Epoch: {epoch}")
        epoch_loss = 0
        mae_metric.reset_state()
        start_time = time.time()
        for i in range(0, num_samples, batch_size):
            batch_images = train_images[i:i + batch_size]
            batch_labels = train_outputs[i:i + batch_size]
            loss, mae = train_step(batch_images, batch_labels)
            epoch_loss += loss
            pbar.set_postfix_str(f"Train Loss (MSE): {(epoch_loss/(i//batch_size+1)):.10f} Train MAE: {mae_metric.result():.10f}")
            pbar.update(1)

        epoch_loss /= num_batches
        epoch_mae = mae_metric.result()
        
        # Test verisi üzerinde değerlendirme
        test_results = model.evaluate(test_images, test_outputs, batch_size=batch_size, verbose=0)
        test_loss, test_mae = test_results[0], test_results[1]
        
    print(f"Epoch: {epoch + 1}/{epochs} Time: {time.time()-start_time:.4f}")
    print(f"Train Loss (MSE): {epoch_loss:.10f} Train MAE: {epoch_mae:.10f}")
    print(f"Test Loss (MSE): {test_loss:.10f} Test MAE: {test_mae:.10f}")
    print("-" * 50)
        

model.save("trained_model.keras")

