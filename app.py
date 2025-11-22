import tensorflow as tf
from tensorflow.keras import layers, models 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import LeNet5
from tensorflow.keras.optimizers import Adam

(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()

X_train = tf.expand_dims(X_train,axis=-1)
X_test = tf.expand_dims(X_test,axis=-1)

X_train = tf.image.resize(X_train,(32,32))
X_test = tf.image.resize(X_test,(32,32))

X_train = X_train/255.0
X_test = X_test/255.0

model = LeNet5()
model.compile(
    optimizer = Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

history = model.fit(
    X_train,y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.1
)

test_loss, test_acc = model.evaluate(X_test,y_test)
print("Test Accuracy: ",test_acc)
