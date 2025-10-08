import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

st.set_page_config(page_title="MNIST Demo", page_icon="🔢", layout="centered")

@st.cache_data(show_spinner=True)
def load_mnist():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    # scale and reshape
    X_train = (X_train.astype("float32") / 255.0).reshape(-1, 28*28)
    X_test = (X_test.astype("float32") / 255.0).reshape(-1, 28*28)
    return (X_train, y_train), (X_test, y_test)

@st.cache_resource(show_spinner=True)
def build_and_train(X_train, y_train):
    model = keras.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=2, verbose=0)
    return model

st.title("MNIST Digit Classification 🔢")
try:
    (X_train, y_train), (X_test, y_test) = load_mnist()
except Exception as e:
    st.error("Couldn’t download MNIST. If you’re on a restricted network, run once locally with internet so it caches.")
    st.exception(e)
    st.stop()

st.write("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

# show a sample image
fig, ax = plt.subplots()
ax.imshow(X_test[0].reshape(28,28), cmap="gray")
ax.axis("off")
st.pyplot(fig)

model = build_and_train(X_train, y_train)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
st.write("Evaluation [loss, acc]:", [float(loss), float(acc)])

# predictions + confusion matrix
y_pred = model.predict(X_test, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)

st.write("First 5 predictions:", y_pred_labels[:5].tolist())

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_labels).numpy()

fig2, ax2 = plt.subplots()
im = ax2.imshow(cm, interpolation="nearest")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("True")
st.pyplot(fig2)
