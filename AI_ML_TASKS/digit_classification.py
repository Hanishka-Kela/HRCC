import matplotlib
matplotlib.use("Agg")

import streamlit as st
try:
    get_ipython # type: ignore
except NameError:
    def get_ipython():
        class D:
            def run_line_magic(self, *a, **k): pass
        return D()

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

(X_train,y_train) , (X_test,y_test) = keras.datasets.mnist.load_data()

X_train.shape
st.write("X_train.shape:", X_train.shape)

plt.matshow(X_train[1])
st.pyplot(plt)

X_train_flattened = X_train.reshape(len(X_train),28*28)
X_test_flattened = X_test.reshape(len(X_test),28*28)
X_train_flattened.shape
st.write("X_train_flattened.shape:", X_train_flattened.shape)

X_test_flattened.shape
st.write("X_test_flattened.shape:", X_test_flattened.shape)

model = keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss= 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train_flattened,y_train,epochs=5)

model.evaluate(X_test_flattened,y_test)
eval_result_1 = model.evaluate(X_test_flattened,y_test,verbose=0)
st.write("Evaluation (model 1) [loss, acc]:", eval_result_1)

plt.matshow(X_test[0])
st.pyplot(plt)

y_predicted= model.predict(X_test_flattened)
y_predicted[0]
st.write("y_predicted[0]:", y_predicted[0])

print(np.argmax(y_predicted[0]))
st.write("argmax(y_predicted[0]):", int(np.argmax(y_predicted[0])))

y_predicted_labels=[np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]
st.write("First 5 predicted labels (model 1):", y_predicted_labels[:5])

cm=tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
cm
st.write("Confusion matrix (model 1):")
st.write(cm.numpy())

import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
st.pyplot(plt)

model = keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation = 'relu'),
    keras.layers.Dense(10,activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss= 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train_flattened,y_train,epochs=5) 

model.evaluate(X_test_flattened,y_test)
eval_result_2 = model.evaluate(X_test_flattened,y_test,verbose=0)
st.write("Evaluation (model 2) [loss, acc]:", eval_result_2)

y_predicted= model.predict(X_test_flattened)
y_predicted_labels=[np.argmax(i) for i in y_predicted]
cm=tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
st.pyplot(plt)
