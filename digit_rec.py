import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dropout

from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from sklearn.metrics import accuracy_score

df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_train.head()

sns.countplot(x = df_train['label'], color= 'green')

#train_dataset = df_train.sample(frac=0.8, random_state=0)
#test_dataset = df_train.drop(train_dataset.index)

#train_label = train_dataset['label']
#test_label = test_dataset['label']
#train_dataset.drop('label',axis=1,inplace=True)
#test_dataset.drop('label',axis=1,inplace=True)
train_label = df_train['label']
df_train.drop('label',axis=1,inplace=True)


def data_preprocessing(raw):
    num_images = raw.shape[0]
    x_as_array = raw.values[:,0:]
    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)
    out_x = x_shaped_array / 255
    return out_x
#final_train = data_preprocessing(train_dataset)
#final_test = data_preprocessing(test_dataset)
final_train = data_preprocessing(df_train)
final_train.shape



model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))

#model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(final_train, train_label, epochs=10)
#validation_data=(final_test, test_label)

#plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
#plt.legend(loc='lower right')

#test_loss, test_acc = model.evaluate(final_test,  test_label, verbose=2)
#print(test_acc)

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
df_test.head()
df_test_img = data_preprocessing(df_test)

predictions = model.predict(df_test_img)
score = tf.nn.softmax(predictions)

sbmt_label = []
for i in range(len(score)):
    x = np.argmax(score[i])
    sbmt_label.append(x)

submit_res = pd.DataFrame(columns=['label'],data=sbmt_label)
submit_res.head()
submit_id = pd.DataFrame(columns=['ImageId'], data=(x for x in range(1,(len(submit_res))+1)))
submit_id.tail()
submit_final = pd.concat([submit_id,submit_res],axis=1)
submit_final.to_csv('submission.csv',index=False)