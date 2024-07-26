
import keras
import time
import datetime
import tensorflow as tf
tf.compat.v1.disable_eager_execution() # disable eager 오류 해결을 위함

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist

start = time.time()

# 1단계 > MNIST 데이터셋을 로드
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# 2단계 > 모델 생성
model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(23, 23, 4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(
    loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=["accuracy"]
)

# 3단계 > ART classifier 생성
classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

# 4단계 > ART classifier 훈련
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=20)

# 5단계 > ART classifier 정확도 측정
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("정상적으로 학습시킨 MNIST 모델의 정확도: {}%".format(accuracy * 100))

# 6단계 > 적대적 예제 생성
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

# 7단계 > 적대적 예제에 대한 ART classifier 평가
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("MNIST에 FGSM 공격을 가한 후 정확도: {}%".format(accuracy * 100))

sec = time.time() - start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print(times)