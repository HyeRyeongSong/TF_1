import tensorflow as tf
import matplotlib.pyplot as plt

# 학습 data X와 Y가 주어짐
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# W를 tensorflow variable로 선언하고 random 값을 줌
# W가 잘 내려가는지 확인하기 위해 말도 안되는 큰 값을 W에 넣고 시작
W = tf.Variable(5.0)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
# 편의를 위해 hypothesis를 simple하게 줌
hypothesis = X * W

# cost/Losss function (cost 정의)
# tf.reduce_mean(t): tensor들의 평균을 내주는 함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 우리가 복잡하게 선언했던 것을 이 함수를 통해 간단히 표현 가능
# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a session. (session을 만들고)
sess = tf.Session()
# Initializes global variables in the graph. (tensorflow variable들을 initialize해줌)
# 앞서 정의한 tensorflow variable인 W와 b를 사용하기 위해서는 이함수를 실행시켜서 variable들을 initialize해줘야 함
sess.run(tf.global_variables_initializer())

for step in range(10):
    print(step, sess.run(W))
    sess.run(train, feed_dict={X: x_data, Y: y_data})