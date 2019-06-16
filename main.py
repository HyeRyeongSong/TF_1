import tensorflow as tf
import matplotlib.pyplot as plt

# 학습 data X와 Y가 주어짐
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# W를 tensorflow variable로 선언하고 random 값을 줌
W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
# 편의를 위해 hypothesis를 simple하게 줌
hypothesis = X * W

# cost/Losss function (cost 정의)
# tf.reduce_mean(t): tensor들의 평균을 내주는 함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 수동으로 cost를 minimize하는 코드
# 수동으로 구현한 Gradient descent 알고리즘
# 수식을 그대로 써주면 미분을 이용한 Gradient descent 알고리즘을 tensorflow로 구현 가능
# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
# tensorflow는 '='으로 바로 assign 불가 (assign 함수를 통해 이뤄져야 함)
# assign하는 operation을 update라는 노드에 할당해줌
# 뒤에서 update 노드를 실행시키면 Graph 전체가 실행되며 위의 일련의 동작들이 일어남
update = W.assign(descent)

# Launch the graph in a session. (session을 만들고)
sess = tf.Session()
# Initializes global variables in the graph. (tensorflow variable들을 initialize해줌)
# 앞서 정의한 tensorflow variable인 W와 b를 사용하기 위해서는 이함수를 실행시켜서 variable들을 initialize해줘야 함
sess.run(tf.global_variables_initializer())

for step in range(21):
    # update 노드를 실행시킴(Graph 실행)
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    # 잘 실행이 되는 지 cost와 W 값 출력 (W가 1이 되야함)
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))