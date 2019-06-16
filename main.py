import tensorflow as tf
import matplotlib.pyplot as plt

# 학습 data X와 Y가 주어짐
X = [1, 2, 3]
Y = [1, 2, 3]

# W를 placeholder로 줌
# W의 값을 임의대로 바꿔가면서 그 값이 어떻게 되는지 보기위함
W = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
# 편의를 위해 hypothesis를 simple하게 줌
hypothesis = X * W

# cost/Losss function (cost 정의)
# tf.reduce_mean(t): tensor들의 평균을 내주는 함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Launch the graph in a session. (session을 만들고)
sess = tf.Session()
# Initializes global variables in the graph. (tensorflow variable들을 initialize해줌)
# 앞서 정의한 tensorflow variable인 W와 b를 사용하기 위해서는 이함수를 실행시켜서 variable들을 initialize해줘야 함
sess.run(tf.global_variables_initializer())

# Variables for plotting cost function
# Graph를 실행시키면서 W의 값과 cost의 값을 저장할 list를 만들어줌
W_val = []
cost_val = []

# range를 (-30 ~ 50)까지 이동시키면서
for i in range(-30, 50):
    # W를 (-3 ~ 5)까지 0.1 간격으로 작게 움직이겠다
    feed_W = i * 0.1
    # feed_dict를 통해 W에 변동된 값들을 전달함
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    # W와 cost가 어떻게 변하는지를 curr_W와 curr_cost에 저장하고
    # 이를 각각 cost_val, W_val 리스트에 넣어준다
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# cost 함수가 어떤 모양인지 시각화를 해보자
# Show the cost function
# plt.plot(x축, y축)
plt.plot(W_val, cost_val)
plt.show()