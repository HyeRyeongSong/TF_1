import tensorflow as tf
import matplotlib.pyplot as plt

# 학습 data X와 Y가 주어짐
X = [1, 2, 3]
Y = [1, 2, 3]

# W를 tensorflow variable로 선언하고 random 값을 줌
# Set wrong model weights
W = tf.Variable(5.0)

# Our hypothesis for linear model X * W
# 편의를 위해 hypothesis를 simple하게 줌
hypothesis = X * W

# 미분을 통해 수식적으로 계산한 gradient
# Manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# cost/Losss function (cost 정의)
# tf.reduce_mean(t): tensor들의 평균을 내주는 함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 우리가 복잡하게 선언했던 것을 이 함수를 통해 간단히 표현 가능
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# optimizer에서 바로 minimize하라고 하지 않고
# optimizer에게 이 cost에 맞는 gradient를 계산해줘라고 하면 gradient를 계산한 값을 돌려줌 (이를 gvs에 저장)
# Get gradients
gvs = optimizer.compute_gradients(cost, [W])

# 이 gradient를 계산한 값을 원하는대로 수정 가능
# (이 예제에서는 수정하지 않음) --> manual gradient와 같은 값이 나와야 함

# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

# Launch the graph in a session. (session을 만들고)
sess = tf.Session()
# Initializes global variables in the graph. (tensorflow variable들을 initialize해줌)
# 앞서 정의한 tensorflow variable인 W와 b를 사용하기 위해서는 이함수를 실행시켜서 variable들을 initialize해줘야 함
sess.run(tf.global_variables_initializer())

# 수식적으로 계산한 gradient와 GradientDescentOptimizer가 계산한 gradient가 같은지 비교해보자
# --> same
for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    # 각 step을 돌아갈 때마다 apply_gradients를 해서 학습을 시켜야 gradient, W, gvs의 값이 변함
    sess.run(apply_gradients)