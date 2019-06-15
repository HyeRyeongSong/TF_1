import tensorflow as tf

# X and Y data (학습할 data가 주어졌고)
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# trainable한 tensorflow의 variable 선언
# W와 b의 값을 모르니까 random한 값을 주게 됨
# tensorflow의 shape을 정의하고 그 값을 준다
# (값이 하나인 1차원 array(rank가 1)를 주게 됨)
# tf.Variable: tensorflow가 사용하는 variable
#              trainable variable(tensorflow가 학습하는 과정에서 자기가 변경시킨다)
#              (tensorflow를 실행시키면 tensorflow 자체적으로 변경시키는 값)
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# Our hypothesis XW+b (hypothesis 정의)
hypothesis = x_train * W + b

# cost/Losss function (cost 정의)
# tf.reduce_mean(t): tensor들의 평균을 내주는 함수
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize(train할 때 cost를 minimize하라 해줌)
# 1.optimizer를 GradientDescentOptimizer를 통해 정의하고
# 2.optimizer의 minimize라는 함수를 호출하면서 무엇을 minimize할 지 넘겨줌
# 3.optimizer가 우리가 앞서 정의한 tensorflow variable W와 b를 조정해서 자기가 스스로 minimize함
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session. (session을 만들고)
sess = tf.Session()
# Initializes global variables in the graph. (tensorflow variable들을 initialize해줌)
# 앞서 정의한 tensorflow variable인 W와 b를 사용하기 위해서는 이함수를 실행시켜서 variable들을 initialize해줘야 함
sess.run(tf.global_variables_initializer())

# Fit the line
# 2000번 정도 step을 돌텐데 이때마다 출력하긴 힘이 드니까 한 20번에 한 번씩 출력해줌

# train노드를 실행시켜야 cost를 minimize하고
# (cost를 minimize) = tf.reduce_mean(tf.square(hypothesis - y_train))을 minimize
# 이 hypothesis는 W와 b로 연결됨

# train 노드를 실행 = 이 그래프를 따라 들어가서 결국은 W와 b에 어떤 값들을 저장할 수 있게
# 다 연결이 되어있게 building이 된 Grapp를 실행한다는 의미

for step  in range(2001):
    sess.run(train) # train 노드를 실행(이때 학습이 일어남)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b)) # 학습이 일어난 뒤에 cost, W, b의 값이 어떻게 되는지 관찰