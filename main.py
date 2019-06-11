import tensorflow as tf

#placeholder 노드 2개와 이를 더해주는 노드를 만들고 feed-dict를 통해 placeholder의 값을 넘겨받음
#feed-dict: 이 그래프를 실행시키고 싶은데 placeholder 노드의 값을 모르겠으니 그 값들을 좀 넘겨줘
#값을 넘겨주면서 그래프를 실행시키면 됨
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)

#session을 만들고 이 session을 실행시킴
sess = tf.Session()
print(sess.run(adder_node, feed_dict = {a: 3, b: 4.5}))
#array를 통해 placeholder에 여러개의 값들을 넘겨줄 수도 있음
print(sess.run(adder_node, feed_dict = {a: [1, 3], b: [2, 4]}))