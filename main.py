import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) #also tf.float32 implicitly
node3 = tf.add(node1, node2) #node3 = node1 + node2

#그냥 출력하면 이것은 그래프의 하나의 요소다.
#이런 형태의 Tensor야 라고만 말해주고 결과값이 나오진 않음
#print("node1: ", node1, "node2: ", node2)
#print("node2: ", node3)

#결과값을 출력하기 위해서는 session을 만들고 여기에 내가 실행시키고 싶은 노드를 넣어줘야 함
sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))