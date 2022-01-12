import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. 파이썬으로 구현하기

# 1.1 cost 함수 만들기
X = np.array([1,2,3])
Y = np.array([1,2,3])

def cost_func(W,X,Y):
    c = 0
    for i in range(len(X)):
        c += np.square(W * X[i] - Y[i])
    return c / len(X)
  
# 1.2 비용 계산하기 - W_value 정해줌
cost_values = []
W_values = np.linspace(-3,5,num=15)
print("{:>6} | {:>10}".format("W","cost"))

for feed_W in W_values:
    curr_cost = cost_func(feed_W, X, Y)
    cost_values.append(curr_cost)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))
    
# 1.3 cost 함수 개형 확인
plt.figure(figsize = (10,10))
plt.plot(W_values, cost_values, "b")
plt.ylabel('Cost(W)')
plt.xlabel('W')
plt.show()


# 2. 텐서플로우로 구현하기 - W_value 정해줌

X = np.array([1,2,3])
Y = np.array([1,2,3])
print("{:>6} | {:>10}".format("W","cost"))

def cost_func_tf(W,X,Y):
    hypothesis = W*X
    return tf.reduce_mean(tf.square(hypothesis-Y))

cost_values = []
W_values = np.linspace(-3,5,num=15)

for feed_W in W_values:
    curr_cost = cost_func_tf(feed_W, X, Y)
    cost_values.append(curr_cost)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))
    
plt.figure(figsize = (10,10))
plt.plot(W_values, cost_values, "b")
plt.ylabel('Cost(W)')
plt.xlabel('W')
plt.show()


# 3. Gradient Descent

# 3.1 W_value 난수 생성 / 업데이트
tf.random.set_seed(0)

X = np.array([1,2,3])
Y = np.array([1,2,3])
print("{:>5} | {:>10} | {:>10}".format("step", "cost", "W"))

W = tf.Variable(tf.random.normal([1], -100., 100))

for step in range(300):
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(hypothesis - Y, X))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)
    
    if step % 10 == 0:
        print("{:5} | {:10.4f} | {:10.6f}".format(step, cost.numpy(), W.numpy()[0]))

# 3.2 W_value tf.Variable() 로 생성 / 업데이트
X = np.array([1,2,3])
Y = np.array([1,2,3])
print("{:>5} | {:>10} | {:>10}".format("step", "cost", "W"))

W = tf.Variable(5.0)

for step in range(300):
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(hypothesis - Y, X))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)
    
    if step % 10 == 0:
        print("{:5} | {:10.4f} | {:10.6f}".format(step, cost.numpy(), W.numpy()))
