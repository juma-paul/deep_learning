import tensorflow as tf

# 1. Create Dense layer from scartch

class DenseLayer(tf.keras.layer.Layer):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()

        # Initailize weights and bias
        self.weights = self.add_weight([input_dim, output_dim])
        self.bias = self.add_weight([1, output_dim])

    def call(self, inputs):
        # Forward propagate the inputs
        z = tf.matmul(inputs, self.weights) + self.bias

        # Feed through a non-linear activation
        output = tf.math.sigmoid(z)

        return output

# 2. Create a Dense Layer using tensorflow

layer = tf.keras.layers.Dense(
    units=2
)

# 3. Multi output perceptron
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n=0),
    tf.keras.layers.Dense(2)
])

# 4. Deep Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n1=0),
    tf.keras.layers.Dense(n2=0),
    # .
    # .
    # .
    tf.keras.layers.Dense(2)
])

# Binary Cross Entropy Loss (Classification problems)
y = 'actual value'
predicted = 'predicted value'
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, predicted))

# Mean Squared Error Loss (Regression problems)
loss = tf.reduce_mean(tf.square(tf.subtract(y, predicted)))
loss = tf.keras.losses.MSE(y, predicted)

# Gradient Descent
def compute_loss(weights):
    return tf.reduce_mean(tf.square(tf.subtract(y, predicted)))

weights = tf.Variable([tf.random.normal()])
lr = tf.keras.optimizer.SGD()

while True:
    with tf.GradientTape() as g:
        loss = compute_loss(weights)
        gradient = g.gradient(loss, weights)

    weights = weights - lr * gradient
    break # shouldn't be here, this is a place for specifying convergence point (Check below)

# PUTTING IT ALL TOGETHER

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n1=0),
    tf.keras.layers.Dense(n2=0),
    tf.keras.layers.Dense(2)
])

optimizer = tf.keras.optimizer.SGD()

while True:
    prediction = model(x='inputs to model')

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(tf.subtract(y, predicted)))
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))