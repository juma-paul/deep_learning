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
n = int(input('Number of units for hidden layers: '))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n),
    tf.keras.layers.Dense(2)
])