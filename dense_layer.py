import tensorflow as tf

class DenseLayer(tf.keras.layer.Layer):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()

        # Initailize weights and bias
        self.weights = self.add_weight([input_dim, output_dim])
        self.bias = self.add_weight([1, output_dim])