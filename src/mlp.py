import tensorflow as tf


class MLP(object):
    def __init__(self, layers_names=['input', 'output'], neurons_num=[784, 2]):
        self.name = 'MLP'
        self.layers_name = layers_names
        self.neurons_num = neurons_num
        self.args = {}
        for index, layer in enumerate(layers_names):
            self.args[layer] = neurons_num[index]
        self.x = tf.placeholder(tf.float32, [None, self.neurons_num[0]])
        self.Y = tf.placeholder(tf.float32, [None, self.neurons_num[-1]])
        self.weights = {}
        self.biases = {}
        self.layer_output = {}
        self.logits = None
        self.loss_op = None
        self.optimizer = None
        self.train_op = None

    def build_model(self):
        for index, layer in enumerate(self.layers_name[1:]):
            self.weights[layer] = tf.Variable(tf.random_normal([self.neurons_num[index], self.neurons_num[index+1]]))
            self.biases[layer] = tf.Variable(tf.random_normal([self.neurons_num[index+1]]))

        # Hidden fully connected layer former layers' neurons
        for index, layer in enumerate(self.layers_name[1:]):
            if index == 0:
                self.layer_output[layer] = tf.add(tf.matmul(self.x, self.weights[layer]), self.biases[layer])
            else:
                self.layer_output[layer] = tf.add(tf.matmul(self.layer_output[self.layers_name[index]],
                                                            self.weights[layer]), self.biases[layer])

    def train_model(self, dataset, tf_sess, learning_rate=0.001, training_epochs=15, batch_size=100, display_step=1):
        # build model
        self.build_model()

        # final output of mlp
        self.logits = self.layer_output[self.layers_name[-1]]

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Test model
        pred = tf.nn.softmax(self.logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))

        # Initializing the variables
        init = tf.global_variables_initializer()

        tf_sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(dataset.train.num_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([self.train_op, self.loss_op], feed_dict={self.x: batch_x,
                                                                self.Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        pred = tf.nn.softmax(self.logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({self.x: mnist.test.images, self.Y: mnist.test.labels}))


if __name__ == '__main__':
    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    layers_name = ['input', 'h1', 'output']
    neurons_num = [784, 256, 10]
    with tf.Graph().as_default():
        with tf.Session() as sess:
            mlp = MLP(layers_names=layers_name, neurons_num=neurons_num)
            mlp.train_model(mnist, sess, training_epochs=500)
