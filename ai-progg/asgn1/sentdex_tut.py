import tensorflow as tf
import mnist_basics as mn
import numpy as np
import tflowtools as tft

writer = tf.summary.FileWriter("hello")


images_raw, labels_raw = mn.load_mnist()

images = []
labels = []


for image in images_raw:
    t = np.reshape(image, (784))
    images.append(t)

for label in labels_raw:
    one_hot = tft.int_to_one_hot(int(label), 10)
    labels.append(one_hot)

train_images = images[:50000]
train_labels = labels[:50000]

test_images = images[-10000:]
test_labels = labels[-10000:]


n_nodes_hidden_layer1 = 200
n_nodes_hidden_layer2 = 200
n_nodes_hidden_layer3 = 100

n_classes = 10
batch_size = 64

x = tf.placeholder('float', [None, 784], name='x')
y = tf.placeholder('float', name='y')


def neural_network_model(data):

    #Defines the shape of the different layers
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hidden_layer1]), name = 'Weights'),
                       'biases': tf.Variable(tf.random_normal([n_nodes_hidden_layer1]), name = 'Bias')}


    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer1, n_nodes_hidden_layer2]), name = 'Weights'),
                       'biases': tf.Variable(tf.random_normal([n_nodes_hidden_layer2]), name = 'Bias')}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer2, n_nodes_hidden_layer3]), name = 'Weights'),
                       'biases': tf.Variable(tf.random_normal([n_nodes_hidden_layer3]), name = 'Bias')}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer3, n_classes]), name = 'Weights'),
                       'biases': tf.Variable(tf.random_normal([n_classes]), name = 'Bias')}

    l1 = tf.add(tf.matmul(data, hidden_1_layer["weights"]), hidden_1_layer["biases"])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer["weights"]), hidden_2_layer["biases"])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer["weights"]), hidden_3_layer["biases"])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer["weights"]), output_layer["biases"], name="output")
    return output

def train_neural_network(input_data):
    prediction = neural_network_model(input_data)
   # prediction = tf.nn.softmax(prediction)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    hm_epochs = 5 #hm = how many 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("started sessions")
        #print("Optimizier: {}".format(optimizer))
        print("Prediction: {}".format(prediction))

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_images):
                start = i
                end = i + batch_size                
                epoch_x = train_images[start:end]
                epoch_y = train_labels[start:end]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                #print("C:", c)
                #print("Epoch_loss:", epoch_loss)
                i += batch_size
            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'), name="XD")
        #print(sess.run(accuracy, feed_dict={x: test_images,
        #                            y: test_labels}))
        print('Accuracy:', accuracy.eval({x:train_images, y:train_labels}))
        writer.add_graph(sess.graph)

train_neural_network(x)         
 




























