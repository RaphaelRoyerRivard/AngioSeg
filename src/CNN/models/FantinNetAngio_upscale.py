import tensorflow as tf
import os


def error_rate(predictions, labels, numlabels):
  """Return the error rate based on dense predictions and sparse labels."""
  flat_logits = tf.reshape(predictions, [-1, numlabels])
  flat_labels = tf.reshape(labels, [-1, numlabels])
  correct_prediction = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(flat_logits, 1), tf.argmax(flat_labels, 1)), dtype=tf.float32))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100.0
  immax = tf.reshape(tf.argmax(flat_logits, 1), [tf.shape(predictions)[0],tf.shape(predictions)[1], tf.shape(predictions)[2]])
  return accuracy, immax

def get_image_summary_disp_grayscale(img):
    disp = tf.slice(img, (0,0,0), (1,-1,-1))
    disp -= tf.reduce_min(disp)
    disp /= tf.reduce_max(disp)
    disp *=255
    disp = tf.cast(tf.reshape(disp, [1, tf.shape(disp)[1], tf.shape(disp)[2], 1]),  dtype=tf.uint8)
    return disp

def get_image_summary_disp(img):
    disp = tf.slice(img, (0,0,0,0), (1,-1,-1,-1))
    disp -= tf.reduce_min(disp)
    disp /= tf.reduce_max(disp)
    disp *=255
    disp = tf.reshape(disp, [1, tf.shape(disp)[1], tf.shape(disp)[2], tf.shape(disp)[3]])
    return disp

def get_image_summary(img, numchannel):
    disp = tf.slice(img, (0, 0, 0, 0), (1, -1, -1, -1))
    if numchannel==2:
        disp = tf.concat([disp, tf.zeros((1, tf.shape(disp)[1], tf.shape(disp)[2], 1))], 3)
    disp = tf.reshape(disp, [1, tf.shape(disp)[1], tf.shape(disp)[2], tf.shape(disp)[3]])
    return disp

def get_image_summary_2labels_disp(img):
    disp = tf.slice(img, (0, 0, 0, 0), (1, -1, -1, -1))
    disp = tf.concat([disp,tf.zeros((1,tf.shape(disp)[1], tf.shape(disp)[2],1))], axis=3)
    disp -= tf.reduce_min(disp)
    disp /= tf.reduce_max(disp)
    disp *=255
    disp = tf.reshape(disp, [1, tf.shape(disp)[1], tf.shape(disp)[2], tf.shape(disp)[3]])
    return disp

def get_image_summary_2labels(img):
    disp = tf.slice(img, (0, 0, 0, 0), (1, -1, -1, -1))
    disp = tf.concat([disp,tf.zeros((1,tf.shape(disp)[1], tf.shape(disp)[2],1))], axis=3)
    disp = tf.reshape(disp, [1, tf.shape(disp)[1], tf.shape(disp)[2], tf.shape(disp)[3]])
    return disp

def get_conv_summary(img):
    disp = tf.slice(img, (0, 0, 0, 0), (1, -1, -1, 1))
    disp = tf.reshape(disp, [1, tf.shape(disp)[1], tf.shape(disp)[2], tf.shape(disp)[3]])
    return disp

def model(data, train, imagesize, numchannels, numlabels):

  if os.name == 'nt':
      conv1_weights = tf.get_variable("conv1_weights", shape=[3, 3, numchannels, 32], initializer=tf.contrib.layers.xavier_initializer())
      conv1_biases = tf.get_variable("conv1_biases", initializer = tf.constant(0.0, shape=[32]))

      conv1b_weights = tf.get_variable("conv1b_weights", shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
      conv1b_biases = tf.get_variable("conv1b_biases", initializer = tf.constant(0.0, shape=[32]))

      conv1c_weights = tf.get_variable("conv1c_weights", shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
      conv1c_biases = tf.get_variable("conv1c_biases", initializer = tf.constant(0.0, shape=[32]))

      conv2_weights = tf.get_variable("conv2_weights", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
      conv2_biases = tf.get_variable("conv2_biases", initializer = tf.constant(0.0, shape=[64]))

      conv2b_weights = tf.get_variable("conv2b_weights", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
      conv2b_biases = tf.get_variable("conv2b_biases", initializer = tf.constant(0.0, shape=[64]))

      conv2c_weights = tf.get_variable("conv2c_weights", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
      conv2c_biases = tf.get_variable("conv2c_biases", initializer = tf.constant(0.0, shape=[64]))

      conv3_weights = tf.get_variable("conv3_weights", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
      conv3_biases = tf.get_variable("conv3_biases", initializer = tf.constant(0.0, shape=[128]))

      conv3b_weights = tf.get_variable("conv3b_weights", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
      conv3b_biases = tf.get_variable("conv3b_biases", initializer = tf.constant(0.0, shape=[128]))

      conv3c_weights = tf.get_variable("conv3c_weights", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
      conv3c_biases = tf.get_variable("conv3c_biases", initializer = tf.constant(0.0, shape=[128]))

      conv4_weights = tf.get_variable("conv4_weights", shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
      conv4_biases = tf.get_variable("conv4_biases", initializer = tf.constant(0.0, shape=[256]))

      conv4b_weights = tf.get_variable("conv4b_weights", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
      conv4b_biases = tf.get_variable("conv4b_biases", initializer = tf.constant(0.0, shape=[256]))

      conv4c_weights = tf.get_variable("conv4c_weights", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
      conv4c_biases = tf.get_variable("conv4c_biases", initializer = tf.constant(0.0, shape=[256]))

      conv5_weights = tf.get_variable("conv5_weights", shape=[3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
      conv5_biases = tf.get_variable("conv5_biases", initializer = tf.constant(0.0, shape=[512]))

      conv5b_weights = tf.get_variable("conv5b_weights", shape=[3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
      conv5b_biases = tf.get_variable("conv5b_biases",  initializer = tf.constant(0.0, shape=[512]))

      conv5c_weights = tf.get_variable("conv5c_weights", shape=[3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
      conv5c_biases = tf.get_variable("conv5c_biases", initializer = tf.constant(0.0, shape=[512]))

      deconv5_weights = tf.get_variable("deconv5_weights", shape=[2, 2, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
      deconv5_biases = tf.get_variable("deconv5_biases", initializer = tf.constant(0.0, shape=[256]))

      conv5de_weights = tf.get_variable("conv5de_weights", shape=[3, 3, 512, 256], initializer=tf.contrib.layers.xavier_initializer())
      conv5de_biases = tf.get_variable("conv5de_biases",  initializer = tf.constant(0.0, shape=[256]))

      deconv4_weights = tf.get_variable("deconv4_weights", shape=[2, 2, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
      deconv4_biases = tf.get_variable("deconv4_biases",  initializer = tf.constant(0.0, shape=[128]))

      conv4de_weights = tf.get_variable("conv4de_weights", shape=[3, 3, 256, 128], initializer=tf.contrib.layers.xavier_initializer())
      conv4de_biases = tf.get_variable("conv4de_biases", initializer = tf.constant(0.0, shape=[128]))

      deconv3_weights = tf.get_variable("deconv3_weights", shape=[2, 2, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
      deconv3_biases = tf.get_variable("deconv3_biases", initializer = tf.constant(0.0, shape=[64]))

      conv3de_weights = tf.get_variable("conv3de_weights", shape=[3, 3, 128, 64],initializer=tf.contrib.layers.xavier_initializer())
      conv3de_biases = tf.get_variable("conv3de_biases",  initializer = tf.constant(0.0, shape=[64]))

      deconv2_weights = tf.get_variable("deconv2_weights", shape=[2, 2, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
      deconv2_biases = tf.get_variable("deconv2_biases", initializer = tf.constant(0.0, shape=[32]))

      conv2de_weights = tf.get_variable("conv2de_weights", shape=[3, 3, 64, 32],initializer=tf.contrib.layers.xavier_initializer())
      conv2de_biases = tf.get_variable("conv2de_biases", initializer = tf.constant(0.0, shape=[32]))

      deconv1_weights = tf.get_variable("deconv1_weights", shape=[2, 2, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
      deconv1_biases = tf.get_variable("deconv1_biases", initializer = tf.constant(0.0, shape=[16]))

      conv1de_weights = tf.get_variable("conv1de_weights", shape=[3, 3, 32, 16],initializer=tf.contrib.layers.xavier_initializer())
      conv1de_biases = tf.get_variable("conv1de_biases", initializer = tf.constant(0.0, shape=[16]))

      convlast_weights = tf.get_variable("convfinal_weights", shape=[3, 3, 16, numlabels], initializer=tf.contrib.layers.xavier_initializer())
      convlast_biases = tf.get_variable("convfinal_biases", initializer = tf.constant(0.0, shape=[numlabels]))

      # conv5_weights = tf.get_variable("conv4_weights", shape=[3, 3, 256, 512])
      # conv5_biases = tf.Variable(tf.constant(0.0, shape=[512]))

      # conv5_weights = tf.get_variable("conv4_weights", shape=[3, 3, 256, 512])
      # conv5_biases = tf.Variable(tf.constant(0.0, shape=[512]))

      fc1_weights = tf.get_variable("fc1_weights", shape=[imagesize // 16 * imagesize // 16 * 256, 1024])
      fc1_biases = tf.Variable(tf.constant(0.0, shape=[1024]))

      fc2_weights = tf.get_variable("fc2_weights", shape=[1024, numlabels])
  else:
    conv1_weights = tf.get_variable("conv1_weights", shape=[3, 3, numchannels, 32],
                                  initializer=tf.contrib.layers.xavier_initializer())
    conv1_biases = tf.Variable(tf.zeros([32]))

    conv2_weights = tf.get_variable("conv2_weights", shape=[3, 3, 32, 64],
                                  initializer=tf.contrib.layers.xavier_initializer())
    conv2_biases = tf.Variable(tf.constant(0.0, shape=[64]))

    conv3_weights = tf.get_variable("conv3_weights", shape=[3, 3, 64, 128],
                                  initializer=tf.contrib.layers.xavier_initializer())
    conv3_biases = tf.Variable(tf.constant(0.0, shape=[128]))

    conv4_weights = tf.get_variable("conv4_weights", shape=[3, 3, 128, 256],
                                  initializer=tf.contrib.layers.xavier_initializer())
    conv4_biases = tf.Variable(tf.constant(0.0, shape=[256]))

    fc1_weights = tf.get_variable("fc1_weights", shape=[imagesize // 16 * imagesize // 16 * 256, 1024],
                                initializer=tf.contrib.layers.xavier_initializer())
    fc1_biases = tf.Variable(tf.constant(0.0, shape=[1024]))

    fc2_weights = tf.get_variable("fc2_weights", shape=[1024, numlabels],
                                initializer=tf.contrib.layers.xavier_initializer())
  fc2_biases = tf.Variable(tf.constant(0.0, shape=[numlabels]))

  """The Model definition."""
  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].
  out1 = tf.nn.conv2d(data,
                      conv1_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  # Bias and rectified linear non-linearity.
  out1pb = tf.nn.bias_add(out1, conv1_biases)
  out1nl = tf.nn.relu(out1pb)

  out1b = tf.nn.conv2d(out1nl,
                      conv1b_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  # Bias and rectified linear non-linearity.
  out1nlb = tf.nn.relu(tf.nn.bias_add(out1b, conv1b_biases))

  out1c = tf.nn.conv2d(out1nlb,
                     conv1c_weights,
                    strides=[1, 1, 1, 1],
                     padding='SAME')
  # Bias and rectified linear non-linearity.
  out1nlc = tf.nn.relu(tf.nn.bias_add(out1c, conv1c_biases))
  # Max pooling. The kernel size spec {ksize} also follows the layout of
  # the data. Here we have a pooling window of 2, and a stride of 2.
  pool = tf.nn.max_pool(out1nlc,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
  out2 = tf.nn.conv2d(pool,
                      conv2_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(out2, conv2_biases))
  out2b = tf.nn.conv2d(relu,
                      conv2b_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(out2b, conv2b_biases))
  out2c = tf.nn.conv2d(relu,
                      conv2c_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(out2c, conv2c_biases))
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
  out3 = tf.nn.conv2d(pool,
                      conv3_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')

  relu = tf.nn.relu(tf.nn.bias_add(out3, conv3_biases))
  out3b = tf.nn.conv2d(relu,
                      conv3b_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(out3b, conv3b_biases))
  out3b = tf.nn.conv2d(relu,
                      conv3c_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(out3b, conv3c_biases))
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

  conv = tf.nn.conv2d(pool,
                      conv4_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')

  relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))
  out4b = tf.nn.conv2d(relu,
                      conv4b_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(out4b, conv4b_biases))
  out4c = tf.nn.conv2d(relu,
                      conv4c_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(out4c, conv4c_biases))
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

  conv = tf.nn.conv2d(pool,
                      conv5_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')

  relu = tf.nn.relu(tf.nn.bias_add(conv, conv5_biases))
  out5b = tf.nn.conv2d(relu,
                      conv5b_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(out5b, conv5b_biases))
  out5c = tf.nn.conv2d(relu,
                      conv5c_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(out5c, conv5c_biases))
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
  # Reshape the feature map cuboid into a 2D matrix to feed it to the
  # fully connected layers.
  #pool_shape = pool.get_shape().as_list()
  # reshape = tf.reshape(
  #   pool,
  #   [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
  # # Fully connected layer. Note that the '+' operation automatically
  # # broadcasts the biases.
  # hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

  # reshapeh = tf.reshape(
  #   reshape,
  #   [pool_shape[0], pool_shape[1], pool_shape[2], pool_shape[3]])

  shapecour = tf.shape(pool)
  #shapecour = tf.Print(shapecour, [shapecour])
  outputshape = tf.stack([shapecour[0], shapecour[1]*2, shapecour[2]*2, shapecour[3]//2])
  #deconv = tf.nn.conv2d_transpose(pool, deconv5_weights, outputshape,
  #                                strides=[1,2,2,1], padding='SAME')

  #tf.summary.image('deconv5', get_conv_summary(deconv))

  #relu = tf.nn.relu(tf.nn.bias_add(deconv, deconv5_biases))
  relu = tf.image.resize_images(pool, (shapecour[1]*2, shapecour[2]*2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  conv = tf.nn.conv2d(relu, conv5de_weights, strides=[1, 1, 1, 1], padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv5de_biases))

  #tf.summary.image('deconv5conv5', get_conv_summary(relu))

  shapecour = tf.shape(relu)
  #shapecour = tf.Print(shapecour, [shapecour, ' fdg ' , shapecour[3]])
  outputshape = tf.stack([shapecour[0], shapecour[1]*2, shapecour[2]*2, shapecour[3]//2])
  #deconv = tf.nn.conv2d_transpose(relu, deconv4_weights, outputshape,
  #                                strides=[1,2,2,1], padding='SAME')

  #relu = tf.nn.relu(tf.nn.bias_add(deconv, deconv4_biases))
  relu = tf.image.resize_images(relu, (shapecour[1]*2, shapecour[2]*2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  conv = tf.nn.conv2d(relu, conv4de_weights, strides=[1, 1, 1, 1], padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv4de_biases))

  shapecour = tf.shape(relu)
  #outputshape = tf.stack([shapecour[0], shapecour[1]*2, shapecour[2]*2, shapecour[3]//2])
  #deconv = tf.nn.conv2d_transpose(relu, deconv3_weights, outputshape,
  #                                strides=[1,2,2,1], padding='SAME')
  #relu = tf.nn.relu(tf.nn.bias_add(deconv, deconv3_biases))
  relu = tf.image.resize_images(relu, (shapecour[1] * 2, shapecour[2] * 2),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  conv = tf.nn.conv2d(relu, conv3de_weights, strides=[1, 1, 1, 1], padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv3de_biases))

  shapecour = tf.shape(relu)
  #outputshape = tf.stack([shapecour[0], shapecour[1]*2, shapecour[2]*2, shapecour[3]//2])
  #deconv = tf.nn.conv2d_transpose(relu, deconv2_weights, outputshape,
  #                                strides=[1,2,2,1], padding='SAME')
  #relu = tf.nn.relu(tf.nn.bias_add(deconv, deconv2_biases))
  relu = tf.image.resize_images(relu, (shapecour[1] * 2, shapecour[2] * 2),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  conv = tf.nn.conv2d(relu, conv2de_weights, strides=[1, 1, 1, 1], padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv2de_biases))

  shapecour = tf.shape(relu)
  #shapecour = tf.Print(shapecour, [shapecour])
  #outputshape = tf.stack([shapecour[0], shapecour[1]*2, shapecour[2]*2, shapecour[3]//2])
  #deconv = tf.nn.conv2d_transpose(relu, deconv1_weights, outputshape,
  #                               strides=[1,2,2,1], padding='SAME')
  #relu = tf.nn.relu(tf.nn.bias_add(deconv, deconv1_biases))
  relu = tf.image.resize_images(relu, (shapecour[1] * 2, shapecour[2] * 2),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  conv = tf.nn.conv2d(relu, conv1de_weights, strides=[1, 1, 1, 1], padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv1de_biases))

  conv = tf.nn.conv2d(relu,
                      convlast_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  outputmap = tf.nn.relu(tf.nn.bias_add(conv, convlast_biases))




  #shapecour = tf.shape(outputmap)
  #shapecour = tf.Print(shapecour, [shapecour], 'test here: ')
  #outputshape = tf.stack([shapecour[0], shapecour[1] * 2, shapecour[2] * 2, shapecour[3] // 2])

  # Add a 50% dropout during training only. Dropout also scales
  # activations such that no rescaling is needed at evaluation time.
  #if train:
    #hidden = tf.nn.dropout(hidden, 0.25, seed=None)
  return outputmap, out1pb, conv1_biases, outputshape, out1,  out2

def get_loss(outputmap, labels, numlabels):
    flat_logits = tf.reshape(outputmap, [-1, numlabels])
    flat_labels = tf.reshape(labels, [-1, numlabels])

    logits_log = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
    loss = tf.reduce_mean(logits_log)

    return loss

def build_feedfoward_ops(train_data_node, train_labels_node, train, imagesize, numchannel, numlabels, reusetest=True):



  if train:
    with tf.variable_scope("variable", reuse=None) as scope:
        outputmap, conv1_weights, conv1_biases, out1, out1nl, out2 = model(train_data_node, train,
                                                                                             imagesize, numchannel,
                                                                                             numlabels)

      #logits_log = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=train_labels_node)
    #train_prediction = tf.nn.softmax(logits)
    loss = get_loss(outputmap, train_labels_node, numlabels)
    tf.summary.image('label', get_image_summary_2labels(train_labels_node), collections=['train'])
    tf.summary.scalar('Loss', loss, collections=['train'])
    optimizer = tf.train.AdamOptimizer(0.0001, name='AdamOptimizer').minimize(loss)
    accuracyt, outmapmax = error_rate(outputmap, train_labels_node, numlabels)

    tf.summary.image('outputmap', get_image_summary_2labels_disp(outputmap), collections=['train'])
    tf.summary.image('input', get_image_summary(train_data_node, numchannel), collections=['train'])
    tf.summary.image('outmapmax', get_image_summary_disp_grayscale(outmapmax), collections=['train'])
    tf.summary.scalar('Accuracy', accuracyt, collections=['train'])

    return outputmap, loss, optimizer, accuracyt, out1
  else:
    with tf.variable_scope("variable", reuse=reusetest) as scope:
        outputmapeval, conv1_weights, conv1_biases,out1eval, out1nleval, out2eval = model(train_data_node, train, imagesize, numchannel,numlabels)
    accuracyeval, outmapmaxeval = error_rate(outputmapeval, train_labels_node, numlabels)
    tf.summary.image('outputmap_test', get_image_summary_2labels_disp(outputmapeval),collections=['test'])
    tf.summary.image('input_test', get_image_summary(train_data_node, numchannel), collections=['test'])
    tf.summary.image('outmapmax_test', get_image_summary_disp_grayscale(outmapmaxeval), collections=['test'])
    return outputmapeval, outmapmaxeval, accuracyeval,  conv1_weights, conv1_biases

