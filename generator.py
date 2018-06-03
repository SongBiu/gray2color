def network_g(image_gray, reuse, is_train):
    '''
    input: 224*224*1
    output: 224*224*3  
    '''
    with tf.variable_scope('network_g', reuse=reuse) as vs:
        
        tl.layers.set_name_reuse(reuse)
        
        # 224*224*1
        img_in = tl.layers.InputLayer(inputs=image_gray, name="img_in")
        
        # 224*224*16
        conv1 = tl.layers.Conv2d(
            img_in, n_filter=16, filter_size=(3, 3), strides=(1, 1), name="conv1")
        conv1 = tl.layers.BatchNormLayer(conv1, is_train=is_train)
        # ???
        conv1 = tl.activation.LeakyRelu(conv1)
        
        # 112*112*32
        conv2 = tl.layers.Conv2d(
            conv1, n_filter=32, filter_size=(3, 3), strides=(2, 2), name="conv2")
        conv2 = tl.layers.BatchNormLayer(conv2, is_train=is_train)
        conv2 = tl.activation.LeakyRelu(conv2)
        
        # 56*56*64
        conv3 = tl.layers.Conv2d(
            conv2, n_filter=64, filter_size=(3, 3), strides=(2, 2), name="conv3")
        conv3 = tl.layers.BatchNormLayer(conv3, is_train=is_train)
        conv3 = tl.activation.LeakyRelu(conv3)
        
        # 28*28*128
        conv4 = tl.layers.Conv2d(
            conv3, n_filter=128, filter_size=(3, 3), strides=(2, 2), name="conv4")
        conv4 = tl.layers.BatchNormLayer(conv4, is_train=is_train)
        conv4 = tl.activation.LeakyRelu(conv4)
       
        # 14*14*128
        conv5 = tl.layers.Conv2d(
            conv4, n_filter=128, filter_size=(3, 3), strides=(2, 2), name="conv5")
        conv5 = tl.layers.BatchNormLayer(conv5, is_train=is_train)
        conv5 = tl.activation.LeakyRelu(conv5)
        
        # # 7*7*256
        # conv6 = tl.layers.Conv2d(
        #     conv5, n_filter=256, filter_size=(3, 3), strides=(2, 2), name="conv6")
        # conv6 = tl.layers.BatchNormLayer(conv6, is_train=is_train)
        # conv6 = tl.activation.LeakyRelu(conv6)
        
        # # 14*14*128
        # deconv5 = tl.layers.DeConv2d(
        #     conv6, n_filter=128, filter_size=(3, 3), strides=(2, 2), name="deconv5")
        # deconv5 = tl.layers.BatchNormLayer(deconv5, is_train=is_train)
        # deconv5 = tl.activation.LeakyRelu(deconv5)
        # deconv5 = tf.concat([conv5, deconv5], axis=3)
        
        # 28*28*128
        deconv4 = tl.layers.DeConv2d(
            deconv5, n_filter=128, filter_size=(3, 3), strides=(2, 2), name="deconv4")
        deconv4 = tl.layers.BatchNormLayer(deconv4, is_train=is_train)
        deconv4 = tl.activation.LeakyRelu(deconv4)
        deconv4 = tf.concat([conv4, deconv4], axis=3)
        
        # 56*56*64
        deconv3 = tl.layers.DeConv2d(
            deconv4, n_filter=64, filter_size=(3, 3), strides=(2, 2), name="deconv3")
        deconv3 = tl.layers.BatchNormLayer(deconv3, is_train=is_train)
        deconv3 = tl.activation.LeakyRelu(deconv3)
        deconv3 = tf.concat([conv3, deconv3], axis=3)
        
        # 112*112*32
        deconv2 = tl.layers.DeConv2d(
            deconv3, n_filter=32, filter_size=(3, 3), strides=(2, 2), name="deconv2")
        deconv2 = tl.layers.BatchNormLayer(deconv2, is_train=is_train)
        deconv2 = tl.activation.LeakyRelu(deconv2)
        deconv2 = tf.concat([conv2, deconv2], axis=3)
        
        # 224*224*16
        deconv1 = tl.layers.DeConv2d(
            deconv2, n_filter=16, filter_size=(3, 3), strides=(2, 2), name="deconv1")
        deconv1 = tl.layers.BatchNormLayer(deconv1, is_train=is_train)
        deconv1 = tl.activation.LeakyRelu(deconv1)
        deconv1 = tf.concat([conv1, deconv1], axis=3)
        
        # 224*224*3
        img_out = tl.layers.Conv2d(
            deconv1, n_filter=3, filter_size=(3, 3), strides=(1, 1), act=tf.nn.tanh, name="img_out")
        
        return img_out