def network_g(image_gray, reuse, is_train):
    '''
    input: 64*64*1
    output: 64*64*3  
    '''
    with tf.variable_scope('network_g', reuse=reuse) as vs:
        
        tl.layers.set_name_reuse(reuse)
        
        # 64*64*1
        img_in = tl.layers.InputLayer(inputs=image_gray, name="img_in")
        
        # 64*64*16
        conv1 = tl.layers.Conv2d(
            img_in, n_filter=16, filter_size=(3, 3), strides=(1, 1), name="conv1_1")
        conv1 = tl.layers.BatchNormLayer(conv1, is_train=is_train, act=tf.nn.leaky_relu)
        conv1 = tl.layers.Conv2d(
            conv1, n_filter=16, filter_size=(3, 3), strides=(1, 1), name="conv1_2")
        conv1 = tl.layers.BatchNormLayer(conv1, is_train=is_train, act=tf.nn.leaky_relu)

        # 32*32*32
        conv2 = tl.layers.Conv2d(
            conv1, n_filter=32, filter_size=(3, 3), strides=(2, 2), name="conv2_1")
        conv2 = tl.layers.BatchNormLayer(conv2, is_train=is_train, act=tf.nn.leaky_relu)
        conv2 = tl.layers.Conv2d(
            conv2, n_filter=32, filter_size=(3, 3), strides=(1, 1), name="conv2_2")
        conv2 = tl.layers.BatchNormLayer(conv2, is_train=is_train, act=tf.nn.leaky_relu)
        
        # 16*16*64
        conv3 = tl.layers.Conv2d(
            conv2, n_filter=64, filter_size=(3, 3), strides=(2, 2), name="conv3_1")
        conv3 = tl.layers.BatchNormLayer(conv3, is_train=is_train, act=tf.nn.leaky_relu)
        conv3 = tl.layers.Conv2d(
            conv3, n_filter=64, filter_size=(3, 3), strides=(1, 1), name="conv3_2")
        conv3 = tl.layers.BatchNormLayer(conv3, is_train=is_train, act=tf.nn.leaky_relu)

        # 8*8*128
        conv4 = tl.layers.Conv2d(
            conv3, n_filter=128, filter_size=(3, 3), strides=(2, 2), name="conv4_1")
        conv4 = tl.layers.BatchNormLayer(conv4, is_train=is_train, act=tf.nn.leaky_relu)
        conv4 = tl.layers.Conv2d(
            conv4, n_filter=128, filter_size=(3, 3), strides=(1, 1), name="conv4_2")
        conv4 = tl.layers.BatchNormLayer(conv4, is_train=is_train, act=tf.nn.leaky_relu)
       
        # 4*4*256
        conv5 = tl.layers.Conv2d(
            conv4, n_filter=256, filter_size=(3, 3), strides=(2, 2), name="conv5_1")
        conv5 = tl.layers.BatchNormLayer(conv5, is_train=is_train, act=tf.nn.leaky_relu)
        conv5 = tl.layers.Conv2d(
            conv5, n_filter=256, filter_size=(3, 3), strides=(1, 1), name="conv5_2")
        conv5 = tl.layers.BatchNormLayer(conv5, is_train=is_train, act=tf.nn.leaky_relu)
        
        '''
        architecture of deconv-layer:
			deconv(upsample) => 
			concat(with corresponding conv) => 
			conv(1*1 conv)
        '''
        # 8*8*128
        deconv4 = tl.layers.DeConv2d(
            conv5, n_filter=128, filter_size=(3, 3), strides=(2, 2), name="deconv4")
        deconv4 = tl.layers.BatchNormLayer(deconv4, is_train=is_train, act=tf.nn.leaky_relu)
        deconv4 = tf.concat([conv4, deconv4], axis=3)
        deconv4 = tl.layers.Conv2d(
            deconv4, n_filter=128, filter_size=(3, 3), strides=(1, 1), name="conv_after_deconv4")
        deconv4 = tl.layers.BatchNormLayer(deconv4, is_train=is_train, act=tf.nn.leaky_relu)
        
        # 16*16*64
        deconv3 = tl.layers.DeConv2d(
            deconv4, n_filter=64, filter_size=(3, 3), strides=(2, 2), name="deconv3")
        deconv3 = tl.layers.BatchNormLayer(deconv3, is_train=is_train, act=tf.nn.leaky_relu)
        deconv3 = tf.concat([conv3, deconv3], axis=3)
        deconv3 = tl.layers.Conv2d(
            deconv3, n_filter=64, filter_size=(3, 3), strides=(1, 1), name="conv_after_deconv3")
        deconv3 = tl.layers.BatchNormLayer(deconv3, is_train=is_train, act=tf.nn.leaky_relu)
        
        # 32*32*32
        deconv2 = tl.layers.DeConv2d(
            deconv3, n_filter=32, filter_size=(3, 3), strides=(2, 2), name="deconv2")
        deconv2 = tl.layers.BatchNormLayer(deconv2, is_train=is_train, act=tf.nn.leaky_relu)
        deconv2 = tf.concat([conv2, deconv2], axis=3)
        deconv2 = tl.layers.Conv2d(
            deconv2, n_filter=32, filter_size=(3, 3), strides=(1, 1), name="conv_after_deconv2")
        deconv2 = tl.layers.BatchNormLayer(deconv2, is_train=is_train, act=tf.nn.leaky_relu)
        
        # 64*64*16
        deconv1 = tl.layers.DeConv2d(
            deconv2, n_filter=16, filter_size=(3, 3), strides=(2, 2), name="deconv1")
        deconv1 = tl.layers.BatchNormLayer(deconv1, is_train=is_train, act=tf.nn.leaky_relu)
        deconv1 = tf.concat([conv1, deconv1], axis=3)
        deconv1 = tl.layers.Conv2d(
            deconv1, n_filter=16, filter_size=(3, 3), strides=(1, 1), name="conv_after_deconv1")
        # last layer before tanh, use relu (instead of leaky_relu)
        deconv1 = tl.layers.BatchNormLayer(deconv1, is_train=is_train, act=tf.nn.relu)
        
        # 64*64*3
        img_out = tl.layers.Conv2d(
            deconv1, n_filter=3, filter_size=(3, 3), strides=(1, 1), act=tf.nn.tanh, name="img_out")
        
        return img_out