# 模型代码
~~~ python
import numpy as np
import tensorflow as tf

# 初始值和全局设置
sample_size = 168
batch_size = 32
IMG_SHAPE = (1, 48, 48)
epochs = 1000
tf.keras.backend.set_image_data_format('channels_first')


# 生成器模型
def generator_model():
    inputs = tf.keras.Input(shape=IMG_SHAPE)  # 输入形状

    # 编码器部分
    x0 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x0)
    x1 = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x1)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(x1)
    x2 = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)


    # 解码器部分
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(x2)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.concatenate([x, x1], axis=1)

    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(x)
    x3 = tf.keras.layers.UpSampling2D((2, 2))(x)

    #残差连接
    x = tf.keras.layers.Add()([x3, x0])

    # 输出层
    outputs = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# 判别器模型
def discriminator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=IMG_SHAPE),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    return model

# 定义损失函数和优化器
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, clipnorm=1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=8e-6, clipnorm=1)

def discriminator_loss(real_output, fake_output):
    real_loss = loss_func(tf.ones_like(real_output), real_output)
    fake_loss = loss_func(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return loss_func(tf.ones_like(fake_output), fake_output)

def pixel_loss(real_image, generate_image):
    return tf.reduce_mean(tf.abs(real_image - generate_image))


# 创建生成器和判别器
generator_AB = generator_model()
generator_BA = generator_model()
discriminator_A = discriminator_model()
discriminator_B = discriminator_model()
# generator = tf.keras.models.load_model('autoencoder_model')
# generator_AB.load_weights('generator_AB_weights.h5')
# generator_BA.load_weights('generator_BA_weights.h5')
# discriminator_A.load_weights('discriminator_A_weights.h5')
# discriminator_B.load_weights('discriminator_B_weights.h5')


# 读入数据，作为生成器目标值
train_results_true = np.load('data7000/train_result_true.npy')[:2025,:,:].reshape(-1,5,40,40)
train_results_true_edge = tf.tile(train_results_true,[1,1,3,3]).numpy()[:,:,36:84,36:84].reshape(-1,1,48,48)


for epoch in range(epochs):
    for i in range(12):
        #循环读入数据，作为生成器输入值
        train_results = np.load(f'./data7000/train_results{i + 1}.npy').reshape(-1, 8, 10, 1600)[:, :1, 5:10, :].reshape(-1, 5, 40, 40)
        train_results_edge = tf.tile(train_results, [1, 1, 3, 3]).numpy()[:, :, 36:84, 36:84].reshape(-1, 1, 48, 48)[:sample_size,:,:,:]
        train_results_true_edge_copy = train_results_true_edge[i*sample_size:(i+1)*sample_size,:,:,:]
        for j in range(int(sample_size/batch_size)):
            train_x = train_results_edge[j*batch_size:(j+1)*batch_size,:,:,:].astype(np.float32)
            train_y = train_results_true_edge[j*batch_size:(j+1)*batch_size,:,:,:].astype(np.float32)
            with tf.GradientTape(persistent=True) as tape:
                fake_B = generator_AB(train_x, training=True)
                cycled_A = generator_BA(fake_B, training=True)

                fake_A = generator_BA(train_y, training=True)
                cycled_B = generator_AB(fake_A, training=True)

                same_A = generator_BA(train_x, training=True)
                same_B = generator_AB(train_y, training=True)

                discriminator_real_A = discriminator_A(train_x, training=True)
                discriminator_real_B = discriminator_B(train_y, training=True)

                discriminator_fake_A = discriminator_A(fake_A, training=True)
                discriminator_fake_B = discriminator_B(fake_B, training=True)

                generator_AB_loss = generator_loss(discriminator_fake_B)
                generator_BA_loss = generator_loss(discriminator_fake_A)

                identity_loss_A = pixel_loss(train_x, same_A)
                identity_loss_B = pixel_loss(train_y, same_B)
                total_cycle_loss = pixel_loss(train_x, cycled_A) + pixel_loss(train_y, cycled_B)
                total_generator_AB_loss = generator_AB_loss + 2*total_cycle_loss + 1.3*identity_loss_B
                total_generator_BA_loss = generator_BA_loss + 2*total_cycle_loss + 1.3*identity_loss_A

                discriminator_A_loss = discriminator_loss(discriminator_real_A, discriminator_fake_A)
                discriminator_B_loss = discriminator_loss(discriminator_real_B, discriminator_fake_B)

            generator_AB_gradients = tape.gradient(total_generator_AB_loss,generator_AB.trainable_variables)
            generator_BA_gradients = tape.gradient(total_generator_BA_loss,generator_BA.trainable_variables)

            generator_optimizer.apply_gradients(zip(generator_AB_gradients, generator_AB.trainable_variables))
            generator_optimizer.apply_gradients(zip(generator_BA_gradients, generator_BA.trainable_variables))

            if j == 1:
                discriminator_A_gradients = tape.gradient(discriminator_A_loss,discriminator_A.trainable_variables)
                discriminator_B_gradients = tape.gradient(discriminator_B_loss,discriminator_B.trainable_variables)

                discriminator_optimizer.apply_gradients(zip(discriminator_A_gradients,discriminator_A.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(discriminator_B_gradients,discriminator_B.trainable_variables))


    print(f"Epoch: {epoch}, Generator_AB Loss: {total_generator_AB_loss}, Discriminator_A Loss: {discriminator_A_loss}\n"
          f"          Generator_BA Loss: {total_generator_BA_loss}, Discriminator_B Loss: {discriminator_B_loss}")

    if epoch %50 == 0:
        generator_AB.save_weights('generator_AB_weights.h5')
        generator_BA.save_weights('generator_BA_weights.h5')
        discriminator_A.save_weights('discriminator_A_weights.h5')
        discriminator_B.save_weights('discriminator_B_weights.h5')
~~~
# 