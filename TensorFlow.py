import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Dummy data
x = tf.random.normal((1000, 20))
y = tf.random.uniform((1000,), maxval=10, dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(32)

for epoch in range(5):
    for step, (batch_x, batch_y) in enumerate(dataset):

        with tf.GradientTape() as tape:
            logits = model(batch_x, training=True)
            loss = loss_fn(batch_y, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print(f"Epoch {epoch} Loss: {loss.numpy()}")