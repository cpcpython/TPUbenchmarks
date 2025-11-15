# Ensemble = 1
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import optax # For optimizers
import flax.linen as nn # For neural network modules
from flax.training import train_state
import tensorflow_datasets as tfds # For loading datasets
import tensorflow as tf # Used for tfds data loading, not for model building
import time
import math # For math.inf

print(f"JAX version: {jax.__version__}")
print(f"Optax version: {optax.__version__}")
print(f"TensorFlow version (used for data loading): {tf.__version__}")

# --- 1. TPU Initialization (JAX style) ---
#******************************************************************************
try:
    devices = jax.devices()
    tpu_devices = [d for d in devices if d.platform == 'tpu']
    if not tpu_devices:
        raise ValueError("No TPU devices found.")
    print(f"Found JAX devices: {devices}")
    print(f"Number of TPU devices available: {len(tpu_devices)}")
except ValueError as e:
    print(f"ERROR: {e}. Please ensure your Colab runtime is set to TPU.")
    print("Go to 'Runtime' -> 'Change runtime type' and select 'TPU'.")
    raise SystemExit("TPU not found or not initialized for JAX.")

# --- 2. Define a Mesh for Sharding ---
#******************************************************************************
num_tpu_cores = len(tpu_devices)
mesh = Mesh(tpu_devices, axis_names=('data',))
print(f"JAX Mesh created with axis_names: {mesh.axis_names}")

# --- 3. Load and Preprocess CIFAR-10 Dataset with Enhanced Augmentation ---
#******************************************************************************
def preprocess_image(image, label, is_training):
    image = tf.cast(image, tf.float32) / 255.0
    
    if is_training:
        # Enhanced Data Augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        # Random crop with padding to ensure output size is 32x32
        paddings = tf.constant([[4, 4], [4, 4], [0, 0]])
        padded_image = tf.pad(image, paddings, "REFLECT")
        image = tf.image.random_crop(padded_image, size=[32, 32, 3])
    return image, label

def apply_cutmix(images, labels):
    # Cast images and labels to the correct data types
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.int64)
    
    # CutMix requires a second batch of images and labels
    shuffled_images = tf.random.shuffle(images)
    shuffled_labels = tf.random.shuffle(labels)
    
    # CutMix logic
    lam = tf.random.uniform(shape=[], dtype=tf.float32) # Scalar lambda
    lam = tf.maximum(lam, 1.0 - lam)
    
    W = tf.cast(tf.shape(images)[1], tf.float32)
    H = tf.cast(tf.shape(images)[2], tf.float32)
    
    cut_w_f = W * tf.sqrt(1.0 - lam)
    cut_h_f = H * tf.sqrt(1.0 - lam)
    cut_w = tf.cast(cut_w_f, dtype=tf.int32)
    cut_h = tf.cast(cut_h_f, dtype=tf.int32)
    
    
    cx = tf.random.uniform(shape=[], maxval=tf.cast(W, tf.int32), dtype=tf.int32) # Scalar cx
    cy = tf.random.uniform(shape=[], maxval=tf.cast(H, tf.int32), dtype=tf.int32) # Scalar cy
    
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, tf.cast(W, tf.int32))
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, tf.cast(H, tf.int32))
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, tf.cast(W, tf.int32))
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, tf.cast(H, tf.int32))
    
    # Create the mixed images
    mixed_images = images
    
    # Extract the patch from shuffled images
    patch = shuffled_images[:, y1:y2, x1:x2, :]
    
    # Get the shape of the patch
    patch_shape = tf.shape(patch)
    batch_size = patch_shape[0]
    patch_height = patch_shape[1]
    patch_width = patch_shape[2]
    num_channels = patch_shape[3]
    
    # Create indices for tensor_scatter_nd_update
    # Indices should be [batch_size, patch_height, patch_width, num_channels, 4]
    # where the last dimension contains [batch_idx, y, x, channel_idx]
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), [-1, 1, 1, 1]), [1, patch_height, patch_width, num_channels])
    y_indices = tf.tile(tf.reshape(tf.range(y1, y2), [1, -1, 1, 1]), [batch_size, 1, patch_width, num_channels])
    x_indices = tf.tile(tf.reshape(tf.range(x1, x2), [1, 1, -1, 1]), [batch_size, patch_height, 1, num_channels])
    channel_indices = tf.tile(tf.reshape(tf.range(num_channels), [1, 1, 1, -1]), [batch_size, patch_height, patch_width, 1])
    
    indices = tf.stack([batch_indices, y_indices, x_indices, channel_indices], axis=-1)
    
    
    # Combine the images
    mixed_images = tf.tensor_scatter_nd_update(mixed_images, indices, patch)
    
    
    # Recalculate lambda_prime based on actual patch size
    lam = 1.0 - (tf.cast(x2-x1, tf.float32) * tf.cast(y2-y1, tf.float32) / (W*H))
    
    # One-hot encode the labels
    labels_one_hot = tf.one_hot(labels, 10) # Assuming 10 classes for CIFAR-10
    shuffled_labels_one_hot = tf.one_hot(shuffled_labels, 10)
    
    # Create the mixed labels
    mixed_labels = lam * labels_one_hot + (1.0 - lam) * shuffled_labels_one_hot
    
    return mixed_images, mixed_labels

def load_dataset(batch_size, split, is_training, cutmix=False):
    ds, ds_info = tfds.load(
        'cifar10',
        split=split,
        shuffle_files=is_training,
        as_supervised=True,
        with_info=True,
        data_dir='./tfds_data'
    )
    
    ds = ds.map(lambda img, lbl: preprocess_image(img, lbl, is_training=is_training))
    ds = ds.cache()
    if is_training:
        ds = ds.shuffle(ds_info.splits[split].num_examples)
        if cutmix:
            ds = ds.batch(batch_size, drop_remainder=True)
            ds = ds.map(lambda images, labels: apply_cutmix(images, labels), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size)
    
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, ds_info.features['label'].num_classes

print("\nLoading CIFAR-10 dataset...")
global_batch_size = 1024*2 ##  128 * num_tpu_cores
train_ds, num_classes = load_dataset(global_batch_size, split='train', is_training=True, cutmix=True)
test_ds, _ = load_dataset(global_batch_size, split='test', is_training=False)
validation_split_ratio = 0.1
num_train_examples = 50000
num_val_examples = int(num_train_examples * validation_split_ratio)
num_train_for_actual_training = num_train_examples - num_val_examples
train_ds_full, _ = load_dataset(global_batch_size, split='train', is_training=True, cutmix=True)
val_ds_full, _ = load_dataset(global_batch_size, split='train', is_training=False, cutmix=False)
train_ds_actual = train_ds_full.skip(num_val_examples // global_batch_size)
val_ds = val_ds_full.take(num_val_examples // global_batch_size)

print(f"Dataset loaded. Number of classes: {num_classes}")
print(f"Global batch size: {global_batch_size}")
print(f"Number of training examples: {num_train_for_actual_training}")
print(f"Number of validation examples: {num_val_examples}")
print(f"Number of test examples: {test_ds.cardinality().numpy() * global_batch_size}")

# --- 4. Define the CNN Model with Batch Normalization ---
#******************************************************************************
class CNN(nn.Module):
    num_classes: int
    dropout_rate: float
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        
        x = nn.Conv(features=256, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        
        x = x.reshape((x.shape[0], -1))
        
        x = nn.Dense(features=512)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        
        x = nn.Dense(features=self.num_classes)(x)
        return x

# --- 5. Define Loss, Metrics, and Optimizer ---

#******************************************************************************
def cross_entropy_loss(logits, labels):
    return -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)

def accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, axis=-1))

# Learning rate schedule
total_steps = (num_train_for_actual_training // global_batch_size) * 500
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=0.0005,
    warmup_steps=total_steps * 0.1,
    decay_steps=total_steps,
    end_value=1e-5
)

weight_decay = 1e-3
optimizer = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)

# --- 6. Define Training and Evaluation Step Functions ---

#******************************************************************************
class TrainState(train_state.TrainState):
    batch_stats: dict # Added batch_stats to the state


@jit
def train_step(state, batch, dropout_rng):
    images, labels = batch
    dropout_rng, new_dropout_rng = random.split(dropout_rng)
    
    def loss_and_metrics(params):
        # We need to compute both the loss and the updated batch_stats
        logits, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            images,
            train=True,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng}
        )
        loss = jnp.mean(cross_entropy_loss(logits, labels))
        acc = accuracy(logits, labels)
        return loss, (acc, new_model_state['batch_stats'])
    
    (loss, (acc, new_batch_stats)), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(state.params)
    
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    
    return state.replace(params=new_params, opt_state=new_opt_state, batch_stats=new_batch_stats), loss, acc, new_dropout_rng

@jit
def eval_step(state, batch): # Eval step now takes the full state
    images, labels = batch
    # Pass train=False to use running averages for Batch Norm and turn off Dropout
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        images,
        train=False
    )
    # Cast labels to float32 for one-hot encoding
    one_hot_labels = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(cross_entropy_loss(logits, one_hot_labels))
    acc = accuracy(logits, one_hot_labels)
    return loss, acc



# --- 7. Ensemble Modeling ---

#******************************************************************************
num_models = 1 # Number of models for the ensemble
best_models = []
print(f"\nStarting to train an ensemble of {num_models} models.")

for model_idx in range(num_models):
    print(f"\n--- Training Model {model_idx + 1}/{num_models} ---")
    key = random.PRNGKey(model_idx)
    key, init_key, dropout_init_key = random.split(key, 3)
    
    dummy_input = jnp.ones([1, 32, 32, 3])
    dropout_rate = 0.5
    model_instance = CNN(num_classes=num_classes, dropout_rate=dropout_rate)
    
    variables = model_instance.init(init_key, dummy_input, train=False)
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    opt_state = optimizer.init(params)
    state = TrainState.create(apply_fn=model_instance.apply, params=params, tx=optimizer, batch_stats=batch_stats)
    state = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P())), state)
    
    num_epochs = 500
    patience = 30
    best_val_loss = math.inf
    epochs_no_improve = 0
    best_params = None
    best_batch_stats = None
    dropout_rng = random.PRNGKey(model_idx + 100) 
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # Training Phase
        total_train_loss = 0.0
        total_train_accuracy = 0.0
        num_train_batches = 0
        
        for batch_tf in train_ds_actual.as_numpy_iterator():
            images_np, labels_one_hot_np = batch_tf
            num_train_batches += 1
            sharded_images = jax.device_put(images_np, NamedSharding(mesh, P('data', None, None, None)))
            sharded_labels = jax.device_put(labels_one_hot_np, NamedSharding(mesh, P('data',)))
            state, loss, acc, dropout_rng = train_step(state, (sharded_images, sharded_labels), dropout_rng)
            total_train_loss += loss.item()
            total_train_accuracy += acc.item()
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_accuracy = total_train_accuracy / num_train_batches
        # Validation Phase
        total_val_loss = 0.0
        total_val_accuracy = 0.0
        num_val_batches = 0
        for batch_tf in val_ds.as_numpy_iterator():
            images_np, labels_np = batch_tf
            num_val_batches += 1
            sharded_images = jax.device_put(images_np, NamedSharding(mesh, P('data', None, None, None)))
            sharded_labels = jax.device_put(labels_np, NamedSharding(mesh, P('data',)))
            val_loss, val_acc = eval_step(state, (sharded_images, sharded_labels))
            total_val_loss += val_loss.item()
            total_val_accuracy += val_acc.item()
        avg_val_loss = total_val_loss / num_val_batches
        avg_val_accuracy = total_val_accuracy / num_val_batches
        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {avg_train_accuracy:.4f} | "
              f"Val Loss = {avg_val_loss:.4f}, Val Acc = {avg_val_accuracy:.4f} (Time: {epoch_end_time - epoch_start_time:.2f}s)")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_params = state.params
            best_batch_stats = state.batch_stats
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
    best_models.append({'params': best_params, 'batch_stats': best_batch_stats})

# --- 8. Final Evaluation on Test Set using the Ensemble ---

#******************************************************************************
print("\n--- Final Evaluation on Test Set using Ensemble ---")
total_ensemble_test_loss = 0.0
total_ensemble_test_accuracy = 0.0
num_test_batches = 0
for batch_tf in test_ds.as_numpy_iterator():
    images_np, labels_np = batch_tf
    num_test_batches += 1
    sharded_images = jax.device_put(images_np, NamedSharding(mesh, P('data', None, None, None)))
    sharded_labels = jax.device_put(labels_np, NamedSharding(mesh, P('data',)))
    ensemble_logits = []
    for model_state in best_models:
        eval_state = TrainState(
            apply_fn=model_instance.apply,
            params=model_state['params'],
            tx=state.tx,
            opt_state=state.opt_state, # Include opt_state
            batch_stats=model_state['batch_stats'],
            step=0 # Added step=0 here
        )
        # eval_step returns a tuple, we only need the loss and accuracy
        test_loss_single, test_acc_single = eval_step(eval_state, (sharded_images, sharded_labels))
        # Append the accuracy for ensembling
        ensemble_logits.append(test_acc_single) # Append accuracy, not logits
            # Average the accuracies for ensembling
    avg_logits = jnp.mean(jnp.stack(ensemble_logits, axis=0), axis=0)
    # Calculate loss and accuracy for the ensemble's prediction
    one_hot_labels = jax.nn.one_hot(sharded_labels, 10)
    # Need to re-calculate loss with the averaged predictions, this requires the actual logits
    # Since we only appended accuracy, we can't calculate loss here accurately.
    # Let's change the eval_step to return logits instead of loss and accuracy for ensembling.
    # Corrected ensemble evaluation: get logits from each model, then average logits
    ensemble_logits_list = []
    for model_state in best_models:
         eval_state = TrainState(
            apply_fn=model_instance.apply,
            params=model_state['params'],
            tx=state.tx, # Include tx
            opt_state=state.opt_state, # Include opt_state
            batch_stats=model_state['batch_stats'],
            step=0 # Added step=0 here
        )
         # eval_step only returns loss and accuracy, not logits. Need a separate function or modify eval_step.
         # Let's create a simple predict function for evaluation.
         def predict_eval(state, images):
              return state.apply_fn(
                  {'params': state.params, 'batch_stats': state.batch_stats},
                  images,
                  train=False
              )
         logits = predict_eval(eval_state, sharded_images)
         ensemble_logits_list.append(logits)
    
    # Average the logits for ensembling
    avg_logits = jnp.mean(jnp.stack(ensemble_logits_list, axis=0), axis=0)
    
    # Calculate loss and accuracy for the ensemble's prediction
    one_hot_labels = jax.nn.one_hot(sharded_labels, 10)
    ensemble_loss = jnp.mean(cross_entropy_loss(avg_logits, one_hot_labels))
    ensemble_acc = jnp.mean(jnp.argmax(avg_logits, -1) == sharded_labels)
    
    
    total_ensemble_test_loss += ensemble_loss.item()
    total_ensemble_test_accuracy += ensemble_acc.item()

avg_ensemble_test_loss = total_ensemble_test_loss / num_test_batches
avg_ensemble_test_accuracy = total_ensemble_test_accuracy / num_test_batches

print(f"Ensemble Test Loss = {avg_ensemble_test_loss:.4f}, Ensemble Test Accuracy = {avg_ensemble_test_accuracy:.4f}")

print("\nEnhanced JAX TPU demonstration complete!")
print("This code successfully trained an enhanced CNN on CIFAR-10 using JAX and TPU acceleration,")
print("incorporating data augmentation, regularization, early stopping, and test set evaluation.")
