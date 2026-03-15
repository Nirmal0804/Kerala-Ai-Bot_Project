import tensorflow as tf
import os

# --- Optional: reduce fragmentation ---
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# --- Limit GPU memory growth ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# --- Check available GPU devices ---
physical_gpus = tf.config.list_physical_devices('GPU')
print("Available GPU devices:", physical_gpus)

# --- Test small tensor allocations ---
try:
    print("Testing GPU with small tensor allocations...")
    tensor_size = 1024  # safe size for 6GB GPU
    a = tf.random.uniform((tensor_size, tensor_size), dtype=tf.float32)
    b = tf.random.uniform((tensor_size, tensor_size), dtype=tf.float32)
    c = tf.matmul(a, b)
    print("Tensor multiplication successful. Shape:", c.shape)
except tf.errors.ResourceExhaustedError:
    print("GPU ran out of memory! Reduce tensor size.")

# --- Print GPU memory info correctly ---
for i, gpu in enumerate(gpus):
    try:
        info = tf.config.experimental.get_memory_info(f"GPU:{i}")
        print(f"GPU:{i} memory info:", info)
    except Exception as e:
        print(f"Could not get memory info for GPU:{i}: {e}")