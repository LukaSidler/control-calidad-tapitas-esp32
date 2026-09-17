import os, json, numpy as np, tensorflow as tf
import sys
sys.path.insert(0, '.')
import train as T

SCRATCH = T.SCRATCH

model = tf.keras.models.load_model(os.path.join(SCRATCH, 'tapita_model.keras'))

# --- Representative dataset for int8 calibration: ~150 train images, typical
# (augmented) preprocessing, since that matches what the model will actually see.
def representative_data_gen():
    rng = np.random.default_rng(123)
    recs = list(T.train_recs)
    rng.shuffle(recs)
    for rec in recs[:150]:
        img, _ = T.process_sample(rec, augment=True, rng=rng)
        img = np.expand_dims(img, 0).astype(np.float32)
        yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

out_path = os.path.join(SCRATCH, 'tapita_model_int8.tflite')
with open(out_path, 'wb') as fh:
    fh.write(tflite_model)

size_bytes = os.path.getsize(out_path)
print(f'Saved {out_path}: {size_bytes} bytes ({size_bytes/1024:.1f} KB)')

# also save a float (non-quantized) tflite for size/accuracy comparison reference
converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float = converter2.convert()
out_path_float = os.path.join(SCRATCH, 'tapita_model_float.tflite')
with open(out_path_float, 'wb') as fh:
    fh.write(tflite_float)
size_float = os.path.getsize(out_path_float)
print(f'Saved {out_path_float}: {size_float} bytes ({size_float/1024:.1f} KB)')

print('DONE_CONVERT')
