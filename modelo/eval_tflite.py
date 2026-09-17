import os, sys, numpy as np, tensorflow as tf
sys.path.insert(0, '.')
import train as T

SCRATCH = T.SCRATCH

def build_test_arrays():
    rng = np.random.default_rng(7)
    X, y = [], []
    for rec in T.test_recs:
        img, lab = T.process_sample(rec, augment=False, rng=rng)
        X.append(img); y.append(lab)
    return np.stack(X), np.array(y)

def metrics(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(n=len(y_true), tp=tp, tn=tn, fp=fp, fn=fn, accuracy=acc, precision=precision, recall=recall, f1=f1)

X_test, y_test = build_test_arrays()
print('test set:', X_test.shape, 'label balance sana(0)/rota(1):', (y_test == 0).sum(), (y_test == 1).sum())

# --- Float Keras model ---
model = tf.keras.models.load_model(os.path.join(SCRATCH, 'tapita_model.keras'))
preds_float = model(X_test, training=False).numpy().flatten()
y_pred_float = (preds_float > 0.5).astype(int)
m_float = metrics(y_test, y_pred_float)
print('\n=== FLOAT Keras model on TEST ===')
for k, v in m_float.items():
    print(f'  {k}: {v}')

# --- int8 TFLite model ---
interpreter = tf.lite.Interpreter(model_path=os.path.join(SCRATCH, 'tapita_model_int8.tflite'))
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print('\nTFLite input dtype/scale/zero_point:', inp['dtype'], inp['quantization'])
print('TFLite output dtype/scale/zero_point:', out['dtype'], out['quantization'])

in_scale, in_zp = inp['quantization']
out_scale, out_zp = out['quantization']

y_pred_tfl = []
probs_tfl = []
for i in range(len(X_test)):
    x = X_test[i:i+1]
    x_q = np.round(x / in_scale + in_zp).astype(np.int8)
    interpreter.set_tensor(inp['index'], x_q)
    interpreter.invoke()
    o = interpreter.get_tensor(out['index'])
    prob = (o.astype(np.float32) - out_zp) * out_scale
    probs_tfl.append(float(prob.flatten()[0]))
    y_pred_tfl.append(int(prob.flatten()[0] > 0.5))

m_tfl = metrics(y_test, y_pred_tfl)
print('\n=== INT8 TFLite model on TEST ===')
for k, v in m_tfl.items():
    print(f'  {k}: {v}')

size_int8 = os.path.getsize(os.path.join(SCRATCH, 'tapita_model_int8.tflite'))
size_float_tfl = os.path.getsize(os.path.join(SCRATCH, 'tapita_model_float.tflite'))
print(f'\nint8 tflite size: {size_int8} bytes ({size_int8/1024:.1f} KB)')
print(f'float tflite size: {size_float_tfl} bytes ({size_float_tfl/1024:.1f} KB)')

print('DONE_EVAL')
