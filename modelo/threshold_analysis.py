import os, json
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

# Directorio local de trabajo (no versionado): ajustar a gusto o setear
# la variable de entorno TAPITAS_DATASET.
BASE = os.environ.get('TAPITAS_DATASET', os.path.join(os.path.dirname(__file__), '..', 'dataset_tapitas'))
MODELO = os.path.join(BASE, 'modelo')
CROP_SIZE = 1120
IMG_SIZE = 128

geom = {r['file']: r for r in json.load(open(os.path.join(MODELO, 'geometry_final.json')))}
plan = json.load(open(os.path.join(MODELO, 'split_plan_final2.json')))

LABEL = {'sana': 0, 'rota': 1}

def build_records(split):
    recs = []
    for cls in ['sana', 'rota']:
        for fname in plan[cls][split]:
            g = geom[fname]
            recs.append({
                'path': os.path.join(BASE, cls, fname),
                'cx': g['cx'], 'cy': g['cy'], 'r': g['r'],
                'label': LABEL[cls], 'cls': cls, 'file': fname,
            })
    return recs

test_recs = build_records('test')

def fixed_crop(img, cx, cy, crop_size):
    H, W = img.shape[:2]
    x0 = cx - crop_size / 2
    y0 = cy - crop_size / 2
    x0 = min(max(x0, min(0, W - crop_size)), max(0, W - crop_size))
    y0 = min(max(y0, min(0, H - crop_size)), max(0, H - crop_size))
    x0i, y0i = int(round(x0)), int(round(y0))
    x1i, y1i = x0i + crop_size, y0i + crop_size
    pad_left = max(0, -x0i); pad_top = max(0, -y0i)
    pad_right = max(0, x1i - W); pad_bottom = max(0, y1i - H)
    sx0, sy0 = max(0, x0i), max(0, y0i)
    sx1, sy1 = min(W, x1i), min(H, y1i)
    crop = img[sy0:sy1, sx0:sx1]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
    return crop

def process_sample(rec):
    img = cv2.imread(rec['path'])
    if img is None:
        raise FileNotFoundError(rec['path'])
    cx, cy = rec['cx'], rec['cy']
    crop = fixed_crop(img, cx, cy, CROP_SIZE)
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = gray[:, :, None]
    return gray, rec['label']

X, y = [], []
for rec in test_recs:
    img, lab = process_sample(rec)
    X.append(img); y.append(lab)
X = np.stack(X); y = np.array(y)
print('test set:', X.shape, 'sana(0)/rota(1):', (y == 0).sum(), (y == 1).sum())

interpreter = tflite.Interpreter(model_path=os.path.join(MODELO, 'tapita_model_int8.tflite'))
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
in_scale, in_zp = inp['quantization']
out_scale, out_zp = out['quantization']

probs = []
for i in range(len(X)):
    x = X[i:i+1]
    x_q = np.round(x / in_scale + in_zp).astype(np.int8)
    interpreter.set_tensor(inp['index'], x_q)
    interpreter.invoke()
    o = interpreter.get_tensor(out['index'])
    prob = (o.astype(np.float32) - out_zp) * out_scale
    probs.append(float(prob.flatten()[0]))
probs = np.array(probs)

def metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=acc, precision=precision, recall=recall, f1=f1)

print(f"\n{'thr':>5} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  tp tn fp fn   fn_files")
print('-' * 90)
for thr in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]:
    y_pred = (probs > thr).astype(int)
    m = metrics(y, y_pred)
    fn_files = [test_recs[i]['file'] for i in range(len(y)) if y[i] == 1 and y_pred[i] == 0]
    fp_files = [test_recs[i]['file'] for i in range(len(y)) if y[i] == 0 and y_pred[i] == 1]
    print(f"{thr:5.2f} {m['accuracy']*100:5.1f}% {m['precision']*100:5.1f}% {m['recall']*100:5.1f}% {m['f1']:6.3f}  "
          f"{m['tp']:2d} {m['tn']:2d} {m['fp']:2d} {m['fn']:2d}   FN={fn_files} FP={fp_files}")

print('\nRaw probabilities (rota=1 target), sorted:')
order = np.argsort(probs)
for i in order:
    print(f"  {test_recs[i]['file']:16s} label={y[i]} prob_rota={probs[i]:.4f}")
