import os, json, random
import numpy as np
import cv2
import tensorflow as tf

# Directorios locales de trabajo (no versionados): ajustar a gusto o setear
# las variables de entorno TAPITAS_SCRATCH / TAPITAS_DATASET.
SCRATCH = os.environ.get('TAPITAS_SCRATCH', os.path.join(os.path.dirname(__file__), 'scratch'))
BASE = os.environ.get('TAPITAS_DATASET', os.path.join(os.path.dirname(__file__), '..', 'dataset_tapitas'))
CROP_SIZE = 1120
JITTER = 70
IMG_SIZE = 128
BATCH = 16
SEED = 42

geom = {r['file']: r for r in json.load(open(os.path.join(SCRATCH, 'geometry_final.json')))}
plan = json.load(open(os.path.join(SCRATCH, 'split_plan_final2.json')))

LABEL = {'sana': 0, 'rota': 1}  # 0=sana(healthy) 1=rota(broken)

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

train_recs = build_records('train')
val_recs = build_records('val')
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

def process_sample(rec, augment, rng):
    img = cv2.imread(rec['path'])
    cx, cy, r = rec['cx'], rec['cy'], rec['r']
    if augment:
        jx = rng.uniform(-JITTER, JITTER)
        jy = rng.uniform(-JITTER, JITTER)
        zoom = rng.uniform(0.95, 1.05)
        crop_sz = int(round(CROP_SIZE * zoom))
    else:
        jx = jy = 0.0
        crop_sz = CROP_SIZE
    crop = fixed_crop(img, cx + jx, cy + jy, crop_sz)

    if augment:
        angle = rng.uniform(0, 360)
        M = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), angle, 1.0)
        crop = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < 0.5:
            crop = cv2.flip(crop, 1)
        if rng.random() < 0.5:
            crop = cv2.flip(crop, 0)
        alpha = rng.uniform(0.85, 1.15)  # contrast
        beta = rng.uniform(-20, 20)      # brightness
        crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    # Grayscale, not color: cap color carries no label signal (both classes span
    # every color; the separate HSV rule-based stage already owns color), but
    # sana's train/val split is partitioned BY color to avoid identity leakage,
    # so train and val see disjoint color palettes for that class. Feeding color
    # let the model shortcut on it instead of the actual hole/break shape -- fit
    # train (0.86 acc) but collapsed on val (0.43). Hue-jitter augmentation was
    # tried first but fighting color at every step alongside rotation/flip/zoom
    # was too much simultaneous variation for this tiny net to learn from in the
    # epoch budget; stripping color outright is simpler and removes the shortcut
    # at the source (as a bonus, a 1-channel input is also smaller on-device).
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = gray[:, :, None]
    return gray, rec['label']

def make_dataset(recs, augment, shuffle, repeat=False):
    def gen():
        rng = np.random.default_rng()
        idx = list(range(len(recs)))
        if shuffle:
            rng.shuffle(idx)
        for i in idx:
            yield process_sample(recs[i], augment, rng)
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )
    if repeat:
        ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(buffer_size=min(300, len(recs)))
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model():
    # No BatchNorm: with only ~14 steps/epoch, its running stats never converge
    # well and the train/inference mismatch tanks val accuracy (verified empirically).
    # Filters wider than a first guess (16/32/64/128, not 8/16/32/64): full 0-360
    # rotation augmentation means the net must learn edge/blob detectors at many
    # orientations, which a very narrow first layer doesn't have room for.
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = tf.keras.layers.Rescaling(1. / 255)(inputs)
    for filters in [16, 32, 64, 128]:
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    # Max, not average pooling: the task is "is there a localized hole/break
    # anywhere on the cap", not a global image statistic. Averaging dilutes a
    # small dark blob against a lot of uniform "no damage" surface; max pooling
    # asks the exact right question and should learn far faster.
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name='TapitaNet')

def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print(f'train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}')
    import collections
    print('train label balance:', collections.Counter(r['label'] for r in train_recs))

    steps_per_epoch = -(-len(train_recs) // BATCH)
    val_steps = -(-len(val_recs) // BATCH)

    train_ds = make_dataset(train_recs, augment=True, shuffle=True, repeat=True)
    val_ds = make_dataset(val_recs, augment=False, shuffle=False, repeat=True)
    test_ds = make_dataset(test_recs, augment=False, shuffle=False, repeat=False)

    model = build_model()
    model.summary()
    print('Total params:', model.count_params())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=80,
                         steps_per_epoch=steps_per_epoch, validation_steps=val_steps,
                         callbacks=callbacks, verbose=2)

    print('\n=== Evaluación en TEST (modelo float) ===')
    results = model.evaluate(test_ds, verbose=0)
    for name, val in zip(model.metrics_names, results):
        print(f'{name}: {val:.4f}')

    model.save(os.path.join(SCRATCH, 'tapita_model.keras'))
    with open(os.path.join(SCRATCH, 'history.json'), 'w') as fh:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, fh)

    print('DONE_TRAINING')

if __name__ == '__main__':
    main()
