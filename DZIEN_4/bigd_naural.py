"""
Dask -> tf.data -> Keras
- Czyta wiele plików (CSV/Parquet) jako jeden logiczny zbiór (lazy).
- Liczy statystyki (mean/std) w Dasku.
- Strumieniuje batche z partycji Daska do TensorFlow.
- Uczy prostą sieć MLP.

Wymagania:
  pip install dask[complete] pandas pyarrow fsspec s3fs tensorflow
"""

import os
import math
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import numpy as np
import tensorflow as tf

# -----------------------------
# 1) Dask – klaster lokalny
# -----------------------------
cluster = LocalCluster(
    n_workers=os.cpu_count() // 2 or 1,
    threads_per_worker=2,
    memory_limit="2GB"
)
client = Client(cluster)
print(client)

# -----------------------------
# 2) Ścieżki do danych
#    * Wybierz JEDNĄ z opcji:
# -----------------------------
# A) Lokalnie (Parquet/CSV rozbite na wiele plików)
DATA_PATH = "data/train-*.parquet"   # lub "data/train-*.csv"

# B) Amazon S3 (wymaga s3fs)
# DATA_PATH = "s3://twoj-bucket/train-*.parquet"

# -----------------------------
# 3) Definicja kolumn
# -----------------------------
FEATURE_COLS = [f"f{i}" for i in range(50)]   # 50 cech numerycznych (przykład)
LABEL_COL = "label"                           # kolumna z etykietą (0/1 lub k-klas)
FILE_FORMAT = "parquet"                       # zmień na "csv" jeśli używasz CSV
BATCH_SIZE = 2048

# -----------------------------
# 4) Wczytanie danych (lazy)
# -----------------------------
if FILE_FORMAT == "parquet":
    ddf = dd.read_parquet(
        DATA_PATH,
        columns=FEATURE_COLS + [LABEL_COL],
        engine="pyarrow"
    )
else:
    ddf = dd.read_csv(
        DATA_PATH,
        usecols=FEATURE_COLS + [LABEL_COL],
        # przy CSV – dopasuj separatory/typu dtype jeśli trzeba
    )

# W opcjonalnym case’ie – wymuś dtypes (ważne gdy CSV ma niejednolite typy)
for c in FEATURE_COLS:
    ddf[c] = ddf[c].astype("float32")
ddf[LABEL_COL] = ddf[LABEL_COL].astype("int32")

# -----------------------------
# 5) Statystyki do normalizacji
#    (liczone rozproszenie, a nie w RAM naraz)
# -----------------------------
means = ddf[FEATURE_COLS].mean().compute()      # pandas.Series
stds  = ddf[FEATURE_COLS].std().compute().replace(0.0, 1.0)  # unikamy dzielenia przez 0
n_total = int(ddf.shape[0].compute())
n_features = len(FEATURE_COLS)

print(f"Liczba rekordów: {n_total:,}")
print(f"Liczba cech: {n_features}")

# -----------------------------
# 6) Generator batchy z Daska
#    – iteruje po partycjach
#    – normalizuje wg mean/std
#    – dzieli na batche
# -----------------------------
def make_batch_generator(ddf, feature_cols, label_col, batch_size, means, stds):
    # to_delayed() zwraca listę "zadanych" (po jednej na partycję)
    partitions = ddf.to_delayed()
    for part_delayed in partitions:
        # materializacja JEDNEJ partycji (pandas.DataFrame) – nie całego zbioru
        part_df = part_delayed.compute()  # bezpieczne: to tylko część danych
        # fillna i normalizacja
        X = part_df[feature_cols].fillna(0.0).astype("float32").to_numpy(copy=False)
        # (X - mean) / std dla każdej kolumny
        # dopasuj kształt: (n,) -> (1,n) dla broadcastu
        X = (X - means.values.reshape(1, -1).astype("float32")) / stds.values.reshape(1, -1).astype("float32")
        y = part_df[label_col].astype("int32").to_numpy(copy=False)

        # krojenie na batche
        n = X.shape[0]
        if n == 0:
            continue
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield X[start:end], y[start:end]

# -----------------------------
# 7) tf.data.Dataset.from_generator
#    – strumień batchy dla Keras
# -----------------------------
output_signature = (
    tf.TensorSpec(shape=(None, n_features), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int32),
)

train_ds = tf.data.Dataset.from_generator(
    lambda: make_batch_generator(ddf, FEATURE_COLS, LABEL_COL, BATCH_SIZE, means, stds),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

# W razie potrzeby można dodać cache() (np. do RAM/dysku) – ale uważaj przy bardzo dużych danych

# -----------------------------
# 8) Prosty model Keras (MLP)
# -----------------------------
# (opcjonalnie) jeżeli używasz GPU – włącz "memory growth"
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

num_classes = int(ddf[LABEL_COL].max().compute() + 1)  # zakładamy etykiety 0..K-1

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_features,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax" if num_classes > 1 else "sigmoid"),
])

loss = "sparse_categorical_crossentropy" if num_classes > 1 else "binary_crossentropy"
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])

# -----------------------------
# 9) Trenowanie
#    – steps_per_epoch liczymy na podstawie n_total i batch_size
# -----------------------------
steps_per_epoch = max(1, math.ceil(n_total / BATCH_SIZE))

history = model.fit(
    train_ds,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
)

# -----------------------------
# 10) Zapis modelu
# -----------------------------
model.save("model_dask_tf.keras")
print("Model zapisany do: model_dask_tf.keras")
