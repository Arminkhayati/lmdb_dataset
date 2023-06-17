import lmdb, os, json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


sanity_dataset = False
sanity_samples = 10
train_csv_path = "~/khayati/projects/ocr/ocrdata_train.csv"
valid_csv_path = "~/khayati/projects/ocr/ocrdata_valid.csv"
train_df = pd.read_csv(train_csv_path)
train_df = train_df[train_df.text != " "]
valid_df = pd.read_csv(valid_csv_path)
valid_df = valid_df[valid_df.text != " "]
output_path = "~/khayati/projects/ocr/lmdb_dataset/ocrdata"


if sanity_dataset:
    train_df = train_df.sample(sanity_samples, random_state=1, ignore_index=True)
    valid_df = valid_df.sample(sanity_samples, random_state=1, ignore_index=True)
    output_path = output_path + "_sanity"

data = {"train": train_df, "val": valid_df}
dirs = ["train", "val"]

for dir in dirs:
    output_full_path = os.path.join(output_path, dir)
    Path(output_full_path).mkdir(parents=True, exist_ok=True)
    env = lmdb.open(output_full_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    for index, row in data[dir].iterrows():
        image_key = 'image-%09d'.encode() % (index + 1)
        label_key = 'label-%09d'.encode() % (index + 1)
        with open(row["path"], 'rb') as f:
            imageBin = f.read()
        cache[image_key] = imageBin
        cache[label_key] = row["text"].encode()
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            # print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print(f'Created {dir} dataset with {nSamples} samples.')









