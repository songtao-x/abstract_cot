import json

import json

file = "cd4_test.jsonl"
eval_f = "cd4_eval.jsonl"

# ds = []
# with open(file, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
#         ds.append(json.loads(line))

# eval_ds = ds[:500]

# with open(eval_f, "w", encoding="utf-8") as f:
#     for item in eval_ds:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")

# test_ds = ds[500:]

# with open(file, "w", encoding="utf-8") as f:
#     for item in test_ds:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")


train_f = "cd4_train_all.jsonl"
ds = []
with open(train_f, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ds.append(json.loads(line))

ds = ds[:2000]

with open("cd4_train.jsonl", "w", encoding="utf-8") as f:
    for item in ds:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

