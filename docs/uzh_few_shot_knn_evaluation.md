# UZH PopSign 10-shot kNN evaluation

## Goal

Compare the pose representations from the following matched YouTube-ASL runs:

- Softmax, global batch 512, learning rate `2e-4`, checkpoint step 4000.
- Ring sigmoid, global batch 512, learning rate `2e-4`, checkpoint step 4000.

Both checkpoints have processed 2.048 million training examples. On UZH they are stored under:

```text
/home/faxu/signclip_eval_models/signclip_global512_step4000_portable/
```

## Protocol

The support set is sampled once with seed 42 and reused for both models. For every PopSign test class, at most ten examples are sampled from the training split. A StandardScaler is fitted only on these support embeddings. Classification uses uniform-vote Euclidean kNN on the test pose embeddings and reports R@1, R@5, R@10, MedianR, and MeanR.

The [SignCLIP paper](https://aclanthology.org/2024.emnlp-main.518/) and its
[current public evaluation script](https://github.com/J22Melody/fairseq/blob/main/examples/MMPT/test_recognition_few_shot_knn.py)
differ in their number of neighbours:

- `paper`: number of neighbours equals the number of evaluated classes.
- `repo`: number of neighbours is `round(sqrt(number of support examples))`.

The evaluator reports both. Here, "10-shot" describes the number of support examples per class; it does not mean ten neighbours.

## UZH paths and launch

The prepared PopSign dataset is expected at:

```text
/home/faxu/scratch/signclip/setup/popsign_pretrain_v1/setup/datasets/default
```

After updating the main MultimodalHugs checkout, inspect the paths without starting a job:

```bash
for p in \
  /home/faxu/signclip_eval_models/signclip_global512_step4000_portable/{softmax_global512_step4000,ring_sigmoid_global512_step4000,processor} \
  /home/faxu/scratch/signclip/setup/popsign_pretrain_v1/setup/datasets/default; do
  test -e "$p" && echo "OK  $p" || echo "MISS $p"
done
```

Also confirm that the prepared dataset exposes the expected splits and label column:

```bash
cd /home/faxu/multimodalhugs
pixi run python - <<'PY'
from datasets import load_from_disk

path = "/home/faxu/scratch/signclip/setup/popsign_pretrain_v1/setup/datasets/default"
dataset = load_from_disk(path)
print(dataset)
for split in ("train", "test"):
    print(split, dataset[split].column_names, dataset[split][0])
PY
```

Launch:

```bash
cd /home/faxu/multimodalhugs
sbatch scripts/slurm/signclip_eval_popsign_knn_10shot.sh
```

Outputs include a fixed `support_manifest.tsv`, cached embeddings for each model, `results.json`, and `results.tsv`.
