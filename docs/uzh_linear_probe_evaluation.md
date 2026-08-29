# UZH PopSign linear probe evaluation

## Protocol

This evaluation follows the linear protocol described in the original SignCLIP paper:

- freeze the pretrained SignCLIP model;
- extract pose embeddings for the complete PopSign train and test splits;
- fit `sklearn.linear_model.LogisticRegression()` with its default settings on all train embeddings;
- rank the classifier scores for the test split and report R@1, R@5, R@10, MedianR, and MeanR.

Unlike 10-shot kNN, the linear probe is supervised and uses the complete downstream training split. The embedding cache and fitted classifier are retained for reproducibility.

## Inputs

- Repository: `/home/faxu/multimodalhugs`
- PopSign dataset: `/home/faxu/scratch/signclip/setup/popsign_pretrain_v1/setup/datasets/default`
- Portable models: `/home/faxu/signclip_eval_models/signclip_global512_step4000_portable`
- Softmax checkpoint: `softmax_global512_step4000`
- Ring-sigmoid checkpoint: `ring_sigmoid_global512_step4000`
- Shared processor: `processor`

Both checkpoints are the global-batch-512, learning-rate-`2e-4`, step-4000 models used by the corresponding kNN evaluation.

## Submission

```bash
cd /home/faxu/multimodalhugs
pixi install
JOB_ID=$(sbatch --parsable scripts/slurm/signclip_eval_popsign_linear_probe.sh)
echo "JOB_ID=${JOB_ID}"
squeue -j "${JOB_ID}" -o "%.18i %.10P %.24j %.8T %.10M %R"
```

## Logs and outputs

- Logs: `/home/faxu/scratch/signclip/logs/signclip-popsign-linear-<JOB_ID>.{out,err}`
- Results: `/home/faxu/scratch/signclip/evals/popsign_linear_global512_step4000/results.{json,tsv}`
- Embedding caches: one `.npz` file per checkpoint in the result directory
- Fitted classifiers: one `.joblib` file per checkpoint in the result directory

```bash
tail -f \
  "/home/faxu/scratch/signclip/logs/signclip-popsign-linear-${JOB_ID}.out" \
  "/home/faxu/scratch/signclip/logs/signclip-popsign-linear-${JOB_ID}.err"
```
