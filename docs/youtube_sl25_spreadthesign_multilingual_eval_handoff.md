# YouTube-SL-25 -> SpreadTheSign Multilingual Evaluation Handoff

## 任务目标

使用 YouTube-SL-25 预训练得到的 SignCLIP checkpoint，在 SpreadTheSign 上进行多语言 zero-shot sign-to-text retrieval evaluation。

这次不能直接复用此前准备的 SpreadTheSign ASL-English metadata。请结合准备 ASL-English 数据时积累的数据路径、缺失样本、重复文本、pose 长度和 metadata 格式等经验，从完整 SpreadTheSign 数据中重新准备 multilingual evaluation metadata。

## UZH 原始数据结构核验

2026-09-28 通过只读 SSH 浅层检查确认，原始数据位于：

```text
/shares/iict-sp2.ebling.cl.uzh/common/spreadthesign/
├── SperadTheSign.csv
├── sign-mt-poses/
│   ├── sts<hash>.pose
│   └── ...
└── splits/
    └── 1.0.0-uzh/
        ├── all.txt
        ├── train.txt
        ├── val.txt
        ├── test.txt
        └── split.py
```

注意主 CSV 的实际文件名拼写为 `SperadTheSign.csv`。`sign-mt-poses`
是一个扁平目录，pose 文件名采用 `sts<hash>.pose`，没有按语言建立子目录。
本次检查没有递归统计或读取全部 pose 文件。

### CSV schema

```text
pose,videoLanguage,language,text
```

- `pose`：相对于 `sign-mt-poses/` 的 pose 文件名。
- `videoLanguage`：视频侧手语代码，例如 `ssp`、`csl`。
- `language`：文本语言代码，例如 `ro`、`hr`。
- `text`：该语言下的文本标签。

两条已核验样例：

```text
stsfacebeec5a8aaeb6d946890a20ea5d9d.pose,ssp,ro,taxa de drum
sts60f4f9a28b5f6baf204e91695b5c5a6b.pose,csl,hr,kompresirani zrak
```

对应的两个 pose 文件均存在。CSV 本身没有显式的 concept ID 字段，因此在构造
multi-positive retrieval ground truth 之前，不能假设能够直接由一个现成 concept
列聚合跨语言同义项。需要进一步检查相同 pose、多语言文本和重复文本之间的关系。

### Split format

`train.txt`、`val.txt`、`test.txt` 和 `all.txt` 保存的是 CSV 的零起始行索引，
不是 pose 文件名。`split.py` 使用固定随机种子 `3407` 打乱全部行索引，并按
`98% / 1% / 1%` 划分 train、validation 和 test。

`all.txt` 的最后一个索引为 `10728788`，因此总行数为 `10,728,789`。根据
`split.py` 的整数截断逻辑，各 split 的预期行数为：

| Split | Rows |
| --- | ---: |
| train | 10,514,215 |
| validation | 107,287 |
| test | 107,287 |
| all | 10,728,789 |

这些是 metadata 行数，不等于 unique pose、unique text 或 unique concept 数量。

### Language-pair distribution

2026-09-28 对 659 MB CSV 做了一次只读顺序统计，没有扫描 pose 目录。
全部 `10,728,789` 行都有非空的 `language` 和 `videoLanguage`：

| Item | Count |
| --- | ---: |
| Text languages | 30 |
| Sign-language codes | 41 |
| Text/sign combinations | 1,206 |

Prompt 的顺序应为 `<language> <videoLanguage>`，即
`<text_language> <sign_language>`。最常见的组合如下：

| Text language | Sign language | Prompt | Rows |
| --- | --- | --- | ---: |
| Swedish (`sv`) | Swedish Sign Language (`swl`) | `<sv> <swl>` | 22,996 |
| Lithuanian (`lt`) | Lithuanian Sign Language (`lls`) | `<lt> <lls>` | 20,655 |
| Italian (`it`) | Italian Sign Language (`ise`) | `<it> <ise>` | 19,615 |
| German (`de`) | Swedish Sign Language (`swl`) | `<de> <swl>` | 18,637 |
| English (`en`) | Swedish Sign Language (`swl`) | `<en> <swl>` | 18,599 |
| Spanish (`es`) | Swedish Sign Language (`swl`) | `<es> <swl>` | 18,401 |
| Danish (`da`) | Danish Sign Language (`dsl`) | `<da> <dsl>` | 18,184 |
| German (`de`) | German Sign Language (`gsg`) | `<de> <gsg>` | 18,135 |
| German (`de`) | Austrian Sign Language (`asq`) | `<de> <asq>` | 17,713 |
| Lithuanian (`lt`) | Swedish Sign Language (`swl`) | `<lt> <swl>` | 17,644 |
| German (`de`) | Lithuanian Sign Language (`lls`) | `<de> <lls>` | 17,597 |
| Italian (`it`) | Swedish Sign Language (`swl`) | `<it> <swl>` | 17,418 |
| Swedish (`sv`) | Lithuanian Sign Language (`lls`) | `<sv> <lls>` | 17,127 |
| Spanish (`es`) | Lithuanian Sign Language (`lls`) | `<es> <lls>` | 17,045 |
| Swedish (`sv`) | Danish Sign Language (`dsl`) | `<sv> <dsl>` | 16,955 |
| English (`en`) | Lithuanian Sign Language (`lls`) | `<en> <lls>` | 16,871 |
| Swedish (`sv`) | Italian Sign Language (`ise`) | `<sv> <ise>` | 16,835 |
| Croatian (`hr`) | Croatian Sign Language (`csq`) | `<hr> <csq>` | 16,807 |
| German (`de`) | Italian Sign Language (`ise`) | `<de> <ise>` | 16,795 |
| Estonian (`et`) | Estonian Sign Language (`eso`) | `<et> <eso>` | 16,791 |

The most frequent sign-language codes by row count are `swl` (416,964),
`lls` (386,345), `ise` (383,658), `dsl` (375,537), `gsg` (363,763),
`asq` (344,559), `rsl` (340,719), `lsl` (339,746), `tsm` (338,112), and
`csq` (337,100). The most frequent text languages are `de` (466,302), `es`
(463,110), `sv` (463,032), `en` (456,913), `lt` (442,390), and `it`
(441,278).

No single pair dominates: even `<sv> <swl>`, the largest pair, is only about
0.21% of all metadata rows. The frequent cross-language pairs, such as
`<de> <swl>` and `<en> <swl>`, indicate that the CSV expands sign videos
against translations in several spoken languages rather than containing only
country-matched text/sign pairs. Evaluation metadata should therefore group
explicitly by `(language, videoLanguage)` and must not interpret every CSV row
as an independent sign video.

## Recommended evaluation panel aligned with YouTube-SL-25

The existing YouTube-SL-25 language summary was compared with the complete
SpreadTheSign CSV. The table below records exact or directly compatible prompt
pairs. SpreadTheSign `all rows` are the full pair-specific metadata counts;
`official test rows` are the rows selected by `splits/1.0.0-uzh/test.txt`.

| Prompt | Approx. YouTube-SL-25 sign hours | YouTube-SL-25 VTT files with this prompt | SpreadTheSign all rows | Official test rows | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| `<en> <ase>` | 1,394 | 12,288 | 12,490 | 119 | Dominant-resource anchor |
| `<en> <ins>` | 209 | 1,310 | 6,086 | 58 | High-resource non-Western sign language |
| `<pl> <pso>` | 137 | 1,554 | 13,892 | 142 | Medium-resource, matched spoken/sign language |
| `<de> <gsg>` | 108 | 841 | 18,135 | 176 | Medium-resource German pair; previously selected |
| `<en> <bfi>` | 74 | 373, plus 649 `<en-GB> <bfi>` files | 16,026 | 136 | English-text control with a non-ASL sign language |
| `<it> <ise>` | 63 | 869 | 19,615 | 188 | Lower-resource matched pair with strong evaluation size |
| `<ja> <jsl>` | 62 | 1,065 | 9,719 | 107 | Different writing system and typological setting |

These seven pairs form the recommended core panel. They cover the dominant
ASL setting, high- and medium-resource training languages, an English-text
control across different sign languages, a lower-resource European pair, and
a non-Latin writing system.

Useful extension pairs are:

| Prompt | Approx. training hours | Training VTT files | SpreadTheSign all rows | Official test rows | Reason |
| --- | ---: | ---: | ---: | ---: | --- |
| `<ru> <rsl>` | 60 | 704 | 15,371 | 151 | Cyrillic and lower-resource transfer |
| `<fr> <fsl>` | 49 | 865 | 13,271 | 124 | Low-resource floor |
| `<pt> <bzs>` | 101 | 341, plus 478 `<pt-BR> <bzs>` files | 8,330 | 82 | Brazilian Sign Language setting |
| `<en> <asf>` | 67 | 702 | 11,743 | 118 | Second English-text sign-language control |
| `<sv> <swl>` | below the displayed top-hour group | 352 | 22,996 | not measured in the targeted pass | Largest SpreadTheSign pair |

Two training-heavy pairs are poor primary choices for this evaluation:

- `<en> <ils>` has 1,548 YouTube-SL-25 VTT files but only 868 complete
  SpreadTheSign rows and 9 official test rows, so its metrics would be noisy.
- `<hu> <hsh>` has 1,685 YouTube-SL-25 VTT files, but the exact pair has no
  rows in this SpreadTheSign CSV.

For the primary zero-shot comparison, use all valid rows for each selected
pair after pose readability checks, `max_frames <= 256` filtering, and
pair-specific deduplication. This matches the historical ASL-English setup:
`<en> <ase>` has 12,490 raw rows and previously produced 12,383 valid max-256
examples. The official 1% test subsets contain only 58--188 examples for most
core pairs and are better treated as a secondary sensitivity check than as the
sole multilingual benchmark.

## Metadata 准备要求

1. 检查 SpreadTheSign 完整数据目录，统计实际具备可读 pose 和文本标注的语言。
2. 将可用语言与 YouTube-SL-25 预训练涵盖的语言取交集，并记录：
   - language code
   - sign language 名称
   - 样本数量
   - 可用 pose 数量
   - 缺失或损坏数量
3. 沿用此前 ASL-English metadata 的字段格式，使其可以直接由 MultimodalHugs Dataset/Processor 加载。
4. 每条记录至少需要正确关联：
   - pose 路径
   - 对应文本
   - language 或 language code
   - concept 或 sample identifier（如果原数据提供）
5. 复用此前的数据处理经验：
   - 剔除不存在或无法解析的 pose
   - 处理重复路径和重复记录
   - 检查空文本
   - 输入最多保留 256 个 sign frames
   - 避免再次出现输入长度 273 与模型位置容量 258 不匹配的问题
6. 特别检查同一 concept 在不同语言中产生多个正确文本的问题。当前 retrieval evaluator 默认按对角线定义唯一正样本，因此：
   - 优先分别对每种语言运行 retrieval evaluation
   - 汇总每种语言的结果并计算 macro average
   - 不要直接把所有语言混合进一个 diagonal-only retrieval matrix，除非实现了 multi-positive evaluation
7. 如果各语言规模差异很大，同时报告：
   - 每种语言的独立结果
   - macro average
   - 总样本量
   - 可选的 sample-weighted average

## 模型信息

- Objective：softmax/CLIP
- Pretraining data：YouTube-SL-25 clean max256
- Training setup：single GPU，batch size 128，learning rate `5e-5`
- Checkpoint：

  ```text
  /home/faxu/scratch/signclip/runs/youtube_sl25_clean_max256_softmax_b128_130k/train/checkpoint-36000
  ```

- Processor：

  ```text
  /home/faxu/scratch/signclip/setup/youtube_sl25_clean_max256_v1/setup/sign_clip_processor
  ```

- UZH repository：

  ```text
  /home/faxu/multimodalhugs
  ```

- Git branch：

  ```text
  codex/youtube-sl25-pretrain
  ```

## 已完成的模型加载核验

该 checkpoint 已成功完成 PopSign zero-shot evaluation，可用于核对模型和评估管线是否加载正确：

| Metric | Result |
| --- | ---: |
| R@1 | 11.90% |
| R@5 | 32.54% |
| R@10 | 43.64% |
| MedianR | 15 |
| MeanR | 38.12 |

结果文件：

```text
/home/faxu/scratch/signclip/evals/popsign_zeroshot_youtube_sl25_current_best/checkpoint-36000/train/eval_results.json
```

## SpreadTheSign 评估设置

- 不训练模型，只执行 evaluation。
- 使用上述 YouTube-SL-25 processor，不要换用其他 processor。
- Retrieval direction 使用 `v2t`。
- 建议 `per_device_eval_batch_size: 128`；显存不足时再降低。
- 每个语言单独运行和保存结果。
- 输入在进入模型前必须限制为最多 256 个 sign frames。

每种语言记录以下指标：

- R@1
- R@5
- R@10
- MedianR
- MeanR
- eval_loss
- eval_samples

建议结果根目录：

```text
/home/faxu/scratch/signclip/evals/spreadthesign_multilingual_zeroshot_youtube_sl25_checkpoint36000
```

每个语言对的 `eval_results.json` 位于：

```text
/home/faxu/scratch/signclip/evals/spreadthesign_multilingual_zeroshot_youtube_sl25_checkpoint36000/<text>_<sign>/eval_results.json
```

## 已准备的实现

2026-09-28 已加入以下文件：

- `scripts/prepare_spreadthesign_multilingual_eval.py`
  - 对 659 MB 主 CSV 只做一次顺序读取。
  - 只打开所选语言对引用的 pose，不递归扫描 `sign-mt-poses/`。
  - 完整解析选中的 pose，显式剔除搬运截断、其他不可解析文件和超过
    256 帧的样本；这一步使用 4 个 worker，避免对共享存储施加过高并发。
  - 写入统一的 `master.tsv`、清理报告 `rejected.tsv`、`summary.json`，
    以及每个语言对独立的 `pairs/<text>_<sign>/test.tsv`。
- `scripts/evaluation/evaluate_signclip_v2t_fast.py`
  - 可直接读取每个语言对的 TSV，不需要复制多份 setup YAML 或建立多份
    Hugging Face dataset cache。
  - 文本候选按完整 prompt + text 去重，并分块计算检索分数，避免构造完整
    sign-by-text 分数矩阵。
- `scripts/evaluation/summarize_spreadthesign_multilingual.py`
  - 汇总每个语言对结果，生成 pair-level TSV、macro average 和
    sample-weighted average。
- `scripts/slurm/signclip_prepare_spreadthesign_multilingual_eval.sh`
  - CPU metadata 清理任务。
- `scripts/slurm/signclip_eval_spreadthesign_multilingual_youtube_sl25.sh`
  - 单 GPU 顺序评估任务；已存在的语言对结果默认跳过，支持超时后续跑。

默认 metadata 输出：

```text
/home/faxu/scratch/signclip/metadata/spreadthesign_multilingual_youtube_sl25
```

默认结果输出：

```text
/home/faxu/scratch/signclip/evals/spreadthesign_multilingual_zeroshot_youtube_sl25_checkpoint36000
```

### Smoke test

先在独立目录准备并评估 `<en> <ase>` 的 32 条样本：

```bash
cd /home/faxu/multimodalhugs

SMOKE_METADATA=/home/faxu/scratch/signclip/metadata/spreadthesign_multilingual_smoke
SMOKE_RESULTS=/home/faxu/scratch/signclip/evals/spreadthesign_multilingual_smoke

PREP_JOB=$(sbatch --parsable \
  --export=ALL,PAIR_KEYS_OVERRIDE=en_ase,LIMIT_PER_PAIR=32,METADATA_ROOT="$SMOKE_METADATA" \
  scripts/slurm/signclip_prepare_spreadthesign_multilingual_eval.sh)

EVAL_JOB=$(sbatch --parsable \
  --dependency=afterok:"$PREP_JOB" \
  --export=ALL,PAIR_KEYS_OVERRIDE=en_ase,METADATA_ROOT="$SMOKE_METADATA",RESULTS_ROOT="$SMOKE_RESULTS" \
  scripts/slurm/signclip_eval_spreadthesign_multilingual_youtube_sl25.sh)

echo "PREP_JOB=$PREP_JOB EVAL_JOB=$EVAL_JOB"
```

### Full evaluation

Smoke 通过后，可让 metadata 和 GPU 任务自动按成功依赖衔接：

```bash
cd /home/faxu/multimodalhugs

PREP_JOB=$(sbatch --parsable \
  scripts/slurm/signclip_prepare_spreadthesign_multilingual_eval.sh)

EVAL_JOB=$(sbatch --parsable \
  --dependency=afterok:"$PREP_JOB" \
  scripts/slurm/signclip_eval_spreadthesign_multilingual_youtube_sl25.sh)

echo "PREP_JOB=$PREP_JOB EVAL_JOB=$EVAL_JOB"
```

检查状态和输出：

```bash
squeue -j "$PREP_JOB,$EVAL_JOB" -o "%.18i %.34j %.10T %.10M %R"
sacct -X -j "$PREP_JOB,$EVAL_JOB" \
  --format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList

tail -n 100 "/home/faxu/scratch/signclip/logs/signclip-prep-sts-multilingual-${PREP_JOB}.out"
tail -n 160 "/home/faxu/scratch/signclip/logs/signclip-prep-sts-multilingual-${PREP_JOB}.err"
tail -n 120 "/home/faxu/scratch/signclip/logs/signclip-eval-sts-multilingual-${EVAL_JOB}.out"
tail -n 200 "/home/faxu/scratch/signclip/logs/signclip-eval-sts-multilingual-${EVAL_JOB}.err"

python -m json.tool \
  /home/faxu/scratch/signclip/metadata/spreadthesign_multilingual_youtube_sl25/summary.json
python -m json.tool \
  /home/faxu/scratch/signclip/evals/spreadthesign_multilingual_zeroshot_youtube_sl25_checkpoint36000/summary.json
```

## 可参考的旧实现

- `configs/signclip_eval_popsign_zeroshot_youtube_sl25_current_best.server.yaml`
- `scripts/slurm/signclip_eval_popsign_zeroshot_youtube_sl25_current_best.sh`

新流程直接读取 pair-specific TSV，因此不再需要为每个语言对复制一份 YAML。
模型 checkpoint 和 processor 路径保持不变。

## 建议执行顺序

1. 提交 32 条 `<en> <ase>` 的 metadata smoke 与依赖 evaluation。
2. 核对两个任务均为 `COMPLETED`，并查看 smoke `eval_results.json`。
3. 提交完整 metadata 清理任务。
4. 检查每个语言对 `kept_rows`、`unique_poses`、`unique_texts` 和拒绝原因。
5. 让依赖任务顺序执行 7 个语言对的完整 evaluation。
6. 核对 pair-level 结果以及 `summary.json` 中的 macro/weighted 汇总。
