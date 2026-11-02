# HE-Gene-RL

[English](README.md)

HE-Gene-RL 是一套根据 H&E 图像特征与空间上下文预测细胞表达基因集合的方法。条件预测器融合中心细胞、邻居图像和此前已选基因的预测计数；PPO 策略决定下一步选择哪个基因，或者停止。

**推理不需要真实表达计数。** 输入为预先提取好的图像特征和坐标。本项目不包含切片图像预处理或 UNI 特征提取。

**当前状态：项目整理已完成，但这次整理没有运行测试、训练或推理。** 原实验的训练和评价已经完成，其历史结果记录在 [docs/results.md](docs/results.md)。这些历史结果不能当作当前独立项目已经验证通过的证明。

## 方法与训练阶段

```text
中心 H&E 特征＋邻居图像与坐标＋此前已选基因及预测计数
                          ↓
                    条件表达预测器
                          ↓
               表达状态概率＋连续计数预测
                          ↓
                  PPO 选择基因或停止
                          ↓
          选中基因的预测计数回填历史，进入下一步
```

| 阶段 | 输入历史中的计数 | 更新的网络 | 真实表达用途 |
|---|---|---|---|
| 条件预测预训练 | 随机选中基因的真实计数，部分样本只给身份 | 预测器 | 构造预训练历史、计算监督损失 |
| 预测历史适配 | 模型自己的预测计数 | 预测器 | 计算监督损失 |
| PPO | 模型自己的预测计数 | Actor/Critic，预测器冻结 | 轨迹完成后计算奖励 |
| 推理与评价 | 模型自己的预测计数 | 不更新 | 推理不读取；单独评价时使用 |

预测器有两个共享主干的输出头：一个预测 `计数=0 / 计数=1 / 计数≥2` 的概率，另一个预测 `log(1+计数)`。监督损失为三分类 CE 加 `0.1 × log计数MSE`；计数为 1 的基因参与 CE，但不参与计数回归。代码中记录的 BCE 是指标，不是这套预测器的优化项。

PPO 选择计数为零、为一、至少为二的基因，分别得到 `-1 / 0 / +2` 奖励，停止奖励为零。不折扣时，最大化累计奖励等价于最小化最终选集的 `FP+2FN`。详细结构见 [docs/method.md](docs/method.md)。

## 安装

使用 Python 3.10 或以上版本，依赖包括 NumPy、SciPy 和 PyTorch 2.7 或以上版本，具体见 [pyproject.toml](pyproject.toml)。安装适合本机的 PyTorch，然后在项目根目录执行：

```bash
python -m pip install -e '.[test]'
```

以下指令用于后续运行，本次整理尚未执行。

## 数据

数据通过一个 JSON manifest 描述，路径可以相对于 manifest 所在目录填写。图像 bank 保存特征、细胞 ID、图像坐标和有序基因 ID。训练划分需要真实计数；仅推理的划分不需要计数。

格式模板见 [examples/data_manifest.json](examples/data_manifest.json)，完整说明见 [docs/data_format.md](docs/data_format.md)。

如果本机保留了原 Rank 工作区，可用迁移脚本生成独立数据目录：

```bash
python scripts/export_legacy_data.py --rank-root .. --output data
```

只有这个迁移步骤需要原工作区。正常训练和推理不依赖旧项目的 Python 模块。

## 训练

```bash
he-gene-rl train \
  --config configs/default.json \
  --data data/manifest.json \
  --output runs/main
```

默认依次执行预训练、适配和 PPO。`--stage pretrain|adapt|ppo|all` 可选择阶段；单独运行后续阶段时，需要同一运行目录中已有前序权重。模型在源 monitor 集上选优，测试表达不参与拟合、归一化或选模。

也可以先导入原实验已训练的权重：

```bash
python scripts/import_legacy_artifacts.py --rank-root .. --output runs/legacy
```

导入之后，把下方命令中的 `runs/main` 换成 `runs/legacy`。迁移与复现的说明见 [docs/reproduction.md](docs/reproduction.md)。

## 推理与评价

先生成预测。即使划分文件里存在计数，推理命令也不读取它：

```bash
he-gene-rl predict \
  --run runs/main --data data/manifest.json \
  --source Rep1 --target Rep2 --seed 42 --split test \
  --output predictions/Rep1_to_Rep2_seed42.npz --device cpu
```

`--method` 默认为最终方法 `ppo_sequence`，也可选 `adapted_sequence`、`teacher_sequence`、`teacher_one_shot`、`adapted_one_shot`，分别调用同一运行目录中对应的预测器与策略。`--batch-size` 默认是 512。

再使用真实表达单独评价：

```bash
he-gene-rl evaluate \
  --predictions predictions/Rep1_to_Rep2_seed42.npz \
  --data data/manifest.json --target Rep2 --split test \
  --output results/Rep1_to_Rep2_seed42.json
```

也可对一个运行目录中的配置模型统一评价：

```bash
he-gene-rl evaluate-run \
  --run runs/main --data data/manifest.json \
  --output results/main --device cuda
```

默认评价上述五种方法，可用 `--methods` 指定子集。程序先完成所有请求的预测，再读取真实表达计分。

如果已经有外部基线的最终选集，可直接比较：

```bash
he-gene-rl compare \
  --candidate predictions/Rep1_to_Rep2_seed42.npz \
  --reference predictions/external_baseline.npz \
  --data data/manifest.json --target Rep2 --split test \
  --output results/ppo_vs_external.json
```

两个预测文件都需要布尔型 `selected_mask` 或 `selected`，以及 `cell_ids`、完整且有序的 `gene_ids`。细胞顺序可以自动对齐，基因顺序必须预先与数据面板一致。格式详见 [docs/data_format.md](docs/data_format.md#prediction-and-external-baseline-archives)。这只是已经准备好的比较接口，必须在后续手工运行测试与评价后，才能得知独立项目的新结果。

项目包含本方法及其内部对照，不包含 SpatialEx 或原 STRank 基线的独立训练实现。原实验中的基线结果保留在历史结果文档中。

## 代码入口

| 文件 | 用途 |
|---|---|
| `src/he_gene_rl/model.py` | 预测器结构与监督损失 |
| `src/he_gene_rl/pretraining.py` | 随机真实历史预训练 |
| `src/he_gene_rl/adaptation.py` | 自预测历史适配 |
| `src/he_gene_rl/ppo.py` | PPO 策略、奖励与更新 |
| `src/he_gene_rl/core.py` | 逐步选择、预测计数回填与停止 |
| `src/he_gene_rl/data.py` | 数据接口、邻居和归一化 |
| `configs/default.json` | 正式训练配置 |
| `scripts/` | 原工作区数据与权重迁移工具 |
| `tests/` | 留待下一轮运行的测试 |

后续验证可从 `python -m pytest` 开始。本次没有运行测试，也没有重新训练或测量性能。

## 上传 GitHub

项目的 `.gitignore` 排除了数据、权重、预测文件和本地运行目录。小体积的历史汇总可以与源码一起上传。数据与权重可以另行发布，具体以其来源许可为准。

当前没有擅自指定软件许可证、作者、论文 DOI 或仓库地址。发布前应由项目所有者选择所需许可证。本次整理不会创建远程仓库或自动上传。
