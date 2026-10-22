# SRCF-Repro

**Reproduction of SR-CF for cold-start recommendation** · **面向冷启动推荐的 SR-CF 算法复现**

![CI](https://github.com/ZJY-HSBL/SRCF-Repro/actions/workflows/ci-release.yml/badge.svg) [Releases](https://github.com/ZJY-HSBL/SRCF-Repro/releases) · [Replication protocol / 复现规范](docs/REPLICATION.md)

## 研究背景 · Research overview

本项目复现乔雨（2018）硕士论文《面向冷启动问题的推荐算法研究》第四章的 **SR-CF**（融合用户相似度与评分信息的协同过滤算法）。它结合用户基本属性 `D`、种子项目评分 `S` 和历史评分 `R`，以增强新用户的邻居发现能力。**这是论文复现工程，不声称算法原创。**

This repository implements the chapter-4 **Similarity and Rating-based Collaborative Filtering (SR-CF)** approach in Qiao Yu's 2018 master's thesis. It combines demographic similarity (`D`), seed-item similarity (`S`), and historical-rating similarity (`R`) for sparse / new-user recommendation. **The SR-CF method originates in the cited thesis; this repo does not claim its invention.**

![SR-CF workflow, source chapter 4](artifacts/figures/srcf_flow.svg)

## 核心公式 · Method (LaTeX)

本节使用 **GitHub 官方支持的 LaTeX / MathJax** 源码，公式不使用截图、SVG 图片或代码块。/ Equations are written as native GitHub Markdown mathematics (not images or code fences). GitHub Web renders MathJax; rendering in some GitHub mobile apps can differ.

**论文 SR-CF 融合相似度 · Thesis Eq. (4.8), printed p. 34:**

$$
\operatorname{Sim}(u,v)=\alpha D(u,v)+\beta S(u,v)+\gamma R(u,v)
$$

$$
\alpha+\beta+\gamma=1,\qquad (\alpha,\beta,\gamma)=(0.2,0.3,0.5)
$$

其中 / where $D(u,v)$ is demographic similarity, $S(u,v)$ is similarity on seed-item ratings and $R(u,v)$ is similarity on observed historical ratings. **The thesis's Eq. (4.4) is an *unnormalized sum***; `--prediction paper` preserves it without normalization or clipping.

**本项目的归一化预测器 · Independently corrected normalized predictor (NOT the thesis Eq. 4.4):**

$$
\widehat r_{ui}=\frac{\sum_{v\in N_k^{+}(u,i)}\operatorname{Sim}(u,v)\,r_{vi}}{\sum_{v\in N_k^{+}(u,i)}\operatorname{Sim}(u,v)}
$$

这里 $N_k^{+}(u,i)$ 表示对电影 $i$ 评分且与目标用户具有正相似度的选中邻居。如果分母为零，则使用**仅由训练集**计算的项目均值或全局均值回退。原论文公式与修正版务必分开评价。

**本项目 v0.5.0 研究假设（非原论文算法）· Support-confidence hypothesis:**

$$
q(n;\lambda)=\begin{cases}
0,&n<2\text{ or similarity is undefined},\\
\dfrac{n}{n+\lambda},&n\ge2\text{ and similarity is defined}.
\end{cases}
$$

$$
\operatorname{Sim}_{\lambda}(u,v)=
\frac{\alpha D(u,v)+\beta q(n_S;\lambda)S(u,v)+\gamma q(n_R;\lambda)R(u,v)}
{\alpha+\beta q(n_S;\lambda)+\gamma q(n_R;\lambda)}
$$

式中 $n_S$ 与 $n_R$ 分别是共同种子评分与共同历史评分数量。$\lambda$ 只从验证用户中选择，再锁定参数测试未见用户。公式实现见 [`confidence.py`](src/srcf_repro/confidence.py)，协议见 [`confirmatory.py`](src/srcf_repro/confirmatory.py)。此方案是有待验证的机制，不能提前声称带来提升。

## 数据与实测 · Data & measured results

| Source / 来源 | Purpose / 用途 | Status / 状态 |
|---|---|---|
| Thesis Table 4.2 | Published coverage comparison / 原论文覆盖率比较 | **Reported only / 论文报告** |
| Synthetic: 42 users, 60 items, 1,092 ratings | Functionality smoke benchmark / 功能验证 | **Actually executed / 已运行** |
| MovieLens-100K | Genuine replication evaluation / 真实数据复现 | **Real experiment complete / 已完成真实实验** |

![Coverage transcribed from the thesis, NOT reproduced](artifacts/figures/paper_coverage.svg)

**原论文报告**：邻居数 `k=100` 时，SR-CF、DR-CF 和 UB-CF 的覆盖率分别是 **27.3%、23.8%、21.3%**；这张图不是本项目运行所得。 / **Paper-reported only**: coverage at `k=100` is 27.3%, 23.8%, and 21.3%, respectively; not independently reproduced here.

![Measured synthetic benchmark; not MovieLens](artifacts/demo/demo_mae.svg)

![Measured synthetic cold-start experiment; fixed holdouts](artifacts/demo/cold_start.svg)

合成冷启动实验采用完全留出的目标用户，固定每个目标用户的 5 条测试评分，然后分别开放 0/1/3/5/10 条历史行为。0 条时，SR-CF 与 DR-CF 的 MAE 都是 **0.7292**；3 条时 DR-CF 为 **0.6583**，SR-CF 为 **0.8805**。这意味着固定权重可能在极稀疏条件下表现不稳定，必须用真实数据验证。 / A user-disjoint synthetic cold-start study holds out five ratings per user and exposes 0/1/3/5/10 history ratings. At zero observations, SR-CF and DR-CF have identical MAE **0.7292**; at three, DR-CF achieves **0.6583** versus SR-CF **0.8805**. This motivates, but does not establish, an adaptive-weight hypothesis.

**真实运行的合成数据**，种子 2026，32 个测试用户，160 条测试评分，各模型均使用归一化预测器 / **Measured on synthetic data** with seed 2026, 32 evaluated users, 160 test ratings, normalized predictor:

| k | UB-CF MAE ↓ | DR-CF MAE ↓ | SR-CF MAE ↓ |
|---:|---:|---:|---:|
| 5 | 0.7525 | 0.8256 | **0.7505** |
| 10 | 0.6748 | **0.6426** | 0.6500 |
| 20 | 0.7068 | **0.7066** | 0.7073 |

在合成实验中，SR-CF 并非全面领先。例如 k=10 时 DR-CF 更好，且模型之间的差异较小。后续重点是 **在独立真实数据上检验权重与种子集的有效性**，不能根据上述结果宣称论文性能已被复现。 / SR-CF does not dominate in this synthetic experiment (e.g. DR-CF has lower MAE at k=10). No published-paper performance claim is inferred from this toy result.

Full raw rows: [`benchmark.csv`](artifacts/demo/benchmark.csv) · [`benchmark.json`](artifacts/demo/benchmark.json). Coverage data transcribed verbatim: [`paper_coverage.csv`](artifacts/paper_coverage.csv).

## MovieLens-100K 真正实测 · Real-data evaluation

真实数据已在 GitHub Actions 环境中运行，含常规留出集和固定测试项目的 0/1/3/5/10 条历史行为冷启动实验。详见 [真实数据实验报告 / real-data results](docs/REAL_RESULTS.md)。**Independent real-data results, not exact numerical replication of the thesis.**

![Official MovieLens-100K benchmark](artifacts/real/ml100k/demo_mae.svg)

![Official MovieLens-100K cold-start benchmark](artifacts/real/cold/cold_start.svg)

## 多随机批次冷启动 · Multi-batch cold-start evaluation

<!-- STUDY_RESULTS_START -->
### 多批次冷启动实验 · Repeated cold-start study

**SYNTHETIC**, 16 independent target users, fixed holdouts, no test-set tuning. 结果全部保留，既包括支持假设也包括反例。

| Visible | ub | dr | sr | sr_no_seed | sr_no_history | sr_schedule |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.1087 | 0.6762 | 0.6762 | 0.6762 | 0.6762 | 0.6762 |
| 1 | 1.1099 | 0.6648 | 0.6648 | 0.6648 | 0.6648 | 0.6648 |
| 3 | 1.0505 | 0.6699 | 0.7281 | 0.7285 | 0.6710 | 0.6789 |
| 5 | 1.0265 | 0.6764 | 0.8417 | 0.8398 | 0.6767 | 0.6688 |
| 10 | 0.7818 | 0.6570 | 0.7759 | 0.7482 | 0.7411 | 0.7759 |

![Measured SYNTHETIC cold-start comparisons](artifacts/experimental/synthetic-repeated/cold_start_ablations.svg)

[完整实验报告、配对 Bootstrap 置信区间 / full study and CIs](docs/REPEATED_RESULTS.md)

<!-- STUDY_RESULTS_END -->

## 种子评分信号诊断 · Seed-signal diagnostic

<!-- SEED_STUDY_START -->
### MovieLens-100K 新目标用户种子集实验 · Seed-set follow-up

**MOVIELENS-100K**, 48 new target users (earlier 96 excluded), fixed user holdouts and seed-set-size ablations.

| Known ratings | UB-CF | DR-CF | SR(12) | SR(40) | SR(100) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8002 | 0.9330 | 0.9330 | 0.9330 | 0.9330 |
| 1 | 0.8002 | 0.9252 | 0.9252 | 0.9252 | 0.9252 |
| 3 | 0.8653 | 0.9251 | 0.9290 | 0.9287 | 0.9225 |
| 5 | 0.8757 | 0.9250 | 0.8333 | 0.8401 | 0.8345 |
| 10 | 0.9155 | 0.9249 | 0.8513 | 0.8700 | 0.8853 |

![Seed-set sensitivity, measured](artifacts/real/seed-diagnostics/seed_count_mae.svg)

![Seed similarity support, measured](artifacts/real/seed-diagnostics/seed_signal_availability.svg)

[实验方案、配对区间与完整 CSV / protocol, CIs and data](docs/SEED_DIAGNOSTICS.md)

<!-- SEED_STUDY_END -->

### 结果解读 · Interpretation

**新目标用户真实实验 / Newly sampled MovieLens-100K users:** 在 3 条历史评分且只使用 12 部种子电影时，**0/48** 位目标用户具有可用的非零种子评分相似度；在 10 条历史评分时比例为 **14.6%**。扩展为 100 部种子电影能把 10 条交互下的信号可用率提升至 **81.2%**，但对应 SR-CF 的 MAE 从 0.8513 增加至 0.8853，**更多种子不必然提高准确率**。

**重要负面结果 / Important counterexample:** 对零历史评分的新用户，UB-CF 通过训练集项目均值回退实现 MAE **0.8002**，优于 DR-CF/SR-CF 的 **0.9330**。这些实验属于单数据集探索性验证，不声称具有统计确认的优越性。下一版将检验基于共评分数量的相似度置信收缩，并使用严格隔离的验证用户选择超参数。

## 可信度自适应实验 · Validation-frozen confidence study (v0.5.0)

新方法通过共评分样本量收缩不稳定的评分相似度，在互不重叠的验证用户中选择 $\lambda\in\{2,10,30\}$，随后在全新测试用户中与 UB-CF、DR-CF、原 SR-CF 对比。/ Shrink unreliable Pearson components; tune only on validation users, then compare on untouched test users.

<!-- CONFIDENCE_STUDY_START -->
实验待官方 MovieLens-100K CI 运行后填充。/ Measured official-data report is generated only after CI finishes.
<!-- CONFIDENCE_STUDY_END -->

## 安装与复现 · Installation & reproduction

Python **3.10+**:

```bash
python -m pip install -e ".[plot,test]"
pytest
# Fully local, deterministic synthetic smoke benchmark / 本地合成数据
python -m srcf_repro.experiment --out artifacts/local/demo --seed 2026
python -m srcf_repro.experiment --cold --max-users 15 --out artifacts/local/cold --seed 2026
```

真实数据 **MovieLens-100K**：从 [GroupLens 官方下载页](https://grouplens.org/datasets/movielens/100k/) 获取并解压 `ml-100k`，确保包含 `u.data` 和 `u.user`。然后运行 / For the actual MovieLens-100K benchmark, obtain the official archive and run:

```bash
python -m srcf_repro.experiment --data-dir ./ml-100k --max-users 200 --out artifacts/local/ml100k
python -m srcf_repro.experiment --data-dir ./ml-100k --max-users 200 --prediction paper --out artifacts/local/ml100k-paper
```

**论文存在数据集标识冲突**：第四章写的是 `ml-1m`，但其用户数 **943**、电影数 **1682**、评分数 **100000** 对应 **MovieLens-100K**。具体处理及其他未公开实验细节见 [复现规范 / replication notes](docs/REPLICATION.md)。/ The thesis calls its data “ml-1m” while providing the MovieLens-100K counts. We do **not** silently equate the two datasets.

## 工程与路线图 · Engineering & roadmap

- `src/srcf_repro/model.py` — three modes, mixed similarities, literal/normalized score prediction.
- `src/srcf_repro/data.py` — train/test split, popularity-based seed selection, official `u.data` / `u.user` parser.
- `src/srcf_repro/experiment.py` — deterministic MAE/RMSE/coverage benchmarks and SVG export.
- `src/srcf_repro/studies.py` — pre-registered ablations, user-disjoint repeated cold-start studies, subject-level paired bootstrap; automatically generates data/report/SVG.
- `tests/` — mathematical edge cases, reproducibility, train/test isolation, and smoke results.
- `.github/workflows/ci-release.yml` — quality gate and release-on-version-commit workflow.
- `src/srcf_repro/seed_diagnostics.py` — independent exploratory target cohort, seed-count ablations, paired estimates and support diagnostics.

Next / 下一步：本版本执行严格用户隔离的多批次重复实验、固定权重消融及探索性分段权重比较。后续将在独立验证集上调参，并在全新数据集上确认，而不能凭单轮实验宣称创新有效。 / Multiple user-disjoint batches and paired bootstrap intervals precede any validation-only tuning and external confirmation.

## 引用 · Citation

乔雨. **面向冷启动问题的推荐算法研究**. 南京邮电大学工学硕士学位论文, 2018 年 2 月, 第四章, pp. 29–37. / Qiao Yu, *Research on Recommendation Algorithms Against Cold-Start Problem*, Nanjing University of Posts and Telecommunications, Feb. 2018, Chapter 4.

License for **this implementation**: MIT. Original thesis copyright remains with its author.
