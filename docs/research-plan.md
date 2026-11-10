# RePSS 从零研究计划

## 1. 这个项目研究什么

RePSS 全称是 **Vision-based Remote Physiological Signal Sensing**（基于视觉的远程生理信号感知）。它不是单一模型名，而是自 2020 年起连续举办的挑战赛与研讨会，近年固定作为 IJCAI 的 workshop。

核心现象是 **远程光电容积脉搏波（rPPG）**：心跳引起面部皮肤血流变化，摄像头能拍到极弱的颜色起伏。从这段视频里可以估计：

- 心率（HR）
- 心跳间期（IBI）与心率变异性（HRV）
- 呼吸率（RR）
- 血氧、血压等更难的指标

第五届（IJCAI 2026，不来梅）的主题是 **多模态融合**：可见光 RGB、近红外 NIR、毫米波雷达。RGB 在正常光照下信息丰富，但怕暗光、运动和肤色差异；NIR 和雷达补的是暗光、隐私和抗环境干扰。

当前时间点（2026-10）第五届 Kaggle 正式提交已于 2026-06-18 截止，workshop 已于 2026-08-17 举行。本仓库的目标是：**先把单模态 rPPG 跑通，再复现 RGB–NIR 融合设定，形成可发表的实验基线。** 不把已关闭的排行榜当作第一里程碑。

组织者长期包括浙江大学李晓白、中科院计算所韩琥与山世光、奥卢大学赵国英、INRIA Antitza Dantcheva 等。

## 2. 历届任务地图

| 届次 | 会议 | 任务 | 主要数据 |
|------|------|------|----------|
| 第 1 届 2020 | CVPR Workshop | 彩色人脸视频估计平均心率 | VIPL、OBF |
| 第 2 届 2021 | ICCV Workshop | Track 1：逐拍间期 IBI；Track 2：呼吸率 | 人脸视频；Track 2 有新数据集 |
| 第 3 届 2024 | IJCAI，济州，CEUR Vol-3750 | Track 1：无标签视频自监督心率；Track 2：视频估血压 | 预训练用人脸视频；微调 VIPL-V2；血压用 Vital Videos，测试 OBF |
| 第 4 届 2025 | IJCAI，广州，CEUR Vol-4095 | RGB + NIR 融合估计心率 | 训练：VIPL-HR 的 10 秒 RGB–NIR 片段（107 人）；测试：VIPL-HR-v2 与 OBF 各 100 人 |
| 第 5 届 2026 | IJCAI，不来梅 | Track 1：运动与弱光下 RGB–NIR 融合；Track 2：雷达心跳提取 | Kaggle：`the-5th-repss`、`the-5th-repss-t2` |

历届主指标基本都是 **RMSE**（第 3 届血压赛道是收缩压与舒张压 RMSE 的平均）。研究论文里同时报告 MAE、MAPE、Pearson 相关系数和信噪比 SNR。

第 4 届的经验可以直接当成研究假设：

- 训练在 VIPL-HR（会议室、年轻人），测试在 VIPL-HR-v2（年龄更广、场景更不受控）时误差明显变大。
- OBF 上肤色更分散。前几名的特点是浅、中、深肤色之间的 RMSE 差距更小。
- 头部转动幅度和速度都会拉高误差。
- 优胜方案的共同点是：先得到波形，再用频域主峰读心率；损失同时约束时域波形和频域分布；融合发生在特征层而不是简单平均两个心率数字。

第 5 届议程里反复出现的方法类型：谱域 RGB–NIR 融合、质量引导的双分支、弱光下的运动校准、雷达距离维注意力与对比学习。说明当前前沿是 **质量感知融合 + 跨模态对齐**，不是再堆一个更大的单模态网络。

## 3. 必读资料（按顺序）

### 3.1 先建立生理和信号直觉

1. Verkruysse 等，用普通摄像头从面部测 PPG 的起点工作。
2. 无监督颜色空间方法，按这个顺序读：Green、ICA、CHROM、POS。POS 是后面所有深度模型都要打过的传统基线。
3. 知道心率怎么从波形来：带通（常见约 0.7–4 Hz，对应大约 42–240 bpm）之后做功率谱，取主峰。

### 3.2 挑战赛总览（必读，短，信息密度高）

- 第 1 届：<https://openaccess.thecvf.com/content_CVPRW_2020/papers/w19/Li_The_1st_Challenge_on_Remote_Physiological_Signal_Sensing_RePSS_CVPRW_2020_paper.pdf>
- 第 2 届：<https://openaccess.thecvf.com/content/ICCV2021W/RePSS/papers/Li_The_2nd_Challenge_on_Remote_Physiological_Signal_Sensing_RePSS_ICCVW_2021_paper.pdf>
- 第 3 届论文集：<https://ceur-ws.org/Vol-3750/>
- 第 4 届论文集：<https://ceur-ws.org/Vol-4095/>
- 第 5 届主页：<https://repss-2026.github.io/5th-RePSS/>
- IJCAI 2026 workshop 列表：<https://2026.ijcai.org/accepted-workshops/>（WS03）

第 4 届总览论文要精读：数据协议、RMSE 定义、肤色与运动分组分析都在里面。

### 3.3 开源实现（第一行代码从这里来）

- [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)（UBICOMP Lab，NeurIPS Datasets & Benchmarks）。里面有无监督基线，也有 DeepPhys、PhysNet、TS-CAN、EfficientPhys、PhysFormer、PhysMamba、RhythmFormer 等监督模型，并支持 PURE、UBFC-rPPG、MMPD、SCAMPS、UBFC-Phys、BP4D+、iBVP。
- 论文：Liu 等，*rPPG-Toolbox: Deep Remote PPG Toolbox*，arXiv:2210.00716。

工具箱作者建议入门时用 **UBFC-rPPG、PURE、iBVP 或合成集 SCAMPS**，因为同步质量和规模更适合先跑通。VIPL-HR 帧率不稳定、场景更难，放在第二阶段。

### 3.4 深度模型，按出现顺序选读

| 作用 | 文献 | 读它的原因 |
|------|------|------------|
| 注意力 CNN | DeepPhys, Chen & McDuff, 2018 | 运动与外观解耦的经典结构 |
| 3D CNN 波形回归 | PhysNet, Yu 等, 2019 | 负 Pearson 损失成为事实标准 |
| 端侧时序模型 | TS-CAN / EfficientPhys, Liu 等 | 看清“轻量但仍可用”的下限 |
| 时序差分 Transformer | PhysFormer, Yu 等, 2022 | 第 4 届优胜方案的骨干族 |
| 状态空间模型 | RhythmMamba, AAAI 2025 | 在 PURE、UBFC-rPPG、VIPL-HR、MMPD 上同时报告了速度和精度 |
| 自监督 | Contrast-Phys 系列；第 3 届 PCA Lab、HFUT-VUT 方案 | 标签少时的路线，第 3 届 Track 1 的主体 |
| RGB–NIR | 第 4 届 CEUR Vol-4095 的 DiffRePSS、Temporal Difference Transformer、频域 Mamba | 多模态融合的直接样本 |

RhythmMamba 论文报告的数据集内结果可以当作“简单场景上沿”的参照：PURE MAE 0.23、UBFC-rPPG MAE 0.50、VIPL-HR MAE 4.30、MMPD MAE 3.16（单位 bpm）。工具箱的跨数据集表更重要：用一个数据集训练、另一个数据集测试时，MAE 经常到 4–15 bpm。**项目验收以跨数据集为准，不以数据集内刷分点为准。**

### 3.5 数据集与获取方式

| 数据集 | 规模与特点 | 阶段 |
|--------|------------|------|
| PURE | 10 人、6 种活动、约 1 分钟视频，同步 PPG | 第 1 周就能跑 |
| UBFC-rPPG | 42 人，游戏场景 | 第一个正式基准 |
| MMPD | 33 人、660 段手机视频；肤色、光照、说话、走路 | 鲁棒性测试 |
| SCAMPS | 2800 段合成视频，带心脏和呼吸波形 | 无真实数据时的预训练 |
| VIPL-HR | 107 人；2378 段 RGB + 752 段 NIR；多设备、多光照、头部运动；有 HR、SpO2、BVP | 多模态主训练集，需申请 |
| VIPL-HR-v2 | 年龄与场景更开放 | 泛化测试 |
| OBF | 肤色多样；挑战赛中面部打码并提供 OpenFace 关键点 | 公平性测试，注意许可 |
| 第 5 届 Kaggle | Track 1 RGB–NIR；Track 2 雷达 | 仅在许可允许时做离线复现，不假定还能改官方排名 |

VIPL-HR 论文：Niu 等，*VIPL-HR: A Multi-modal Database for Pulse Estimation from Less-constrained Face Video*。

## 4. 十二周学习与搭建路径

每周按 10–15 小时估算。没有 GPU 时，第 1–4 周仍可完成；第 5 周起需要一块至少 8 GB 显存的消费级显卡。

### 第 1–2 周：信号，不写神经网络

- 用一段公开 PPG，手算心率：带通、FFT、主峰、与标注 bpm 对比。
- 读 POS / CHROM，能讲清：为什么不能直接对 RGB 求平均，运动和光照如何进入颜色通道。
- 验收：给定 30 秒 PPG，估计心率与标注相差不超过 2 bpm。

### 第 3–4 周：传统 rPPG 基线

- 下载 PURE 或 UBFC-rPPG。
- 跑通人脸检测、稳定裁剪、POS 或 CHROM、频谱读心率。
- 按受试者划分，禁止同一人同时出现在训练和测试里（传统方法虽无训练，也按同一协议报测试集）。
- 验收：PURE 上 POS 的 MAE 进入文献常见范围（个位数 bpm）。若偏差很大，先查裁剪框和帧率，不换模型。

### 第 5–8 周：单模态深度模型

- 以 rPPG-Toolbox 的数据接口为参考，在本仓库实现自己的训练循环，避免只留一个无法复现的 notebook。
- 输入：定长人脸片段（常见 128–160 帧）。输出：与 PPG 对齐的波形，再解码为心率。
- 先在 PURE ↔ UBFC-rPPG 上做交叉测试，再加 MMPD。
- 验收：跨数据集 MAE 不差于工具箱中同量级模型的公开表；实验用一份配置文件可完整重跑。

### 第 9–12 周：RGB–NIR 融合

- 申请 VIPL-HR。在拿到数据前，用 RGB 与其灰度或合成噪声模态做融合接口的假数据测试，保证代码路径存在。
- 融合点放在特征层：两路编码器，质量或置信度加权，共享时序解码器，输出一条波形。
- 对比三条线：只用 RGB、只用 NIR、融合。报告整体 RMSE，以及光照、运动子集上的差值。
- 验收：在至少一个困难子集上，融合相对较强单模态的 RMSE 下降；同时给出失败案例（大转头、极暗、压缩严重）。

### 第 12 周之后的研究切口（先不做）

确认基线之后只选一个问题，避免同时开雷达、血压和自监督：

1. 弱光与运动下的模态质量估计，决定每一帧信 RGB 还是 NIR。
2. 肤色分组上的误差差距，把“差距变小”写成显式指标。
3. 雷达距离维信号与视频波形的对比对齐（对应第 5 届 Track 2，数据门槛更高）。

## 5. 仓库目录

研究代码按三种用途分开，避免论文、可复现基线和一次次实验混在同一棵源码树里。

```text
RePSS/
  papers/                 论文资料，只用于阅读
    catalog.md            文献索引
    repss/                历届挑战赛与 workshop 论文
    methods/              基础模型原文
    notes/                阅读笔记
  baselines/              基础模型代码，对照论文，实验不直接改
    classical/            无监督色度方法
    neural/               监督时空网络
  experiments/            基于基础模型的一系列实验程序
    _template/            新实验的复制骨架
    exp001_.../           一次实验一个目录
  docs/                   研究计划与架构
  data/                   本地数据，不进版本库
```

依赖方向只有一条：`experiments` 导入 `baselines`。论文笔记写在 `papers/notes/`，不写进模型源码。一次实验的配置、入口和输出都留在该实验目录内。技术栈为 Python 与 PyTorch。

## 6. 风险

- VIPL-HR、OBF 需要申请，周期可能是数周。前 8 周不依赖它们。
- 挑战赛测试视频的许可通常不允许再分发。复现实验以论文作者公开集为主。
- 帧率不稳定、标签和视频不同步，是 VIPL 系列的常见坑。波形损失前先做互相关对齐检查。
- 数据集内 MAE 接近 0 不代表方法有效。汇报必须包含跨数据集和困难子集。
- 面部视频是生物特征。仓库不提交人脸数据、不写爬取脚本、实验说明里保留数据集许可。
