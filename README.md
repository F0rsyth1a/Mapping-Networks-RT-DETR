# Mapping Networks + RT-DETR (v3.0)

论文级复现工程：基于权重流形假设（Weight-Manifold Hypothesis）的端到端目标检测系统，通过正交解耦控制流（Orthogonal Decoupled Control）将 RT-DETR 的参数生成约束在低维光滑流形上。

## 核心架构

```
                    z = [z_lora, z_film, z_query]  (L2 单位超球面约束)
                         │ 正交切片
          ┌──────────────┼──────────────┐
       z_lora           z_film         z_query
          │                │              │
    [LoRAGenerator]  [FiLMGenerator] [ResModGenerator]
          │                │              │
    W = W₀ + U·Vᵀ    γ,β per-channel    ΔQ residual
          │                │              │
    Backbone+AIFI       CCFF融合节点    Decoder Query
   (Stage 4/5 QKV)    (F' = γ·F + β)  (Q = LN(Q₀+ΔQ))
          └──────────────┼──────────────┘
                         │
                    RT-DETR (frozen weights)
                         │
                   pred_logits, pred_boxes
```

## 三阶段训练

| 阶段 | 名称 | 可训练参数 | 损失函数 | LR |
|------|------|-----------|---------|-----|
| Stage 1 | Manifold Discovery | 仅 z | L_task | 固定 |
| Stage 2 | Joint Modulation | z + γ, β | L_task | 固定 |
| Stage 3 | Manifold Solidification | z + γ, β | L_task + λ₁L_stab + λ₂L_smooth + λ₃L_align | 指数衰减 |

- **L_task**: 分类交叉熵 + L1 + GIoU（匈牙利匹配）
- **L_stab**: 隐空间扰动稳定性损失
- **L_smooth**: 雅可比 Frobenius 范数（Hutchinson 迹估计，VJP）
- **L_align**: 隐向量与映射矩阵方向对齐损失

## 快速开始

### 安装

```bash
pip install torch torchvision numpy pyyaml scipy
```

### 架构验证（合成数据，无需 COCO）

```bash
python main.py --config configs/default.yaml --dry-run
```

### COCO 训练

```bash
# 下载 COCO 2017
mkdir -p /path/to/coco && cd /path/to/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip && unzip annotations_trainval2017.zip

# 单卡训练
python main.py --config configs/default.yaml --coco-path /path/to/coco

# 多卡 DDP
torchrun --nproc_per_node=4 main.py --config configs/default.yaml --coco-path /path/to/coco

# 断点恢复
python main.py --config configs/default.yaml --coco-path /path/to/coco --resume output/ckpt_stage2_epoch020.pt
```

## 项目结构

```
├── configs/default.yaml       # 全局超参数
├── src/
│   ├── latent/                # 正交隐空间 (z 切片 + L2 约束)
│   ├── generators/            # LoRA / FiLM / ResMod 生成器
│   ├── models/                # RT-DETR (Backbone + Encoder + Decoder)
│   ├── losses/                # 任务损失 + 正则化损失
│   ├── training/              # 三阶段训练循环
│   ├── multimodal/            # RGB-D 多模态扩展接口
│   └── utils/                 # 梯度检查 + 显存监控
├── tests/                     # 单元测试
├── SPEC.md                    # v3.0 规格约束清单
└── main.py                    # 入口
```

## 设计原则

- 所有目标网络权重冻结 (`requires_grad=False`)
- 正交映射矩阵 `W_orth` 正交初始化 + 冻结
- JVP 不使用完整雅可比（Hutchinson 迹估计）
- 梯度隔离检查确保三条控制路径互不交叉
- AMP bfloat16 + Gradient Checkpointing 控制显存

## 已知限制

- `torch.func.vjp` 对 nn.Module 在大输出维度（LoRA ~7.8M 元素）下数值不稳定，L_smooth 对 LoRA/Query 分支暂不贡献梯度
- 匈牙利匹配使用 scipy.optimize.linear_sum_assignment

## 参考文献

1. Weight-Manifold Hypothesis: 最优参数 θ* 存在于本征维度 d ≪ P 的可微流形上
2. RT-DETR: DETRs Beat YOLOs on Real-time Object Detection (CVPR 2024)
3. Hutchinson Trace Estimator: A stochastic estimator of the trace of an implicit matrix
