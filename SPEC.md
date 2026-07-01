# Mapping Networks + RT-DETR v3.0 — 规格约束清单

> 每次修改代码前，必须对照此清单验证无相悖或歧义。
> 禁止为"跑通"而偏离规格。若实现与规格冲突，必须在此文档中记录并注明理由。

---

## 0. 系统物理边界与理论基础

- 最优参数 $\theta^*$ 驻留在本征维度 $d \ll P$ 的可微流形 $\mathcal{M}_\theta$ 上
- 全空间映射 $g: \mathbb{R}^d \to \mathbb{R}^P$ 必然导致梯度纠缠与表示坍塌
- 核心工程破局：**正交解耦控制流**，将隐空间约束为独立子空间

---

## 1. 隐空间流形正交切片

- [X]  `z = [z_lora, z_film, z_query]` — 严格正交切片，不得混合
- [X]  `z_k = z_k / ||z_k||_2` — **每条控制路径必须 L2 归一化到单位超球面**
- [X]  拓扑隔离：三条控制路径互不相交

- 实现: `src/latent/orthogonal_latent.py`

---

## 2. 局部低秩流形注入 (Partial LoRA Injection)

- 作用域: **Backbone Stage 4/5** + **AIFI 的 QKV 投影** + **Decoder Cross-Attention QKV**（扩展：Decoder 跨注意力也通过 LoRA 控制流形适配）
- 注入方程: `W_adaptive = W_pretrained + U(z_lora) @ V(z_lora)^T`
- 秩 `r ≪ min(m, n)`，生成量从 O(mn) 降至 O(r(m+n))
- 生成器: `U(z) = σ(γ_u · W_u_orth · z + β_u)`，其中 `W_u_orth` 正交初始化且 `requires_grad=False`
- γ、β 为可训练调制参数
- 实现: `src/generators/lora_generator.py`, `src/models/backbone.py:LoRAConv2d`

---

## 3. 多尺度特征融合的空间感知调制 (Spatial-aware FiLM)

- 作用域: **CCFF 跨尺度特征图融合节点**（不干涉卷积核权重生成）
- 生成器: `γ(z_film) = W_γ·z + b_γ`, `β(z_film) = W_β·z + b_β`
- 应用方式: 融合后逐通道仿射 `F'[c,:,:] = γ_c · F[c,:,:] + β_c`
- **隔离原则**: FiLM 仅在激活输出端实施流形控制，绝不干涉卷积核生成
- 实现: `src/generators/film_generator.py`, `src/models/encoder.py:CCFFFusionBlock`

---

## 4. 动态查询的残差调制 (Residual Modulated Query Selection)

- 作用域: **RT-DETR Decoder 的初始 Query**
- 必须**保留**不确定性最小化查询选择（Uncertainty-minimal Query Selection）
- 残差融合: `Q_final = LayerNorm(Q_init + σ(W_q·z_query + b_q))`
- 不得直接替换或覆盖 Q_init
- 实现: `src/models/decoder.py:RTDETRDecoder`

---

## 5. 基于迹估计的雅可比流形正则化

- 理论上需 $\|\nabla_z g(z)\|_F^2$，但**严禁**实例化完整雅可比图 → OOM
- 必须使用 **Hutchinson 迹估计器**（JVP, Jacobian-Vector Product）
- 三条控制流独立计算平滑性损失：
  `L_smooth ≈ Σ_{k∈{lora,film,query}} E_v[ ||∇_{z_k}(v^T g_k(z_k))||_2^2 ]`
- 稳定性损失: `L_stab = E[||f(z+ε) - f(z)||_2^2]`
- 对齐损失: `L_align = 1 - cos(z, W_mean)`
- 总损失: `L = L_task + λ1·L_stab + λ2·L_smooth + λ3·L_align`
- 实现: `src/losses/hutchinson_jvp.py`, `src/losses/smoothness_loss.py`, `src/losses/stability_loss.py`, `src/losses/alignment_loss.py`

---

## 6. 代码编写协议

- [X]  接口严格分离: `OrthogonalLatentSpace` 管理统一 z; `LoRAGenerator`, `FiLMGenerator`, `ResModGenerator` 各自独立
- [X]  梯度隔离检查: 训练前用 hook 断言控制路径梯度不交叉
- [X]  AMP 混合精度: 前向全链路 `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`
- [X]  JVP 计算不得保留完整导数图，使用 `torch.autograd.grad` 动态释放
- [X]  所有 `W_pretrained` 必须 `requires_grad = False`
- [X]  梯度检查点 (Gradient Checkpointing)

---

## 7. 三阶段训练流


| 阶段    | 名称                    | 冻结                | 可训练          | 损失                                                | LR           |
| ------- | ----------------------- | ------------------- | --------------- | --------------------------------------------------- | ------------ |
| Stage 1 | Manifold Discovery      | 所有权重 + 调制参数 | 仅 z            | L_task                                              | 正常         |
| Stage 2 | Joint Modulation        | 所有权重            | z + 全部 γ, β | L_task                                              | 正常         |
| Stage 3 | Manifold Solidification | 所有权重            | z + 全部 γ, β | L_task + λ1·L_stab + λ2·L_smooth + λ3·L_align | **指数衰减** |

- [X]  Stage 1: 冻结目标网络权重，仅解冻 z
- [X]  Stage 2: 联合训练 z + 调制参数
- [X]  Stage 3: 引入 L_stab + L_smooth，LR 指数级衰减，AMP + GC

---

## 8. 多模态与 3D 扩展接口

- 融合系数生成: `α = g_fuse(z)` (RGB-D)
- 特征对齐: `F_fused = α·F_rgb + (1-α)·F_depth`
- 骨干解耦: 2D (ResNet) / 3D (PointNet++) 通过抽象基类热插拔
- 实现: `src/multimodal/fusion.py`

---

## 实现偏离记录

> 此处记录因工程限制而不得不偏离规格之处，每项须注明理由。


| 日期       | 偏离项                                                                  | 理由                                                                                |
| ---------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 2026-07-02 | L_align 拆分为 Σ(1-cos(z_k, W_mean_k)) 三条路径独立计算 | 正交切片后三个子空间维度不同（256/128/128），无法 stack 为一个 W_mean。数学语义等价 |
| 2026-07-02 | FiLM 正交矩阵参数名不含 "orth" → 对齐损失收集不到 | 已修复：W_gamma→W_gamma_orth, W_beta→W_beta_orth |
| 2026-07-02 | L_smooth 使用 `torch.func.jvp`（前向模式 AD）替代 `autograd.grad(create_graph=True)` | 前向 AD 天然可微，避免二阶图溢出且保留梯度链。fallback 路径（无 torch.func 时）用 detach leaf，梯度不贡献 |
| 2026-07-02 | L_align 手动 L2 归一化替代 `F.normalize` | `W.mean(dim=0)` 对大正交矩阵趋近零向量，`F.normalize` 在零范数附近数值不稳定。手动除范数加 1e-8 保护，公式数学等价 |

---

## 修改前自检清单

- [ ]  是否违反 z 的正交切片隔离？
- [ ]  是否违反 L2 超球约束？
- [ ]  是否违反 W_orth 正交+冻结？
- [ ]  是否违反 FiLM 不干涉卷积权重的隔离原则？
- [ ]  是否违反 ResMod 保留 Q_init 的残差原则？
- [ ]  是否违反 JVP 迹估计（不得使用完整雅可比）？
- [ ]  是否违反 Stage 3 指数衰减 LR？
- [ ]  是否违反 W_pretrained requires_grad=False？
- [ ]  新增的 `init_gain=0.1` 是否改变了数学形式？(否 — 仅缩放初始值，前向方程不变)
