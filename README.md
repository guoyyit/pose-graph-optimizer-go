# Pose Graph Optimizer (Go)

一个零第三方依赖的 **2D（SE(2)）本地位姿图优化器**，支持带不确定性的相对位姿
约束与绝对先验、增量维护、回环闭合融合、求解诊断、并发版本化访问以及崩溃安全的
快照持久化。

```go
import "github.com/guoyyit/pose-graph-optimizer-go/posegraph"
```

---

## 1. 能力概览

| 需求 | 实现 |
| --- | --- |
| 节点 + 带协方差/信息的相对约束，估计全局一致位姿并输出逐节点残差与不确定性 | Gauss-Newton/LM 求解，输出 `Result.Nodes`（位姿+边际协方差）、`Result.Residuals`、`Result.PerNode` |
| 增量增删、重复/乱序/冲突约束按确定规则合并，结果可复现 | 批内规范排序 + 对称信息加权融合；与提交顺序无关 |
| 回环约束加入、识别明显不一致项并拒绝/降权，拒绝后其余图仍可用 | chi² 门控拒绝 + DCS 鲁棒核降权；失败保留上一良好结果 |
| 求解失败 / 病态矩阵 / 收敛停滞诊断，绝不把高误差解标记成功 | `Diagnostics.Status` + 条件数 + 高残差发布门控 |
| 并发提交与查询，读者只看到一致版本 | `sync.RWMutex` 串行化提交，`Query()` 返回不可变深拷贝，`WaitForVersion` 版本等待 |
| 状态与结果持久化，重启从最近一致快照恢复并继续增量 | 原子写（tmp+fsync+rename）JSON 快照 |

---

## 2. 快速开始

```go
cfg := posegraph.DefaultConfig()          // 也可设置 cfg.SnapshotPath 开启持久化
store, err := posegraph.NewStore(cfg)

// 1) 提交节点（A 作为固定锚点）
store.Submit(posegraph.Change{Kind: posegraph.ChangeNode,
    Node: posegraph.Node{ID: "A", Initial: posegraph.Pose2{}, Fixed: true}})
store.Submit(posegraph.Change{Kind: posegraph.ChangeNode,
    Node: posegraph.Node{ID: "B", Initial: posegraph.Pose2{X: 1.1, Y: 0.1}}})

// 2) 提交相对约束：pose(B) = pose(A) ⊕ measurement，附带协方差
store.Submit(posegraph.Change{Kind: posegraph.ChangeConstraint,
    Constraint: posegraph.Constraint{
        ID: "ab", A: "A", B: "B",
        Measurement: posegraph.Pose2{X: 1},
        Covariance:  posegraph.Mat3{0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01},
    }})

// 3) 查询最近一次一致结果（AutoOptimize 默认开启，每次提交后自动优化）
res := store.Query()
for id, m := range res.Nodes {
    fmt.Println(id, m.Pose, m.Covariance)
}
```

批量原子提交使用 `store.Commit([]Change{...})`；关闭自动优化后可手动调用
`store.Optimize()`。

---

## 3. 约束模型（Constraint Model）

### 3.1 位姿与复合

位姿 `Pose2{X, Y, Theta}`，角度始终归一化到 `(-π, π]`。

- 复合：`a ⊕ b = T_a · T_b`
- 逆：`a⁻¹`
- 相对：`between(a, b) = a⁻¹ ⊕ b`

### 3.2 两种约束

- **相对约束 `Relative`**：`pose(B) = pose(A) ⊕ z`，
  误差 `e = z⁻¹ ⊕ (a⁻¹ ⊕ b) ∈ R³`（平移 + 包裹后的角度）。
- **绝对先验 `Prior`**：锚定单节点到全局系，`e = z⁻¹ ⊕ pose(B)`。
  先验用于在不固定任何节点时消除整体自由度（gauge freedom）。

### 3.3 不确定性

每条约束携带 3×3 `Covariance`（按 `x, y, θ`）。求解前统一转换为信息矩阵
`Ω = (sym(Σ) + floor)⁻¹`：

- 自动对称化，容差内的非对称输入被纠正；
- 对角加 `CovarianceFloor`，保证**零/退化协方差不会变成无限信息**；
- 若显式提供非零 `Information`，则直接使用它。

白化残差为 `r = Ω^{1/2} e`，标量代价 `χ² = eᵀ Ω e`。

### 3.4 优化问题

在世界系扰动参数化下线性化，组装正规方程

```
H Δx = -b,   H = Σ Jᵀ (w·Ω) J,  b = Σ Jᵀ (w·Ω) e
```

采用 **Levenberg–Marquart**（对自由块对角阻尼）求解，`w` 为回环鲁棒权重。
固定节点与（无先验时）按 **ID 字典序确定性选出的锚点**不参与求解，从而固定
规范自由度。边际协方差取收敛信息矩阵自由块的逆。

---

## 4. 增量维护与确定性合并规则

| 情形 | 规则 | `ChangeOutcome` |
| --- | --- | --- |
| 相同节点重复提交且初值/固定标记一致 | 忽略重复 | `repeated_ignored` |
| 相同节点但初值冲突 | **首次提交获胜**，冲突在报告中显式给出 | `repeated_ignored` |
| 相同约束 ID、端点/测量/协方差一致 | 忽略重复 | `repeated_ignored` |
| 相同约束 ID 但测量/协方差冲突 | **对称信息加权融合**并对协方差做冲突膨胀 | `fused_conflict` |
| 约束引用了尚不存在的节点 | 进入 `pending` 缓冲，节点补齐后按序重放 | `pending_dangling` |
| 回环候选 | 见第 5 节 | `loop_accepted / loop_downweighted / rejected` |
| 结构性非法输入 | 拒绝并给出原因 | `invalid` |

**冲突融合（与顺序无关）**：

```
Ω₁ = Σ₁⁻¹,  Ω₂ = Σ₂⁻¹
Σ_fused = (Ω₁ + Ω₂)⁻¹ · ConflictInflation
z_fused = 以 tr(Ω) 为标量权重的加权平均（角度用单位向量融合，正确处理 ±π）
```

由于求和与标量加权都是可交换的，`fuse(a,b) == fuse(b,a)`。

**批内顺序无关**：`Commit` 先按（节点 ID）、再按约束规范键排序后处理，
迭代一律按 ID 字典序。因此**固定输入集合与配置下，结果不因提交顺序不同而变化**
（见 `TestOrderIndependence`、`TestIncrementalVsBatch`）。

---

## 5. 回环检测融合与稳健策略

回环约束（`Constraint.Loop = true`）在加入前，先用**不含该候选自身**的当前
最优估计评估，候选不能为自己投票：

1. **硬拒绝**：白化 `χ²/DOF > LoopChi2Gate`（默认 7.815，约 χ²(3) 95% 分位），
   或原始偏差超过硬性尺度（50 m / π），记入 `rejected`，**绝不进入图**。
2. **降权接受**：通过门控但白化范数超过鲁棒核 `RobustKernel` 时，用 **DCS**
   因子降权后加入：
   ```
   w = 1                         (norm ≤ δ)
   w = 2δ / (norm + δ/norm)      (norm > δ)
   ```
   轻微错误的回环因此无法扭曲整图。
3. **全权接受**：一致候选按权重 1 加入。

优化中每轮迭代做一次 IRLS 重加权。**被拒绝的回环不影响其余图**：
`Query()` 仍返回拒绝前最近一次一致结果（见 `TestLoopRejected`）。

### 求解质量门控（不会把高误差解标记成功）

即使最小二乘迭代“收敛”，只要存在白化范数 > 5（约 5σ）的严重违约约束，
该结果**不会**作为成功版本发布，状态置为 `stagnated` 并附带说明；
上一良好结果继续可查。`StatusMaxIterations` 表示仍在下降但触及迭代上限，
同样区别于真正收敛。

---

## 6. 诊断（Diagnostics）

`Result.Diag` / `Optimize()` 返回：

| 字段 | 含义 |
| --- | --- |
| `Status` | `converged / max_iterations / stagnated / ill_conditioned / failed / not_run` |
| `Iterations` | 实际迭代次数 |
| `FinalChi2` | 收敛后总白化误差 |
| `MaxWhitened` | 单条约束最大白化残差范数 |
| `RMSE` | 按 3 分量归一的均方根误差 |
| `ConditionEst` | 正规方程条件数估计（越大越病态；超 `ConditionLimit` 判病态） |
| `StepNorm / CostDelta` | 最后一步步长与代价变化 |
| `Warnings` | 失败/停滞/未发布等人读原因 |

结构性病态（固定后存在无约束连通分量、矩阵无法 Cholesky 分解、非有限步长、
阻尼超过 `MaxDamping` 仍无法下降）分别返回 `ill_conditioned` 或 `failed`。

---

## 7. 并发与版本可见性

- 所有变更经同一把写锁串行化；一次 `Commit` 对读者是原子的。
- 求解在**私有副本**上进行，读者绝看不到中间迭代状态；只有成功结果才整体发布，
  `Result.Version` 单调递增。
- `Query()` 返回**深拷贝**，调用方修改不影响存储内部（`TestConcurrentCommitQuery`
  在 `-race` 下验证）。
- `WaitForVersion(v)` 阻塞直到至少版本 `v` 可见，`Close` 时以 `false` 唤醒。

---

## 8. 持久化与重启恢复

设置 `Config.SnapshotPath` 后：

- 启动时从该路径加载最近一致快照（节点、已接受约束、`pending`、`rejected`、
  版本号、最近结果、配置）；
- 每次接受变更 / 拒绝或挂起 / 关闭时，以 **写临时文件 → fsync → rename** 原子落盘，
  崩溃不会留下半写文件；
- 重启后在恢复的图上继续增量优化，**已接受约束不丢失**
  （见 `TestRestartRecovery`、`TestRejectedLoopPersisted`）。

---

## 9. 配置项（Config）

`DefaultConfig()` 给出保守默认值；未设置/非法字段在建库和加载快照时自动归一化。

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `MaxIterations` | 50 | GN/LM 最大迭代 |
| `ConvergeChi2` | 1e-7 | 相对/绝对代价下降收敛阈值 |
| `ConvergeStep` | 1e-8 | 步长收敛阈值 |
| `StagnationRounds` | 5 | 连续无下降轮数，超过判停滞 |
| `InitialDamping` | 1e-3 | LM 初始阻尼 |
| `DampingUp / DampingDown` | 10 / 0.5 | 失败/成功步的阻尼缩放 |
| `MaxDamping` | 1e12 | 阻尼上限，超过判失败 |
| `ConditionLimit` | 1e12 | 条件数上限，超过判病态 |
| `CovarianceFloor` | (1e-6,1e-6,1e-4) | 测量协方差对角地板 |
| `LoopChi2Gate` | 7.815 | 回环 χ²/DOF 拒绝门限（χ²(3) 95%） |
| `RobustKernel` | 1.0 | DCS 鲁棒核 δ（白化范数）；0 关闭降权 |
| `ConflictInflation` | 4.0 | 冲突融合后协方差膨胀倍数 |
| `AutoOptimize` | true | 每次变更后自动优化 |
| `SnapshotPath` | "" | 非空则启用原子快照持久化 |

---

## 10. 测试

```bash
go test ./...            # 全部用例
go test ./... -race      # 含数据竞争检测
```

覆盖场景：

- **增量乱序**：批内逆序/交错提交与批量提交结果一致；悬空约束缓冲重放。
- **冲突回环**：明显错误回环拒绝、轻微错误降权、一致回环接受、融合对称性。
- **病态退化**：孤立分量判病态、零协方差地板、先验代替锚点、信息增加方差减小。
- **并发更新**：8 写 4 读在 `-race` 下只观察一致快照，返回值深拷贝隔离。
- **重启恢复**：快照落盘后重开，节点/约束/结果不丢并能继续增量。

## 11. 范围与限制

- 面向 **2D SE(2)**；3D 需要替换位姿、复合与雅可比实现。
- 稠密矩阵实现，规模在数百到数千节点量级实用；超大图建议改块稀疏求解。
- 不内置回环候选的几何外观特征检测；候选由调用方给出，本库负责门控与融合。
