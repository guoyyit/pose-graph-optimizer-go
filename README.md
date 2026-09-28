# 本地位姿图优化与回环检测融合（Go）

一个零依赖的纯 Go 实现，用于 **SE(2) 位姿图优化** 与 **回环闭合检测**：

- 以节点（全局位姿）和相对位姿约束（带协方差/信息矩阵）描述图；
- Levenberg–Marquardt 求解全局一致位姿，输出**每节点位姿、边际协方差**与**每约束残差**；
- 增量增删、乱序/重复/冲突约束按**确定性规则**合并，固定输入下结果与提交顺序无关、可复现；
- 回环约束经 **χ² 一致性门控**：拒绝（hard reject）或降权（soft downweight + Huber 核），拒绝后其余图仍可用；
- 求解失败、病态矩阵、收敛停滞均有明确**诊断**，绝不把高残差解标记为成功；
- **并发安全**：读写加锁，读取方永远只看到某个已发布的完整不可变版本；
- **持久化**：原子快照 + 预写日志（WAL），重启后从最近一致快照恢复并继续增量更新。

## 安装

```go
import "github.com/guoyyit/pose-graph-optimizer-go"
```

```bash
go test ./...          # 单元/集成测试
go test -race ./...    # 竞态检测
```

## 快速开始

```go
cfg := posegraph.DefaultConfig()
svc, err := posegraph.NewService(cfg)
if err != nil { log.Fatal(err) }
defer svc.Close()
ctx := context.Background()

// 节点：ID + 可选初始位姿提示
_ = svc.AddNode(ctx, posegraph.Node{ID: "a"})
_ = svc.AddNode(ctx, posegraph.Node{ID: "b", Hint: posegraph.Pose{X: 10}})

// 相对约束：z = Log(T_from^-1 * T_to)，给协方差或信息矩阵之一
err = svc.AddConstraint(ctx, posegraph.Constraint{
    ID:          "ab",
    From:        "a", To: "b",
    Measurement: posegraph.Pose{X: 10},
    Covariance:  posegraph.Mat3{0.05,0,0, 0,0.05,0, 0,0,0.02}, // dx,dy,dθ
    Kind:        posegraph.KindOdometry,
})

res := svc.Query()        // 最新一致版本（不可变）
p, ok := res.PoseAt("b")  // 全局位姿
fmt.Println(res.Diag.Status, res.Diag.MaxEdgeChi2)
for _, r := range res.Residuals { fmt.Println(r.EdgeID, r.Chi2, r.Weight) }
```

协方差/信息是 3×3 行主序矩阵 `Mat3`（顺序 dx, dy, dθ）：

```go
cov := posegraph.Mat3{0.05,0,0, 0,0.05,0, 0,0,0.02}
```

## 约束模型

- **节点 `Node`**：`ID` + `Hint`（初始位姿，仅对锚点或里程计树不可达节点生效）。
- **约束 `Constraint`**：
  - `From`、`To`：端点节点 ID；
  - `Measurement`：相对位姿 z，满足 T_to ≈ T_from ⊞ z（SE(2) 群合成）；
  - `Covariance` **或** `Information`（Σ 或 Σ⁻¹）：显式 `Information` 优先；只给 `Covariance` 时求逆得到信息；都不给则用单位信息；非正定协方差返回 `ErrInvalidConstraint`；
  - `Kind`：`KindOdometry`（顺序里程计，受信任）或 `KindLoop`（回环，需门控）。
- **残差**（右扰动流形）：`e = Log(z⁻¹ · T_from⁻¹ · T_to)`，逐约束代价 `½ w · eᵀ Σ⁻¹ e`。
  Jacobian 由高精度中心有限差分在 SE(2) 切空间上计算。
- **规范自由度（gauge）**：固定锚点节点（`AnchorID`，缺省取字典序最小 ID），位姿解因此唯一。
- **边际协方差**：在**消去锚点自由度**的信息矩阵 H_ff 上求逆得到 `Σ = H_ff⁻¹`；锚点节点协方差为零，表示完全固定。

### SE(2) 约定

- 位姿 `Pose{X, Y, Theta}`，角度弧度并归一化到 `[-π, π)`；
- `Compose`、`Inverse`、`Exp`、`Log` 使用闭式 SE(2) 指数/对数（带小角度泰勒展开）；
- 平移与旋转严格耦合（`V(θ)` 旋转平移雅可比）。

## 增量合并与确定性规则

`AddConstraint` 对**同一无序节点对**按如下规则处理（结果只依赖图内容，不依赖提交顺序）：

1. **乱序（端点缺失）**：约束进入 `pending` 缓冲，返回 `ErrPending`；两端点齐备后按边 ID 字典序依次提升。
2. **幂等重复（同 ID）**：直接忽略，不重复计数。
3. **兼容重复**（里程计，测量差异 χ² ≤ `DuplicateChi2Threshold`，默认 7.815）：
   在 SE(2) 切空间按信息加权**融合**，信息相加（独立观测），老边存活并保留其方向，新边标记 `EdgeFused`。
4. **冲突**（里程计，χ² 超阈值）：信息强度（信息矩阵迹）更大者胜，强度相同按边 ID 字典序决胜；败者标记 `EdgeRejected`，原因写入 `reason`。
5. **回环边**不参与图层冲突取代——每条回环都被保留为独立观测，交由优化器 χ² 门控裁决，避免“再次看到同一地点”被静默丢弃。

初始猜测由**仅里程计边**构成的生成树 BFS 给出（邻居按边 ID 字典序），回环边不参与，因此回环到达顺序不会扭曲线性化点。所有内部遍历均排序，保证固定输入下逐位可复现。

## 回环稳健策略（门控 + 鲁棒核）

每次优化外循环：

1. **预门控（pre-gate）**：在里程计初始解上先评估所有回环。χ² > 门限（默认 99% 分位 ≈ 11.345，自由度 3）的回环在其能扭曲轨迹**之前**剔除（权重置 0，`EdgeRejected`）。
2. **软降权（soft band）**：χ² 介于 95%（≈7.815）与 99% 之间的回环保留但降权 `w = gateLo/χ²`（下限 0.25），记 `EdgeDownweighted`。
3. **Huber 核**：对存活回环按标准化残差 `r=√χ²` 施加 Huber 核（`HuberDelta` 默认 4.0；`r≤δ` 权重 1，否则 δ/r）。
4. 门控/加权后重新 LM 求解，最多 4 轮外迭代直至权重稳定。
5. 被拒回环记录在 `Diagnostics.Rejected`；软降权在 `Downweighted`；图层冲突在 `Conflicts`。拒绝在后续版本持续生效（除非提交不同 ID 的新测量）。

被拒回环不参与最终统计（`MaxEdgeChi2`、RMS 只统计权重 > 0 的边）。

## 失败/病态诊断（绝不“高残差却报成功”）

`Diagnostics` 给出：`Status`、`OK`、`Iterations`、`FinalCost`、`MaxEdgeChi2`、
`RMSError`、`Condition`、`MinPivot`、`Stalled`、`Message` 与门控/冲突事件。

`Status` 取值：

| 状态 | 含义 |
|---|---|
| `SolveEmpty` | 图中无节点 |
| `SolveConverged` | 满足容差且通过质量检查，唯一的“成功”状态（`OK=true`） |
| `SolveStagnated` | 有明显梯度但任何阻尼都无法再降代价（真停滞/冲突） |
| `SolveMaxIterations` | 迭代预算耗尽未收敛 |
| `SolveIllConditioned` | Cholesky 失败或条件数超 `MaxCondition` |
| `SolveHighResidual` | 号称收敛但存在无法再降权的硬（里程计）离群边，χ² 超 `MaxResidualChi2` |
| `SolveUnderconstrained` | 存在无任何有效约束的节点（结构性秩亏） |

实现要点：

- Cholesky 的枢轴判定是**尺度感知**的（按各坐标自身信息的相对容差），不会把强缩放块误判为秩亏；上三角显式清零，回代只读 L。
- 无下降步时，结合**缩放梯度**、**最坏边 χ²** 与代价综合判定收敛或停滞，避免近最优点误报。
- Service 在求解无法产出成功解时返回 `ErrDegenerate` / `ErrNotConverged` 等错误，**但保留并继续暴露上一可用版本**。

## 并发与版本语义

- 所有 `Service` 方法可并发调用；内部 `sync.RWMutex` 串行化写、`Query` 取读锁。
- `Query()` 返回的 `*Result` 在发布后**永不被修改**（每次基于图克隆求解后原子替换指针），读者不会看到半更新或 LM 中间迭代状态。
- 版本号单调递增；`QueryAt(v)` 可取最近若干历史版本（环形保留最近 16 个）。
- 每次提交都会发布新版本（包括标记为失败诊断的版本），便于观测；失败时上一成功版本依旧可查。

## 持久化与恢复

设置 `Config.PersistenceDir` 即启用，目录内含：

- `snapshot.json`：最近一次完整一致快照（节点、边及其状态/权重、门控决定、版本号）；
- `wal.log`：快照之后的 JSON Lines 预写日志（节点/约束提交）。

保证：

- **先写 WAL 再发布**：每条变更 `fsync` 落盘后才进行优化与发布，崩溃不丢已接受约束；
- 快照通过 **临时文件 + fsync + rename 原子替换**，随后截断 WAL；
- 重启时加载快照 → 重放 WAL（确定性地走同一合并/挂起路径）→ 重新优化，恢复门控决定与版本；
- `SnapshotEvery`（默认 100 次变更）控制快照轮转。

## 配置项（`DefaultConfig()`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `AnchorID` | `""`（取最小 ID） | 固定的规范锚点 |
| `MaxIterations` | 50 | LM 最大迭代次数 |
| `FunctionTolerance` | 1e-9 | 相对代价下降收敛阈值 |
| `StepTolerance` | 1e-8 | 切空间步长收敛阈值 |
| `GradientTolerance` | 1e-10 | 绝对梯度收敛阈值（另含缩放梯度判据） |
| `InitialDamping` | 1e-4 | LM 初始阻尼 λ |
| `DampingGrowth` / `DampingShrink` | 10 / 0.5 | λ 自适应倍率 |
| `MinPivotRel` | 1e-10 | 尺度感知 Cholesky 枢轴容差 |
| `MaxCondition` | 1e14 | 条件数上限，超过判病态 |
| `GateConfidence` | 0.99 | 回环硬拒绝 χ² 置信度（dof=3，≈11.345） |
| `DownweightConfidence` | 0.95 | 软降权下沿（≈7.815） |
| `MaxResidualChi2` | 25 | “收敛”解最坏边 χ² 上限（5σ），超过且无法降权则报失败 |
| `HuberDelta` | 4.0 | 回环 Huber 核过渡点（标准化残差），≤0 关闭 |
| `DuplicateChi2Threshold` | 7.815 | 兼容重复融合的 χ² 上限 |
| `PersistenceDir` | `""`（关闭） | 快照+WAL 目录 |
| `SnapshotEvery` | 100 | 多少次变更轮转一次快照 |

## 错误哨兵

`ErrInvalidConstraint`、`ErrPending`、`ErrConstraintRejected`、`ErrDegenerate`、
`ErrNotConverged`、`ErrNoNodes`。

## 测试覆盖

`*_test.go` 覆盖：SE(2) 群运算与卡方分位、矩形/带旋转回环优化、
增量乱序逐位可复现、重复融合与冲突决胜、回环硬拒绝与软降权门限、
孤立节点/病态诊断、边际协方差正定性与回环收紧、并发读写（`-race`）、
快照+WAL 重启恢复及恢复后继续增量、失败后保留可用版本。

## 范围与限制

- 仅 SE(2)（平面 x, y, θ）；3×3 稠密块 + 稠密 Cholesky，面向中小规模图（数百到数千节点）。
- 约束为成对相对位姿；当前不包含全局 GPS/锚点观测类型（可用固定锚点近似）。
- 大规模稀疏场景可将稠密求解替换为稀疏 LDL/PCG（接口已隔离在 `linalg.go`）。
