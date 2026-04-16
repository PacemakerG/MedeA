# Rebuttal 与修订稿对应关系说明

说明：
- Rebuttal 文件为 `rebuttal_vrih_response.tex`。
- 修订稿 LaTeX 文件为 `MedeA`。
- 下文表格中的行号默认分别对应这两个文件的当前工作区版本。
- 这里的“对应改动”指 rebuttal 中说明的修改，在修订稿 LaTeX 中实际落到的具体位置。

## 1. Summary of Major Revisions 对应

| Rebuttal 修改点 | Rebuttal 行号 | `MedeA` 对应行号 | 对应说明 |
| --- | --- | --- | --- |
| 重写方法描述，明确从 ECG 输入到 backbone 输出、Q/K/V、query-wise attention、class-level explanation map 的数据流 | `41-43` | `74-133`, `279-307` | `Shared Backbone`、`Disease-Specific Attention Heads`、`MQAH`、`Attention Visualization Pipeline` 以及 Figure 1/2 说明共同落实了这项修改。 |
| 明确 multi-query 向量是全局可学习参数，不是输入生成；去掉含糊的温度符号，改为 scaled dot-product 写法 | `43-44` | `104-120`, `279-307` | `MQAH` 小节直接说明 query 是 learnable parameters，并在公式中统一改成 scaled dot-product attention。 |
| 扩充实验协议，补充分折、优化器、seed、augmentation，以及区分本地复现实验和文献结果 | `44-45` | `158-180`, `231-253` | `Experimental Protocol` 负责配置与 split；`Representative Local Baselines` 负责区分 same-protocol in-house comparison 与 context-only comparison。 |
| 增加效率比较，包括参数量、FP32 权重体积和轻量 latency benchmark，并弱化部署性表述 | `45-46` | `255-276` | `Efficiency Analysis` 小节和表格完整落实。 |
| 将叙述从“全面优于”改为“有竞争力且具备内生可解释性” | `46-47` | `206`, `253-275`, `333-336`, `356` | 结果、讨论和结论都改成 competitive/accuracy-interpretability trade-off 的表述。 |
| 强化局限性与临床泛化讨论，包括跨医院、设备差异、采样率差异、回顾性公共数据与标注质量依赖 | `47-48` | `29-30`, `337-352`, `356` | 摘要、`Clinical Applicability and Generalization`、`Limitations`、结论都加入了这些限制说明。 |
| 修正 IMLENet 引用错配，并尽量用同行评审版本替代 arXiv-only 引用 | `48-49` | `58`, `380-381` | `Related Work` 中写明已纠正 IMLENet；参考文献区保留了最终 bibliography 插入说明。 |
| 重写 Figure 1--3 的说明，使图更易解释和复现 | `49-50` | `277-329` | `Qualitative Analysis and Figure Clarifications`、Figure 1/2/3 captions 与正文解释共同落实。 |

## 2. Editor Response 对应

| Rebuttal 行号 | `MedeA` 对应行号 | 对应说明 |
| --- | --- | --- |
| `54-58` | `41-49`, `158-180`, `255-329`, `337-356`, `58`, `380-381` | Editor 要求的“逐条回应并明确记录所有实质性修订”，在引言修订概述、实验协议、效率分析、图注改写、讨论/局限性扩展以及参考文献修正中全部落地。 |

## 3. Reviewer #1 逐条对应

| Comment | Rebuttal 行号 | `MedeA` 对应行号 | 对应说明 |
| --- | --- | --- | --- |
| 1.1 Class imbalance handling | `62-68` | `134-156`, `229` | 新增 `Loss Function and Class Imbalance`，明确 BCEWithLogitsLoss、未使用额外 class balancing，并补充 prevalence 与 `pos_weight` 参考值；结果部分再次呼应 HYP 的困难性。 |
| 1.2 Backbone 与 MQAH 关系不清 | `70-76` | `74-133`, `279-307` | Method 中把输入 `(B,12,1000)`、backbone 输出 `(B,512,L')`、MQAH attention shape、query averaging、插值和平滑全部写清，并同步到 Figure 1/2。 |
| 2.1 Table 3 的公平性与可复现性不足 | `78-84` | `158-180`, `231-253` | `Experimental Protocol` 补充优化器、学习率、权重衰减、batch size、seed、augmentation、环境；结果部分明确 local baseline 仅作 context，不再暗示所有方法完全同协议。 |
| 2.2 ECGNet、IMLENet 等 split 与调参协议是否一致 | `86-92` | `179-180`, `231-253` | 修订稿明确区分 same-protocol in-house comparison 与 context-only comparison，并删除“所有外部方法都统一调参”的暗示。 |
| 2.3 缺少效率分析 | `94-116` | `255-276` | `Efficiency Analysis` 中加入参数量、权重体积、CPU forward latency，并明确不是完整 FLOPs/峰值内存 benchmark。 |
| 3.1 临床适用性与泛化性讨论不足 | `118-124` | `29-30`, `337-352`, `356` | 摘要、讨论和结论都显式加入 retrospective public datasets、hospital shift、device shift、sampling-rate mismatch、annotation quality 等限制。 |
| 4.1 IMLENet 引用错配 | `126-132` | `58`, `380-381` | `Related Work` 里写明已修正 IMLENet mismatch；参考文献区保留最终整理版 bibliography 的插入说明。 |
| 5.1 Figure 1/2 缺少维度与输出尺寸说明 | `134-140` | `277-307` | Figure 1 和 Figure 2 的正文描述与 caption 都补上输入维度、feature map 维度、query 维度、attention score 维度和输出含义。 |
| 5.2 Figure 3 需要计数或更清楚的构造说明 | `142-152` | `309-319` | 修订稿明确 Figure 3 是 multi-label prediction-rate matrix `P(Pred=j | True=i)`，并说明其配套 co-occurrence count matrix。 |
| 5.3 arXiv 引用过多 | `154-160` | `58`, `380-381` | 代码层面的对应位置仍是 `Related Work` 的修订说明与参考文献占位说明，表示最终提交版本会使用经过整理的 bibliography。 |

## 4. Reviewer #2 逐条对应

| Comment | Rebuttal 行号 | `MedeA` 对应行号 | 对应说明 |
| --- | --- | --- | --- |
| 1 Figure 1 分辨率低 | `164-170` | `277-293` | Figure 1 被改写为带显式尺寸注释的高质量/矢量风格占位图，并在 caption 中说明应展示 backbone 到 disease heads 的维度接口。 |
| 2 MQAH 细节不清，query 来源与温度参数含糊 | `172-182` | `104-120`, `279-307` | `MQAH` 小节明确 queries 是 sample-independent learnable parameters，温度写法改为 `QK^T / sqrt(d)`；Figure 2 和 caption 也同步修正。 |
| 3 MSW-TN 某些指标优于 MeDeA | `184-192` | `206`, `253-275`, `333-336` | 结果与讨论不再宣称绝对最优，而是明确写成 competitive model，并把优势表述为 interpretability-performance trade-off。 |
| 4 多 query 的计算开销没有说明 | `194-200` | `255-276` | 同样由 `Efficiency Analysis` 落地，并在段落中明确多 query 提升了解释分解能力，但会增加参数量和 forward cost。 |
| 5 缺少足够证据说明不同 query 捕获不同模式 | `202-208` | `114-132`, `322-352` | 修订稿把 per-query attention maps 与 query-averaged explanation 区分开写，并在 `Limitations` 中承认仍需更系统的 per-query visualization。 |

## 5. 当前代码层面的备注

- `Comment 4.1` 和 `Comment 5.3` 已经在 `MedeA:58` 与 `MedeA:380-381` 反映为“已修正/将插入最终整理版参考文献”的说明，但当前 `MedeA` 文件里的 bibliography 仍是占位状态，不是最终完整条目。
- `Comment 2.1`、`Comment 2.2` 中提到的 `Table 3`，在当前修订稿 LaTeX 中并不是以同一个表号直接出现，而是拆解落实在 `Experimental Protocol` 与 `Representative Local Baselines` 两部分。
- `Comment 5.1` 与 Reviewer #2 Comment 1 中关于 Figure 1/2 的修改，目前在 LaTeX 中以可编译 placeholder 形式体现；如果后续替换成最终矢量图文件，这份对应关系文档的章节映射仍然成立。
