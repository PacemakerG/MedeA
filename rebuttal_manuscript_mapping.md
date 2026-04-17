# Rebuttal 与修订稿对应关系说明

说明：
- Rebuttal 文件为 `rebuttal_vrih_response.tex`。
- 修订稿 LaTeX 文件为 `MedeA.tex`。
- 下文表格中的行号默认分别对应这两个文件的当前工作区版本。
- 这里的“对应改动”指 rebuttal 中说明的修改，在修订稿 LaTeX 中实际落到的具体位置。

## 1. Summary of Major Revisions 对应

| Rebuttal 修改点 | Rebuttal 行号 | `MedeA.tex` 对应行号 | 对应说明 |
| --- | --- | --- | --- |
| 重写方法描述，明确从 ECG 输入到 backbone 输出、Q/K/V、query-wise attention、class-level explanation map 的数据流 | `41-43` | `80-142`, `302-337` | `Shared Backbone`、`Disease-Specific Attention Heads`、`MQAH`、`Attention Visualization Pipeline` 以及 Figure 1/2 说明共同落实了这项修改。 |
| 明确 multi-query 向量是全局可学习参数，不是输入生成；去掉含糊的温度符号，改为 scaled dot-product 写法 | `43-44` | `104-128`, `304-337` | `MQAH` 小节直接说明 query 是 learnable parameters，并在公式中统一改成 scaled dot-product attention；Figure 2 与 caption 也同步写清。 |
| 扩充实验协议，补充分折、优化器、seed、augmentation，以及区分本地复现实验和文献结果 | `44-45` | `162-193`, `245-273` | `Experimental Protocol` 负责配置与 split；`Representative Local Baselines` 负责区分 same-protocol in-house comparison 与 context-only comparison。 |
| 增加效率比较，包括参数量、FP32 权重体积和轻量 latency benchmark，并弱化部署性表述 | `45-46` | `275-300` | `Efficiency Analysis` 小节和表格完整落实。 |
| 将叙述从“全面优于”改为“有竞争力且具备内生可解释性” | `46-47` | `220`, `273`, `355-396` | 结果、讨论和结论都改成 competitive/accuracy-interpretability trade-off 的表述。 |
| 强化局限性与临床泛化讨论，包括跨医院、设备差异、采样率差异、回顾性公共数据与标注质量依赖 | `47-48` | `30`, `375-396` | 摘要、`Clinical Applicability and Generalization`、`Limitations`、结论都加入了这些限制说明。 |
| 修正 IMLENet 引用错配，并尽量用同行评审版本替代 arXiv-only 引用 | `48-49` | `58-64`, `420-426` | `Related Work` 中记录了 IMLENet 错配已在最终整理版 bibliography 中改正，并说明相关 ECG 对比方法条目已复核；参考文献区保留了工作副本占位与最终 bibliography 的对应说明。 |
| 重写 Figure 1--3 的说明，使图更易解释和复现 | `49-50` | `302-363` | `Qualitative Analysis and Figure Clarifications`、Figure 1/2/3 captions 与正文解释共同落实。 |

## 2. Editor Response 对应

| Rebuttal 行号 | `MedeA.tex` 对应行号 | 对应说明 |
| --- | --- | --- |
| `54-58` | `41-49`, `162-193`, `275-363`, `375-396`, `58-64`, `420-426` | Editor 要求的“逐条回应并明确记录所有实质性修订”，在引言修订概述、实验协议、效率分析、图注改写、讨论/局限性扩展以及参考文献修正中全部落地。 |

## 3. Reviewer #1 逐条对应

| Comment | Rebuttal 行号 | `MedeA.tex` 对应行号 | 对应说明 |
| --- | --- | --- | --- |
| 1.1 Class imbalance handling | `62-68` | `146-160`, `243` | `Loss Function and Class Imbalance` 现已明确主实验使用标准 `BCEWithLogitsLoss()` 且未使用 `pos_weight` 或其他显式重加权；保留 prevalence 仅用于表征数据不平衡，并补充这一训练选择对少数类可能带来影响的限制说明。 |
| 1.2 Backbone 与 MQAH 关系不清 | `70-76` | `80-142`, `302-337` | Method 中把输入 `(B,12,1000)`、backbone 输出 `(B,512,L')`、MQAH attention shape、query averaging、插值全部写清，并同步到 Figure 1/2。 |
| 2.1 Table 3 的公平性与可复现性不足 | `84-90` | `168-189`, `247-249`, `275-279` | `Experimental Protocol` 现已明确主 MeDeA-family 的默认训练设置，并说明这些设置只对应 main in-house PTB-XL runs；`Representative Local Baselines` 明确 local baseline 仅作 contextual reference；模型规模信息单独移至 `Efficiency Analysis`，避免与协议说明混淆。 |
| 2.2 ECGNet、IMLENet 等 split 与调参协议是否一致 | `94-102` | `191-193`, `251-253` | 修订稿现已明确说明：当外部方法缺乏可直接复现的官方实现或完整协议细节时，不进行作者侧补参重实现，以避免 implementer bias；ECGNet、IMLENet、MSW-TN 等结果因此被定位为 literature-reported task-level references，而统一本地协议下的 model-family comparisons 与 ablations 才是核心受控证据。 |
| 2.3 缺少效率分析 | `104-126` | `277-300` | `Efficiency Analysis` 中加入参数量、FP32 权重体积、batch-1 CPU forward latency，并明确这些只是当前代码库可复现的效率参考量；修订稿同时删除了任何会让人误解为已完成完整 FLOPs/峰值内存/部署级系统 benchmark 的表述。 |
| 3.1 临床适用性与泛化性讨论不足 | `130-136` | `30`, `375-392`, `396` | 摘要、讨论和结论现已明确限制当前证据仅来自 retrospective public datasets，并补充 hospital shift、device shift、sampling-rate mismatch、annotation quality 等外部有效性风险；同时明确指出 interpretability 不能替代临床验证，未来仍需 prospective cross-site / cross-device validation。 |
| 4.1 IMLENet 引用错配 | `140-146` | `58-64`, `420-426` | `Related Work` 现已记录 IMLENet 错配在最终整理版 bibliography 中改正，并补充“相关 ECG 对比方法条目与编号已复核”的说明；参考文献区同时保留了工作副本占位与最终 bibliography 的对应注释。 |
| 5.1 Figure 1/2 缺少维度与输出尺寸说明 | `148-154` | `302-337` | Figure 1 和 Figure 2 的正文描述与 caption 都补上输入维度、feature map 维度、query 维度、attention score 维度和输出含义。 |
| 5.2 Figure 3 需要计数或更清楚的构造说明 | `156-168` | `340-354` | 修订稿明确 Figure 3 是 row-normalized multi-label prediction-rate matrix `P(Pred=j | True=i)`，解释了按 `True=i` 统计并逐行归一化的构造方式，并说明其配套 co-occurrence count matrix 用于报告样本支持规模。 |
| 5.3 arXiv 引用过多 | `168-174` | `58-64`, `420-426` | 当前代码层面的对应位置是 `Related Work` 的 bibliography cleanup 说明与参考文献占位说明，表示最终提交版本会优先使用经过整理的 archival bibliography。 |

## 4. Reviewer #2 逐条对应

| Comment | Rebuttal 行号 | `MedeA.tex` 对应行号 | 对应说明 |
| --- | --- | --- | --- |
| 1 Figure 1 分辨率低 | `178-184` | `304-323` | Figure 1 被改写为带显式尺寸注释的更清晰矢量式 schematic/placeholder，并在 caption 中说明 backbone 到 disease heads 的维度接口。 |
| 2 MQAH 细节不清，query 来源与温度参数含糊 | `186-198` | `112-129`, `304-337` | `MQAH` 小节现已明确 queries 是每个 disease-specific head 中 sample-independent global learnable parameters，用于和 feature-projected keys/values 做注意力计算；温度写法也收紧为标准 `QK^T / sqrt(d)`，并说明若写成 temperature form 则对应固定 `tau = sqrt(d)`；Figure 2 和 caption 同步修正。 |
| 3 MSW-TN 某些指标优于 MeDeA | `200-208` | `213-221`, `274`, `371-374` | 结果与讨论不再宣称绝对最优，而是明确说明：在部分报告判别指标上接受有限性能差距，以换取与预测路径直接绑定的 class-specific intrinsic explanations；同时保留内部 single-query 对照中“收益有限、不夸大”的表述。 |
| 4 多 query 的计算开销没有说明 | `208-214` | `277-300` | 同样由 `Efficiency Analysis` 落地，并在段落中明确 richer query decomposition 会增加参数量和 forward cost。 |
| 5 缺少足够证据说明不同 query 捕获不同模式 | `216-228` | `120-129`, `355-393` | 修订稿把 per-query attention maps 与 query-averaged explanation 区分开写，并进一步明确：当前 query-averaged 主图本身不足以单独证明稳定的 query specialization；目前更直接支持的是相对 single-query 对照下更可分解的解释表示和稳定但有限的性能增益，而更强的 distinct-query 结论仍需 dedicated per-query visual panels 或额外统计分析。 |

## 5. 当前代码层面的备注

- `Comment 4.1` 与 `Comment 5.3` 现在已在 `MedeA.tex:58-64` 与 `MedeA.tex:420-426` 以 `% [Reviewer 4.1 revision]` / `% [Reviewer 5.3 revision]` 注释和对应正文说明显式标出；当前 `MedeA.tex` 里的 bibliography 仍是工作副本占位，不是最终完整条目。
- `Comment 2.1`、`Comment 2.2` 中提到的 `Table 3`，在当前修订稿 LaTeX 中并不是以同一个表号直接出现，而是拆解落实在 `Experimental Protocol` 与 `Representative Local Baselines` 两部分。
- `Comment 5.1` 与 Reviewer #2 Comment 1/2 中关于 Figure 1/2 的修改，现在已在 `MedeA.tex:304-337` 以 `% [Reviewer 5.1 revision]`、`% [Reviewer #2 Comment 1 revision]` 和 `% [Reviewer #2 Comment 2 revision]` 注释显式标出；当前 LaTeX 仍以可编译 placeholder 形式承载最终图示逻辑，如果后续替换成正式矢量图文件，这份章节映射仍然成立。
