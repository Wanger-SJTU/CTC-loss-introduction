
这部分主要介绍 pytorch (1.2.0) 的 ctc_loss 源码实现。

## pytorch ctc loss的使用

### 参数说明
```python
CLASS torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
```

**初始化参数**
- **blank** (int, optional) – blank label. Default 0.
    > 插入空白标签的位置，默认为 0


- **reduction** (string, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the output losses will be divided by the target lengths and then the mean over the batch is taken. Default: 'mean'
    > 对输出loss的处理方式(batch)，有
    > - "none"：不处理 (输出等于N维向量)
    > - "mean"：均值
    > - "sum" ：求和
    > 默认为 **均值**

- **zero_infinity** (bool, optional) – Whether to zero infinite losses and the associated gradients. Default: False Infinite losses mainly occur when the inputs are too short to be aligned to the targets.
    > 当loss变成 `NAN` 时，是否输出为 0
    > 默认 False。 输入序列太小不能跟目标匹配时， loss 可能变成 `NAN`

**调用参数**
- **Log_probs**:  Tensor 大小为 $(T,N,C)$, 其中 $T = \text{input length}$, $N = \text{batch size}$ , and $C = \text{包含blank的类别数}$. 是输出概率的取$\log$计算得到的 (e.g. 由`torch.nn.functional.log_softmax()`得到).

- **Targets**: Tensor 大小为 $(N, S)$ 或者 ($\operatorname{sum}(\text{target\_lengths}))$) , 其中 $N = \text{batch size}$ ，$ \text{如果输入维度是}(N, S), S = \text{其中最长的长度}$. 输入表示了目标的序列，每个元素都是一个标签，但不能包含空白（blank）标签。
    - 输入为$(N,S)$时， 较短的标签padding。
    - 输入为 ($\operatorname{sum}(\text{target\_lengths})$) 时, 标签拼接成为一维向量.

- **Input_lengths**: shape为(N)的张量或元组，但每一个元素的长度必须小于等于T即输出序列长度，一般来说模型输出序列固定后则该张量或元组的元素值均相同

- **Target_lengths**: shape为(N)的张量或元组，其每一个元素指示每个训练输入序列的标签长度，但标签长度是可以变化的

- **Output**: 标量. 只有 当 reduction 是 'none' 的时候输出为N维向量，其中 $N = \text{batch size}$.


### 使用
见于 [ctc pytorch](./src_code/ctc_pytorch.ipynb)

### 源码实现

#### pytorch

#### warp-ctc

[warp-ctc](https://github.com/baidu-research/warp-ctc) 是百度开源的可以应用在CPU和GPU上高效并行的CTC代码库 （library） 介绍 CTC (Connectionist Temporal Classification) 作为一个损失函数，用于在序列数据上进行监督式学习，不需要对齐输入数据及标签。
详细介绍见于 [warp-ctc](wrap-ctc_read.md)
