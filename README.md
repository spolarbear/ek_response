# ek_response
多质点葫芦串杆系结构模型动力响应特征预测

## 模型介绍
模型包括仿真器、动力响应生成器或读取器（地面加速度）、基于LSTM-RNN的学习模块

改进的损失函数，加上方向惩罚，防止正负抵消，解决本例震荡结果在 Huber Loss 和 MSE 在回归任务中正负偏差抵消的问题。

### 模型结构（1d分析）
基于 Seq2Seq + Attention 模型 ，建立单个结构分析模型（糖葫芦简化模型，固定参数），加载n条动力加速度波（seq1），提取顶点位移结果，输出（seq2）

### 模型结构（2d with params）

学习“输入地震动 + 输入结构参数 → 输出结构响应”的映射关系

6个参数输入到MLP编码器，生成256维嵌入向量。

### 模型结构（2d with params and transformer）

采用transformer架构，更为成熟，适配大规模样本


## 可变参数

with params 表示输入的是分析模型+动力加载

分析模型参数包括：

1、楼层数	num_stories	3 ~ 8 层	框架总层数

2、跨数	num_bays	1 ~ 3 跨	X方向跨数

3、跨宽	bay_width	4.0 ~ 8.0 m	每跨宽度

4、层高	story_height	3.0 ~ 4.5 m	每层层高

5、节点质量	mass_per_node	20,000 ~ 60,000 kg	每个节点集中质量

6、阻尼比	damping_ratio	0.03 ~ 0.08	结构阻尼比

#### 特征融合

结构参数：静态、低维度（ 层数, 质量, 刚度）。

时程数据：动态、高维度（3000步）。

融合步骤：用神经网络（MLP）将静态参数编码为上下文向量（Context Vector），然后注入到LSTM的解码器（Decoder）中，指导其时序预测。

实现：结构参数不进入Encoder（只编码加速度），而是经过MLP编码后，与Decoder每一步的输入进行拼接。

#### 架构图逻辑：

参数编码器：(层数, 质量, 刚度...) → MLP → 结构嵌入向量 (Context)。

时序编码器：(加速度序列) → LSTM → 时序特征。

时序解码器：每一步输入 [上一步位移, 当前时刻加速度, 结构嵌入向量] → 输出当前步位移。


## 输入的人工波：
<img width="1800" height="1200" alt="loaded_earthquake_waves" src="https://github.com/user-attachments/assets/605c8cdd-0b97-4e6f-9199-69f6ab987ea2" />

## 仿真结果
提取结构顶点位移时程曲线，用以衡量弹性位移角情况。

### 1d仿真得到的动力响应结果（右）
<img width="2100" height="1800" alt="sample_frame_response" src="https://github.com/user-attachments/assets/5e9216cf-8cfd-45c8-9ed3-19c7b91580e6" />

### 2d仿真结果
<img width="2250" height="1500" alt="sample_frame_params_response" src="https://github.com/user-attachments/assets/64ed0e7e-7d93-401c-84d2-d931bcd1d1d6" />


## 学习过程

### 1d模型

<img width="1500" height="600" alt="training_history" src="https://github.com/user-attachments/assets/53d7ae8b-8584-452f-9c6b-c9a76ca13c34" />

### 2d模型（带模型参数）

<img width="1500" height="600" alt="frame_training_history" src="https://github.com/user-attachments/assets/80a9bf4a-48c5-473b-b2a2-4347ca1f8704" />

### 2d模型（transformer）

<img width="2100" height="1500" alt="training_curves" src="https://github.com/user-attachments/assets/179ab9d7-404c-4a73-923e-e4156a4eed8e" />


## 测试结果

### 测试结果（1d）

<img width="2250" height="4500" alt="random_test_comparison" src="https://github.com/user-attachments/assets/9d4bbbb5-8b8a-4948-884c-e3be0d00290d" />

### 测试结果（2d with params）

<img width="2250" height="4500" alt="frame_params_test_comparison" src="https://github.com/user-attachments/assets/8f7973ad-acfe-4c51-a358-ba23362997f2" />

### 测试结果（2d with params and transformer）

<img width="2250" height="4500" alt="prediction_results" src="https://github.com/user-attachments/assets/f8eab4c3-3e47-492f-b7f3-af4c031401b2" />

<img width="2100" height="1500" alt="error_analysis" src="https://github.com/user-attachments/assets/f6c84198-acc5-4292-8416-9fce588c91df" />


## 其他

### 数据缓存（CACHE_FILE）

作用：保存仿真生成的训练数据，避免重复运行 OpenSees保存时机：数据生成完成后（一次性保存）

大小：10万条样本 × 2000步 × 3个数组 ≈ 2-3 GB

### 训练检查点（CHECKPOINT_FILE）

作用：保存训练状态，支持中断恢复

保存时机：每 5 个 epoch 保存一次（可配置）

大小：约 200-500 MB（取决于模型大小）


### 离散空间估计

离散化参数空间总组合数 = 6 × 3 × 5 × 6 × 5 × 6 = 16,200种结构

当前参数空间的物理覆盖中低层框架结构的典型范围

指标	范围

总高度	9 ~ 36 m

总宽度	4 ~ 24 m (1跨) 或 8 ~ 24 m (3跨)

高宽比	0.375 ~ 9.0

总质量	6 × (20k~60k) × (2~4节点/层) = 120k ~ 1,440k kg

基本周期	约 0.3 ~ 2.0 s (取决于高度和刚度)


### 训练环境

处理器	Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz   2.39 GHz

机带 RAM	128 GB (128 GB 可用)

存储	447 GB ThinkSystem M.2 VD, 4.36 TB Lenovo RAID 730-8i 1GB

显卡	NVIDIA GeForce RTX 3080 (10 GB)

模型2：仿真1h；训练epoch=60约6h
