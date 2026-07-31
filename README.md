# ek_response
多质点葫芦串杆系结构模型动力响应特征预测

## 模型介绍
模型包括仿真器、动力响应生成器或读取器（地面加速度）、基于LSTM-RNN的学习模块

### 模型结构（1d分析）
基于 Seq2Seq + Attention 模型 ，建立单个结构分析模型（糖葫芦简化模型，固定参数），加载n条动力加速度波（seq1），提取顶点位移结果，输出（seq2）

### 模型结构（2d with params）
学习“输入地震动 + 输入结构参数 → 输出结构响应”的映射关系

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

## 1d仿真得到的动力响应结果（右）
<img width="2100" height="1800" alt="sample_frame_response" src="https://github.com/user-attachments/assets/5e9216cf-8cfd-45c8-9ed3-19c7b91580e6" />

## 2d仿真结果
<img width="2250" height="1500" alt="sample_frame_params_response" src="https://github.com/user-attachments/assets/64ed0e7e-7d93-401c-84d2-d931bcd1d1d6" />


## 学习过程

### 1d模型
<img width="1500" height="600" alt="training_history" src="https://github.com/user-attachments/assets/53d7ae8b-8584-452f-9c6b-c9a76ca13c34" />

### 2d模型（带模型参数）
<img width="1500" height="600" alt="frame_training_history" src="https://github.com/user-attachments/assets/80a9bf4a-48c5-473b-b2a2-4347ca1f8704" />


## 测试结果（1d）
<img width="2250" height="4500" alt="random_test_comparison" src="https://github.com/user-attachments/assets/9d4bbbb5-8b8a-4948-884c-e3be0d00290d" />
## 测试结果（2d with params）
<img width="2250" height="4500" alt="frame_params_test_comparison" src="https://github.com/user-attachments/assets/8f7973ad-acfe-4c51-a358-ba23362997f2" />

