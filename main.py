# ============================================
# multi_story_frame_seq2seq.py
# 可调层数框架 (OpenSeesPy) + Seq2Seq 训练
# 支持天然波和人工波切换
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import pickle
import time
import warnings
import random
import glob
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')

try:
    import openseespy.opensees as ops
except ImportError:
    print("请安装 openseespy: pip install openseespy")
    exit()

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ============================================
# ========== 全局配置 ==========
# ============================================

MODE = 'train'           # 'train' 或 'test'

# ===== 数据源选择 =====
USE_SYNTHETIC = False    # True: 使用人工合成波, False: 使用天然波
NUM_SYNTHETIC = 2000     # 仅当 USE_SYNTHETIC=True 时有效

# ===== 天然波参数 =====
DZB_FOLDER = './dzb'     # 天然波文件夹
TARGET_DURATION = 20.0   # 截取前20秒
TARGET_DT = 0.01         # 目标时间步长
TARGET_PGA = 0.035       # 目标峰值加速度 (g) (35cm/s² = 0.0357g，取0.035)

# ===== 框架参数 =====
NUM_STORIES = 8
STORY_HEIGHT = 3.0
MASS_PER_STORY = 100000.0
# 截面尺寸 (边长, m)，从底层到顶层
SECTIONS = [1.2, 1.2, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6]
E = 3.25e10
DAMPING_RATIO = 0.05

# ===== 模型参数 =====
MODEL_FILE = 'multi_story_lstm.pth'
EPOCHS = 500
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001

FORCE_REGEN = True
CACHE_FILE = 'multi_story_data.pkl'

# ============================================
# ============================================


# ============================================
# 1. 天然波加载和预处理函数
# ============================================

def parse_earthquake_file(file_path):
    """
    解析地震动文件，自动识别步长
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    values = []
    dt = 0.01
    
    file_name = os.path.basename(file_path)
    name_parts = file_name.replace('.txt', '').replace('.dat', '').replace('.csv', '').split()
    
    for part in reversed(name_parts):
        try:
            dt = float(part)
            if dt > 0 and dt < 1:
                break
        except:
            continue
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            val = float(line)
            values.append(val)
        except ValueError:
            continue
    
    if len(values) == 0:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for val_str in line.split():
                try:
                    val = float(val_str)
                    values.append(val)
                except:
                    continue
    
    return np.array(values), dt


def load_earthquake_files(folder_path, target_duration=20.0, target_pga=0.035, target_dt=0.01):
    """
    加载文件夹中的所有地震动文件，截取前 target_duration 秒，并缩放到目标PGA
    返回 numpy 数组 (n_samples, n_steps)
    """
    if not os.path.exists(folder_path):
        print(f"警告：文件夹 {folder_path} 不存在")
        return np.array([]), []
    
    patterns = ['*.txt', '*.dat', '*.csv']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(folder_path, pattern)))
    files = list(set(files))
    files = [f for f in files if os.path.isfile(f)]
    
    if len(files) == 0:
        print(f"警告：文件夹 {folder_path} 中没有找到地震动文件")
        return np.array([]), []
    
    print(f"找到 {len(files)} 个地震动文件")
    
    all_motions = []
    all_info = []
    target_steps = int(target_duration / target_dt)
    
    for file_path in files:
        try:
            motion_orig, dt_orig = parse_earthquake_file(file_path)
            
            if len(motion_orig) < 10:
                print(f"  跳过: {os.path.basename(file_path)} (数据点太少)")
                continue
            
            # 截取前 target_duration 秒
            n_orig = len(motion_orig)
            t_orig = np.arange(n_orig) * dt_orig
            
            if t_orig[-1] >= target_duration:
                # 插值到目标时间轴
                t_target = np.arange(target_steps) * target_dt
                f = interp1d(t_orig, motion_orig, kind='linear', fill_value='extrapolate')
                motion_resampled = f(t_target)
            else:
                # 如果地震动短于目标时长，截取并填充
                n_avail = int(t_orig[-1] / target_dt)
                motion_resampled = motion_orig[:n_avail]
                if len(motion_resampled) < target_steps:
                    motion_resampled = np.pad(motion_resampled, (0, target_steps - len(motion_resampled)), 'constant')
            
            # 缩放到目标PGA
            pga = np.max(np.abs(motion_resampled))
            if pga > 1e-10:
                motion_scaled = motion_resampled / pga * target_pga
            else:
                motion_scaled = motion_resampled
            
            all_motions.append(motion_scaled)
            all_info.append({
                'file': os.path.basename(file_path),
                'dt_orig': dt_orig,
                'n_orig': n_orig,
                'pga_orig': pga,
                'scale_factor': target_pga / pga if pga > 1e-10 else 1.0
            })
            
            print(f"  加载: {os.path.basename(file_path)}, 原始dt={dt_orig}s, 原始点数={n_orig}, 缩放因子={target_pga/pga if pga>1e-10 else 1.0:.4f}")
            
        except Exception as e:
            print(f"  加载失败: {file_path}, {e}")
    
    if len(all_motions) == 0:
        print("没有加载到任何天然波")
        return np.array([]), []
    
    # 转换为 numpy 数组
    motions_array = np.array(all_motions)
    print(f"成功加载 {len(all_motions)} 条天然波，每条长度 {motions_array.shape[1]} 步")
    return motions_array, all_info


# ============================================
# 2. 合成地震动生成器
# ============================================

def generate_synthetic_ground_motion(duration=20.0, dt=0.01, target_pga=0.035):
    n = int(duration / dt)
    t = np.linspace(0, duration, n)
    
    envelope = np.exp(-0.3 * t / duration * 5) * (1 - np.exp(-3 * t / duration))
    envelope = envelope / (np.max(envelope) + 1e-10)
    
    n_freqs = np.random.randint(10, 30)
    freqs = np.random.uniform(0.5, 20, n_freqs)
    amps = np.random.uniform(0.3, 1.0, n_freqs)
    phases = np.random.uniform(0, 2*np.pi, n_freqs)
    
    motion = np.zeros(n)
    for f, a, p in zip(freqs, amps, phases):
        motion += a * np.sin(2*np.pi*f*t + p)
    
    motion = motion * envelope
    pga = np.max(np.abs(motion))
    if pga > 1e-10:
        motion = motion / pga * target_pga
    
    return motion


def generate_batch_synthetic(num_samples, duration=20.0, dt=0.01, target_pga=0.035):
    motions = []
    for _ in range(num_samples):
        motion = generate_synthetic_ground_motion(duration, dt, target_pga)
        motions.append(motion)
    return np.array(motions)


# ============================================
# 3. OpenSeesPy 多楼层框架仿真函数
# ============================================

def run_frame(ground_motion, dt=0.01, num_stories=NUM_STORIES, 
              story_height=STORY_HEIGHT, mass_per_story=MASS_PER_STORY,
              sections=SECTIONS, E=E, damping_ratio=DAMPING_RATIO):
    """
    通用框架分析函数
    返回顶部位移 (mm)
    """
    n_steps = len(ground_motion)
    if len(sections) != num_stories:
        raise ValueError(f"sections长度 ({len(sections)}) 必须等于楼层数 ({num_stories})")
    
    A_list = [s**2 for s in sections]
    I_list = [(s**4)/12 for s in sections]
    
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    
    # 节点
    ops.node(1, 0.0, 0.0)
    for i in range(1, num_stories+1):
        ops.node(i+1, 0.0, i*story_height)
    
    ops.fix(1, 1, 1, 1)
    for i in range(2, num_stories+2):
        ops.fix(i, 0, 1, 1)
    
    ops.uniaxialMaterial('Elastic', 1, E)
    ops.geomTransf('Linear', 1)
    
    for i in range(num_stories):
        node_i = i+1
        node_j = i+2
        A = A_list[i]
        I = I_list[i]
        ops.element('elasticBeamColumn', i+1, node_i, node_j, A, E, I, 1)
    
    for i in range(2, num_stories+2):
        ops.mass(i, mass_per_story, 0.0, 0.0)
    
    try:
        eigen = ops.eigen('-fullGenLapack', 1)
        omega = np.sqrt(eigen[0])
    except:
        k_est = 3 * E * I_list[0] / (story_height**3)
        omega = np.sqrt(k_est / mass_per_story) * 0.7
    beta_k = 2 * damping_ratio / omega
    ops.rayleigh(0.0, beta_k, 0.0, 0.0)
    
    accel_values = list(-1.0 * np.array(ground_motion) * 9.81)
    ops.timeSeries('Path', 1, '-dt', dt, '-values', *accel_values)
    ops.pattern('UniformExcitation', 1, 1, '-accel', 1)
    
    ops.wipeAnalysis()
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1e-6, 100)
    ops.algorithm('Newton')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')
    
    top_node = num_stories + 1
    top_disp_m = []
    for step in range(n_steps):
        ok = ops.analyze(1, dt)
        if ok != 0:
            print(f"分析在第 {step} 步失败，返回零")
            return np.zeros(n_steps)
        d = ops.nodeDisp(top_node, 1)
        top_disp_m.append(d)
    
    return np.array(top_disp_m) * 1000.0


# ============================================
# 4. 数据生成和缓存
# ============================================

def get_data_source():
    if USE_SYNTHETIC:
        return "合成地震动"
    else:
        return f"天然波 (来自 {DZB_FOLDER})"


def generate_and_cache_data(num_samples=2000, force_regen=False, 
                            num_stories=NUM_STORIES, sections=SECTIONS):
    if os.path.exists(CACHE_FILE) and not force_regen:
        print(f"✓ 加载缓存数据: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        return data['motions'], data['responses']
    
    print(f"\n数据源: {get_data_source()}")
    
    # 生成或加载地震动
    if USE_SYNTHETIC:
        print(f"生成 {num_samples} 条合成地震动...")
        motions = generate_batch_synthetic(num_samples, duration=TARGET_DURATION, 
                                          dt=TARGET_DT, target_pga=TARGET_PGA)
    else:
        print(f"加载天然波从 {DZB_FOLDER}...")
        motions, info = load_earthquake_files(DZB_FOLDER, 
                                              target_duration=TARGET_DURATION, 
                                              target_pga=TARGET_PGA,
                                              target_dt=TARGET_DT)
        if len(motions) == 0:
            print("错误：没有加载到任何天然波，请检查文件夹路径")
            return None, None
        print(f"  共加载 {len(motions)} 条天然波")
        
        # 绘制加载的天然波
        print("\n绘制前5条天然波...")
        fig, axes = plt.subplots(5, 1, figsize=(12, 8))
        t = np.arange(motions.shape[1]) * TARGET_DT
        for i in range(min(5, len(motions))):
            axes[i].plot(t, motions[i], linewidth=0.8)
            axes[i].set_ylabel('Accel (g)')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'天然波 {i+1}')
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle(f'加载的天然波 (PGA={TARGET_PGA}g)')
        plt.tight_layout()
        plt.savefig('loaded_earthquake_waves.png', dpi=150)
        print("✓ 天然波图已保存: loaded_earthquake_waves.png")
        plt.show()
    
    # 确保 motions 是 numpy 数组
    if not isinstance(motions, np.ndarray):
        motions = np.array(motions)
    
    print(f"运行 OpenSeesPy 框架仿真 (楼层数={num_stories})...")
    responses = []
    if HAS_TQDM:
        iterator = tqdm(motions, desc="仿真进度")
    else:
        iterator = motions
    
    for i, motion in enumerate(iterator):
        resp = run_frame(motion, dt=TARGET_DT, num_stories=num_stories, 
                         sections=sections)
        responses.append(resp)
        if (i+1) % 200 == 0:
            max_disp = np.max(np.abs(resp))
            print(f"  第 {i+1} 条完成, 最大位移: {max_disp:.4f} mm")
    
    responses = np.array(responses)
    
    # 检查
    mean_disp = np.mean(responses)
    std_disp = np.std(responses)
    max_disp = np.max(np.abs(responses))
    print(f"  位移均值: {mean_disp:.4f} mm, 标准差: {std_disp:.4f} mm")
    print(f"  最大绝对位移: {max_disp:.4f} mm")
    
    if max_disp > 1000 or np.isnan(mean_disp) or np.isinf(mean_disp):
        raise ValueError("位移数据异常")
    
    # 绘制前5条响应
    print("\n绘制5条地震动响应曲线...")
    fig, axes = plt.subplots(5, 2, figsize=(14, 12))
    t = np.arange(motions.shape[1]) * TARGET_DT
    for i in range(min(5, len(motions))):
        axes[i, 0].plot(t, motions[i], linewidth=0.8)
        axes[i, 0].set_ylabel('Accel (g)')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_title(f'Sample {i+1} Accel')
        axes[i, 1].plot(t, responses[i], linewidth=0.8, color='red')
        axes[i, 1].set_ylabel('Disp (mm)')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_title(f'Sample {i+1} Disp')
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    plt.suptitle(f'Ground Motions and Responses ({num_stories}-story Frame)')
    plt.tight_layout()
    plt.savefig('sample_frame_response.png', dpi=150)
    print("✓ 响应曲线图已保存: sample_frame_response.png")
    plt.show()
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'motions': motions, 'responses': responses}, f)
    print(f"✓ 缓存已保存: {CACHE_FILE}")
    
    return motions, responses


# ============================================
# 5. Seq2Seq Dataset
# ============================================

class Seq2SeqDataset(Dataset):
    def __init__(self, motions, responses, mean=None, std=None):
        # 统一长度 - 取最小长度
        min_len = min(motions.shape[1], responses.shape[1], 3000)
        self.motions = motions[:, :min_len].astype(np.float32)
        self.responses = responses[:, :min_len].astype(np.float32)
        if mean is None or std is None:
            self.mean = np.mean(self.responses)
            self.std = np.std(self.responses) + 1e-8
        else:
            self.mean = mean
            self.std = std
        self.responses_norm = (self.responses - self.mean) / self.std
        self.seq_len = min_len
        
    def __len__(self):
        return len(self.motions)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.motions[idx]).unsqueeze(1)
        y = torch.FloatTensor(self.responses_norm[idx]).unsqueeze(1)
        return x, y


# ============================================
# 6. Seq2Seq LSTM 模型
# ============================================

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, 
                 output_size=1, seq_len=3000, dropout=0.2):
        super(Seq2SeqLSTM, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        decoder_input = x
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        output = self.fc(decoder_output)
        return output


# ============================================
# 7. 训练函数
# ============================================

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    train_losses = []
    val_losses = []
    
    print(f"\n开始训练 {epochs} 个 epochs...")
    print(f"训练批次数: {len(train_loader):,}, 验证批次数: {len(val_loader):,}")
    print("-" * 70)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                         leave=False, mininterval=0.5)
        
        for batch in train_iter:
            X_batch = batch[0].to(device, non_blocking=True)
            y_batch = batch[1].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                X_batch = batch[0].to(device, non_blocking=True)
                y_batch = batch[1].to(device, non_blocking=True)
                
                with autocast():
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {elapsed:.1f}s')
    
    print("-" * 70)
    print(f"训练完成! 总用时: {time.time() - start_time:.1f}s")
    
    return train_losses, val_losses


# ============================================
# 8. 预测和测试函数
# ============================================

def predict_full_sequence(model, ground_motion, mean, std, device='cuda'):
    model.eval()
    model = model.to(device)
    
    seq_len = len(ground_motion)
    if seq_len > 3000:
        ground_motion = ground_motion[:3000]
        seq_len = 3000
    elif seq_len < 3000:
        ground_motion = np.pad(ground_motion, (0, 3000 - seq_len), 'constant')
        seq_len = 3000
    
    x = torch.FloatTensor(ground_motion).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        y_norm = model(x).cpu().numpy().squeeze()
    y = y_norm * std + mean
    return y[:seq_len]  # 返回原始长度


def test_random_samples(model, val_motions, val_responses, global_mean, global_std, 
                        device='cuda', num_samples=10):
    model.eval()
    model = model.to(device)
    
    n_samples = min(num_samples, len(val_motions))
    indices = random.sample(range(len(val_motions)), n_samples)
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    r2_list = []
    for i, idx in enumerate(indices):
        motion = val_motions[idx]
        true = val_responses[idx]
        pred = predict_full_sequence(model, motion, global_mean, global_std, device)
        
        # 确保长度一致
        min_len = min(len(true), len(pred))
        true = true[:min_len]
        pred = pred[:min_len]
        motion = motion[:min_len]
        
        t = np.arange(min_len) * TARGET_DT
        
        axes[i, 0].plot(t, motion, linewidth=0.8)
        axes[i, 0].set_ylabel('Accel (g)')
        axes[i, 0].set_title(f'Input {i+1}')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(t, true, linewidth=1.5, label='True')
        axes[i, 1].plot(t, pred, linewidth=1.5, linestyle='--', label='Pred')
        axes[i, 1].set_ylabel('Disp (mm)')
        axes[i, 1].set_title(f'Response {i+1}')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].plot(t, pred - true, color='red', linewidth=0.8)
        axes[i, 2].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[i, 2].set_ylabel('Error (mm)')
        axes[i, 2].set_title(f'Error {i+1}')
        axes[i, 2].grid(True, alpha=0.3)
        
        r2 = 1 - np.sum((pred - true)**2) / np.sum((true - np.mean(true))**2)
        r2_list.append(r2)
    
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    axes[-1, 2].set_xlabel('Time (s)')
    avg_r2 = np.mean(r2_list)
    plt.suptitle(f'Random {n_samples} Samples Prediction Comparison (Avg R²={avg_r2:.4f})')
    plt.tight_layout()
    plt.savefig('random_test_comparison.png', dpi=150)
    print(f"✓ 随机测试对比图已保存: random_test_comparison.png")
    print(f"随机 {n_samples} 条平均 R²: {avg_r2:.4f}")
    plt.show()
    
    return r2_list


# ============================================
# 9. 主程序
# ============================================

def main():
    print("="*70)
    print(f"多层框架 ({NUM_STORIES}层) Seq2Seq 训练/测试")
    print(f"数据源: {get_data_source()}")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        device = 'cpu'
        print("⚠ 使用CPU")
    
    print(f"\n框架参数:")
    print(f"  楼层数: {NUM_STORIES}")
    print(f"  层高: {STORY_HEIGHT} m, 总高: {NUM_STORIES*STORY_HEIGHT} m")
    print(f"  每层质量: {MASS_PER_STORY} kg")
    print(f"  截面: 下->上 {[f'{s*1000}mm' for s in SECTIONS]}")
    if not USE_SYNTHETIC:
        print(f"  天然波目标PGA: {TARGET_PGA}g ({TARGET_PGA*9810:.1f} cm/s²)")
        print(f"  天然波截取时长: {TARGET_DURATION} s")
    print("="*70)

    # ===== 数据加载 / 生成 =====
    if MODE == 'train':
        print("\n[1] 生成训练数据...")
        try:
            motions, responses = generate_and_cache_data(NUM_SYNTHETIC, force_regen=FORCE_REGEN,
                                                         num_stories=NUM_STORIES, sections=SECTIONS)
        except Exception as e:
            print(f"数据生成失败: {e}")
            return
    else:  # test 模式
        print("\n[1] 加载缓存数据...")
        if not os.path.exists(CACHE_FILE):
            print(f"错误：缓存文件 {CACHE_FILE} 不存在，请先运行训练模式生成数据。")
            return
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        motions, responses = data['motions'], data['responses']
        print(f"✓ 加载缓存: {CACHE_FILE}, 包含 {len(motions)} 条样本")
    
    if motions is None or responses is None:
        print("错误：数据加载失败")
        return
    
    # 确保是 numpy 数组
    if not isinstance(motions, np.ndarray):
        motions = np.array(motions)
    if not isinstance(responses, np.ndarray):
        responses = np.array(responses)
    
    print(f"  数据形状: motions {motions.shape}, responses {responses.shape}")
    
    if np.isnan(responses).any() or np.isinf(responses).any():
        print("错误：响应数据包含NaN或Inf")
        return
    
    # 2. 计算全局统计量
    global_mean = np.mean(responses)
    global_std = np.std(responses) + 1e-8
    print(f"  全局均值: {global_mean:.4f} mm")
    print(f"  全局标准差: {global_std:.4f} mm")
    print(f"  位移范围: {np.min(responses):.2f} ~ {np.max(responses):.2f} mm")
    
    # 3. 划分训练/验证集
    n = len(motions)
    indices = np.random.permutation(n)
    train_n = int(0.8 * n)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]
    
    train_motions = motions[train_idx]
    train_responses = responses[train_idx]
    val_motions = motions[val_idx]
    val_responses = responses[val_idx]
    
    print(f"  训练集: {len(train_motions)} 条")
    print(f"  验证集: {len(val_motions)} 条")
    
    # 4. 创建Dataset
    print("\n[2] 创建数据集...")
    train_dataset = Seq2SeqDataset(train_motions, train_responses, global_mean, global_std)
    val_dataset = Seq2SeqDataset(val_motions, val_responses, global_mean, global_std)
    
    if len(train_dataset) == 0:
        print("错误：训练数据集为空")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"  训练批次数: {len(train_loader):,}")
    
    # 5. 创建模型
    print("\n[3] 创建模型...")
    model = Seq2SeqLSTM(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,
        seq_len=3000,
        dropout=DROPOUT
    )
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. 训练 (仅 train 模式)
    if MODE == 'train':
        print("\n" + "="*70)
        print("[训练模式]")
        print("="*70)
        
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        print("预热GPU...")
        model = model.to(device)
        for batch in train_loader:
            X_batch = batch[0].to(device)
            with torch.no_grad():
                _ = model(X_batch)
            break
        torch.cuda.synchronize()
        print("GPU预热完成")
        print("-" * 70)
        
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, 
            epochs=EPOCHS, lr=LEARNING_RATE, device=device
        )
        
        torch.save(model.state_dict(), MODEL_FILE)
        print(f"\n✓ 模型已保存: {MODEL_FILE}")
        
        # 训练曲线
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png', dpi=150)
        plt.show()
    
    # 7. 测试 (train 和 test 模式都执行)
    print("\n" + "="*70)
    print("[测试] 随机10条样本预测对比")
    print("="*70)
    
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model = model.to(device)
        print(f"✓ 加载模型: {MODEL_FILE}")
    else:
        print(f"✗ 模型不存在，请先训练")
        return
    
    test_random_samples(model, val_motions, val_responses, global_mean, global_std, device, num_samples=10)
    
    print("\n完成!")


if __name__ == "__main__":
    main()
