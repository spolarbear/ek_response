# ============================================
# rnn-2d-params.py
# 框架结构 + 参数融合 Seq2Seq
# 支持: 中断恢复 + TensorBoard
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import pickle
import time
import warnings
import random
import glob
import json
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')

# ===== 设置中文字体 =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

# ===== 大规模数据生成 =====
NUM_SAMPLES = 10000     # 生成 10 万条样本
NUM_PARAM_COMBOS = 500
TARGET_DURATION = 20.0
TARGET_DT = 0.01
TARGET_PGA = 0.035

# ===== 缓存控制 =====
CACHE_FILE = 'frame_params_data_100k.pkl'
FORCE_REGEN = False      # True: 重新生成, False: 使用缓存

# ===== 框架结构参数 =====
DZB_FOLDER = './dzb'
PARAM_RANGES = {
    'num_stories': [3, 8],
    'num_bays': [1, 3],
    'bay_width': [4.0, 8.0],
    'story_height': [3.0, 4.5],
    'mass_per_node': [20000, 60000],
    'damping_ratio': [0.03, 0.08]
}

# ===== 混凝土材料 =====
E = 3.25e10
COL_SECTIONS = [0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3]
BEAM_SECTION = (0.3, 0.6)

# ===== 模型参数 =====
MODEL_FILE = 'frame_params_lstm_100k.pth'
CHECKPOINT_FILE = 'checkpoint_lstm.pth'  # 检查点文件
EPOCHS = 100
BATCH_SIZE = 256
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.3
LEARNING_RATE = 0.001
PARAM_DIM = 6

# ===== 中断恢复配置 =====
RESUME_TRAINING = True   # True: 从断点恢复, False: 重新开始训练

# ===== DataLoader 优化 =====
NUM_WORKERS = 8
PIN_MEMORY = True
PREFETCH_FACTOR = 4

# ===== TensorBoard =====
TENSORBOARD_DIR = './runs/frame_lstm'  # TensorBoard 日志目录
USE_TENSORBOARD = True

# ============================================
# ============================================


# ============================================
# 1. 天然波加载
# ============================================

def parse_earthquake_file(file_path):
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
    if not os.path.exists(folder_path):
        print(f"警告：文件夹 {folder_path} 不存在")
        return np.array([]), 0
    
    patterns = ['*.txt', '*.dat', '*.csv']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(folder_path, pattern)))
    files = list(set(files))
    files = [f for f in files if os.path.isfile(f)]
    
    if len(files) == 0:
        print(f"警告：文件夹 {folder_path} 中没有找到地震动文件")
        return np.array([]), 0
    
    print(f"找到 {len(files)} 个地震动文件")
    
    all_motions = []
    target_steps = int(target_duration / target_dt)
    
    for file_path in files:
        try:
            motion_orig, dt_orig = parse_earthquake_file(file_path)
            
            if len(motion_orig) < 10:
                continue
            
            n_orig = len(motion_orig)
            t_orig = np.arange(n_orig) * dt_orig
            
            if t_orig[-1] >= target_duration:
                t_target = np.arange(target_steps) * target_dt
                f = interp1d(t_orig, motion_orig, kind='linear', fill_value='extrapolate')
                motion_resampled = f(t_target)
            else:
                n_avail = int(t_orig[-1] / target_dt)
                motion_resampled = motion_orig[:n_avail]
                if len(motion_resampled) < target_steps:
                    motion_resampled = np.pad(motion_resampled, (0, target_steps - len(motion_resampled)), 'constant')
            
            pga = np.max(np.abs(motion_resampled))
            if pga > 1e-10:
                motion_scaled = motion_resampled / pga * target_pga
            else:
                motion_scaled = motion_resampled
            
            all_motions.append(motion_scaled)
            
        except Exception as e:
            print(f"  加载失败: {file_path}, {e}")
    
    if len(all_motions) == 0:
        return np.array([]), 0
    
    motions_array = np.array(all_motions)
    print(f"成功加载 {len(all_motions)} 条天然波")
    return motions_array, len(all_motions)


# ============================================
# 2. 框架结构 OpenSees 仿真
# ============================================

def run_frame_analysis(ground_motion, dt, num_stories, num_bays, 
                       bay_width, story_height, mass_per_node, damping_ratio):
    n_steps = len(ground_motion)
    
    num_stories = int(max(2, min(num_stories, 8)))
    num_bays = int(max(1, min(num_bays, 3)))
    bay_width = float(max(3.0, min(bay_width, 10.0)))
    story_height = float(max(2.5, min(story_height, 5.0)))
    mass_per_node = float(max(10000, min(mass_per_node, 80000)))
    damping_ratio = float(max(0.01, min(damping_ratio, 0.12)))
    
    col_sections = COL_SECTIONS[:num_stories]
    if len(col_sections) < num_stories:
        col_sections = col_sections + [col_sections[-1]] * (num_stories - len(col_sections))
    
    beam_b, beam_h = BEAM_SECTION
    
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    
    node_tags = {}
    node_id = 0
    for floor in range(num_stories + 1):
        for bay in range(num_bays + 1):
            node_id += 1
            x = bay * bay_width
            y = floor * story_height
            ops.node(node_id, x, y)
            node_tags[(floor, bay)] = node_id
    
    for bay in range(num_bays + 1):
        ops.fix(node_tags[(0, bay)], 1, 1, 1)
    
    ops.uniaxialMaterial('Elastic', 1, E)
    ops.geomTransf('Linear', 1)
    
    elem_id = 0
    for floor in range(1, num_stories + 1):
        col_idx = floor - 1
        col_size = col_sections[col_idx]
        A_col = col_size ** 2
        I_col = col_size ** 4 / 12
        
        for bay in range(num_bays + 1):
            elem_id += 1
            node_bottom = node_tags[(floor - 1, bay)]
            node_top = node_tags[(floor, bay)]
            ops.element('elasticBeamColumn', elem_id, node_bottom, node_top, 
                       A_col, E, I_col, 1)
    
    for floor in range(1, num_stories + 1):
        A_beam = beam_b * beam_h
        I_beam = beam_b * beam_h ** 3 / 12
        
        for bay in range(num_bays):
            elem_id += 1
            node_left = node_tags[(floor, bay)]
            node_right = node_tags[(floor, bay + 1)]
            ops.element('elasticBeamColumn', elem_id, node_left, node_right,
                       A_beam, E, I_beam, 1)
    
    for floor in range(1, num_stories + 1):
        for bay in range(num_bays + 1):
            ops.mass(node_tags[(floor, bay)], mass_per_node, 0.0, 0.0)
    
    col_size_0 = col_sections[0]
    I_col_0 = col_size_0 ** 4 / 12
    k_col_est = 12 * E * I_col_0 / (story_height ** 3)
    k_eff = (num_bays + 1) * k_col_est
    m_eff = (num_bays + 1) * mass_per_node
    omega1 = np.sqrt(k_eff / m_eff)
    omega2 = omega1 * 2.5
    
    if omega1 < 0.1:
        omega1 = 1.0
        omega2 = 2.5
    
    alpha_m = 2 * damping_ratio * omega1 * omega2 / (omega1 + omega2)
    beta_k = 2 * damping_ratio / (omega1 + omega2)
    ops.rayleigh(alpha_m, beta_k, 0.0, 0.0)
    
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
    
    top_node = node_tags[(num_stories, 0)]
    top_disp_m = []
    
    for step in range(n_steps):
        ok = ops.analyze(1, dt)
        if ok != 0:
            if len(top_disp_m) > 0:
                return np.pad(np.array(top_disp_m), (0, n_steps - len(top_disp_m)), 'edge')
            return np.zeros(n_steps)
        d = ops.nodeDisp(top_node, 1)
        top_disp_m.append(d)
    
    return np.array(top_disp_m) * 1000.0


# ============================================
# 3. 大规模数据生成
# ============================================

def generate_data_with_params(num_samples=100000, force_regen=False):
    if os.path.exists(CACHE_FILE) and not force_regen:
        print(f"✓ 加载缓存数据: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        return data['motions'], data['responses'], data['params']
    
    print(f"="*70)
    print(f"开始大规模数据生成: {num_samples:,} 条样本")
    print(f"="*70)
    
    motions, num_waves = load_earthquake_files(DZB_FOLDER, TARGET_DURATION, TARGET_PGA, TARGET_DT)
    
    if len(motions) == 0:
        print("未找到天然波，使用合成波...")
        motions = generate_batch_synthetic(236)
        num_waves = len(motions)
    
    print(f"共有 {num_waves} 条地震波")
    
    num_params = max(NUM_PARAM_COMBOS, 1000)
    print(f"生成 {num_params} 组结构参数...")
    
    all_params_pool = []
    for i in range(num_params):
        num_stories = np.random.randint(PARAM_RANGES['num_stories'][0], 
                                        PARAM_RANGES['num_stories'][1] + 1)
        num_bays = np.random.randint(PARAM_RANGES['num_bays'][0], 
                                     PARAM_RANGES['num_bays'][1] + 1)
        bay_width = np.random.uniform(PARAM_RANGES['bay_width'][0], 
                                      PARAM_RANGES['bay_width'][1])
        story_height = np.random.uniform(PARAM_RANGES['story_height'][0], 
                                         PARAM_RANGES['story_height'][1])
        mass_per_node = np.random.uniform(PARAM_RANGES['mass_per_node'][0], 
                                          PARAM_RANGES['mass_per_node'][1])
        damping = np.random.uniform(PARAM_RANGES['damping_ratio'][0], 
                                    PARAM_RANGES['damping_ratio'][1])
        
        all_params_pool.append([float(num_stories), float(num_bays), 
                               float(bay_width), float(story_height),
                               float(mass_per_node), float(damping)])
    
    all_params_pool = np.array(all_params_pool, dtype=np.float32)
    print(f"参数池大小: {len(all_params_pool)} 组")
    
    print(f"交叉组合: {num_waves} 条波 × {len(all_params_pool)} 组参数")
    print(f"采样 {num_samples:,} 条...")
    
    wave_indices = np.random.randint(0, num_waves, num_samples)
    param_indices = np.random.randint(0, len(all_params_pool), num_samples)
    
    batch_size_sim = 200
    num_batches = int(np.ceil(num_samples / batch_size_sim))
    
    all_responses = []
    all_motions_selected = []
    all_params_selected = []
    failed_count = 0
    
    print(f"运行 OpenSees 仿真 (分 {num_batches} 批)...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size_sim
        end_idx = min(start_idx + batch_size_sim, num_samples)
        batch_size = end_idx - start_idx
        
        batch_wave_indices = wave_indices[start_idx:end_idx]
        batch_param_indices = param_indices[start_idx:end_idx]
        
        batch_motions = []
        batch_params = []
        batch_responses = []
        
        for i in range(batch_size):
            wi = batch_wave_indices[i]
            pi = batch_param_indices[i]
            batch_motions.append(motions[wi])
            batch_params.append(all_params_pool[pi])
        
        if HAS_TQDM:
            iterator = tqdm(range(batch_size), desc=f"批次 {batch_idx+1}/{num_batches}")
        else:
            iterator = range(batch_size)
        
        for i in iterator:
            motion = batch_motions[i]
            p = batch_params[i]
            num_stories = int(p[0])
            num_bays = int(p[1])
            bay_width = p[2]
            story_height = p[3]
            mass_per_node = p[4]
            damping = p[5]
            
            try:
                resp = run_frame_analysis(motion, TARGET_DT, num_stories, num_bays,
                                          bay_width, story_height, mass_per_node, damping)
                if np.isnan(resp).any() or np.isinf(resp).any() or np.max(np.abs(resp)) > 500:
                    resp = np.zeros_like(resp)
                    failed_count += 1
                batch_responses.append(resp)
            except Exception as e:
                batch_responses.append(np.zeros(len(motion)))
                failed_count += 1
        
        all_motions_selected.extend(batch_motions)
        all_params_selected.extend(batch_params)
        all_responses.extend(batch_responses)
        
        total_done = end_idx
        print(f"  已完成 {total_done}/{num_samples:,}, 失败: {failed_count}")
    
    motions_selected = np.array(all_motions_selected)
    params_selected = np.array(all_params_selected, dtype=np.float32)
    responses = np.array(all_responses)
    
    mean_disp = np.mean(responses)
    std_disp = np.std(responses)
    max_disp = np.max(np.abs(responses))
    print(f"\n统计结果:")
    print(f"  样本总数: {len(responses):,}")
    print(f"  位移均值: {mean_disp:.4f} mm")
    print(f"  位移标准差: {std_disp:.4f} mm")
    print(f"  最大绝对位移: {max_disp:.4f} mm")
    print(f"  失败次数: {failed_count}/{num_samples}")
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'motions': motions_selected, 'responses': responses, 'params': params_selected}, f)
    
    cache_size = os.path.getsize(CACHE_FILE) / (1024**3)
    print(f"✓ 缓存已保存: {CACHE_FILE} (大小: {cache_size:.2f} GB)")
    
    return motions_selected, responses, params_selected


def generate_batch_synthetic(num_samples):
    motions = []
    for _ in range(num_samples):
        motion = generate_synthetic_ground_motion(TARGET_DURATION, TARGET_DT, TARGET_PGA)
        motions.append(motion)
    return np.array(motions)


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


# ============================================
# 4. Dataset
# ============================================

class Seq2SeqDataset(Dataset):
    def __init__(self, motions, responses, params, mean=None, std=None):
        self.seq_len = min(motions.shape[1], responses.shape[1], 2000)
        self.motions = motions[:, :self.seq_len].astype(np.float32)
        self.responses = responses[:, :self.seq_len].astype(np.float32)
        self.params = params.astype(np.float32)
        
        if mean is None or std is None:
            self.mean = np.mean(self.responses)
            self.std = np.std(self.responses) + 1e-8
        else:
            self.mean = mean
            self.std = std
        
        self.responses_norm = (self.responses - self.mean) / self.std
        
    def __len__(self):
        return len(self.motions)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.motions[idx]).unsqueeze(1)
        p = torch.FloatTensor(self.params[idx])
        y = torch.FloatTensor(self.responses_norm[idx]).unsqueeze(1)
        return x, p, y


# ============================================
# 5. 模型
# ============================================

class Seq2SeqWithParams(nn.Module):
    def __init__(self, param_dim=6, hidden_size=128, num_layers=3, 
                 output_size=1, seq_len=2000, dropout=0.3):
        super(Seq2SeqWithParams, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_size),
            nn.Tanh()
        )
        
        self.encoder = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder_cell = nn.LSTMCell(1 + hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_accel, x_params, teacher_force=None):
        batch_size, seq_len = x_accel.size(0), x_accel.size(1)
        device = x_accel.device
        
        param_emb = self.param_encoder(x_params)
        _, (h0, c0) = self.encoder(x_accel)
        
        h = h0[-1]
        c = c0[-1]
        
        outputs = []
        
        for t in range(seq_len):
            accel_t = x_accel[:, t, :]
            decoder_input = torch.cat([accel_t, param_emb], dim=1)
            
            h, c = self.decoder_cell(decoder_input, (h, c))
            disp_t = self.fc_out(self.dropout(h))
            outputs.append(disp_t)
        
        pred = torch.stack(outputs, dim=1)
        return pred


# ============================================
# 6. 训练函数 (支持中断恢复 + TensorBoard)
# ============================================

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                    model_file=MODEL_FILE, checkpoint_file=CHECKPOINT_FILE):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, checkpoint_file)
    torch.save(model.state_dict(), model_file)
    print(f"  ✓ 检查点已保存 (Epoch {epoch})")


def load_checkpoint(model, optimizer, checkpoint_file=CHECKPOINT_FILE):
    """加载检查点"""
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✓ 从断点恢复: Epoch {checkpoint['epoch']}")
        print(f"  训练损失: {checkpoint['train_loss']:.6f}")
        print(f"  验证损失: {checkpoint['val_loss']:.6f}")
        return start_epoch, checkpoint['train_loss'], checkpoint['val_loss']
    else:
        print("  未找到检查点，从头开始训练")
        return 0, None, None


def train_model(model, train_loader, val_loader, epochs=60, lr=0.001, 
                device='cuda', resume=True, writer=None):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    train_losses = []
    val_losses = []
    
    # 尝试从断点恢复
    start_epoch = 0
    if resume:
        start_epoch, last_train_loss, last_val_loss = load_checkpoint(model, optimizer)
        if start_epoch > 0:
            train_losses = [last_train_loss] if last_train_loss is not None else []
            val_losses = [last_val_loss] if last_val_loss is not None else []
    
    print(f"\n开始训练 {epochs} 个 epochs...")
    print(f"起始 epoch: {start_epoch + 1}")
    print(f"训练批次数: {len(train_loader):,}")
    print(f"批次大小: {train_loader.batch_size}")
    print(f"数据加载线程: {train_loader.num_workers}")
    print("-" * 70)
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        
        for batch in train_iter:
            X_batch = batch[0].to(device, non_blocking=True)
            P_batch = batch[1].to(device, non_blocking=True)
            Y_batch = batch[2].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                y_pred = model(X_batch, P_batch)
                loss = criterion(y_pred, Y_batch)
            
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
                X_batch = batch[0].to(device)
                P_batch = batch[1].to(device)
                Y_batch = batch[2].to(device)
                
                with autocast():
                    y_pred = model(X_batch, P_batch)
                    loss = criterion(y_pred, Y_batch)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # --- 保存最佳模型 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_FILE.replace('.pth', '_best.pth'))
        
        # --- TensorBoard 记录 ---
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
            
            # 记录梯度直方图 (每10个epoch)
            if epoch % 10 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                        writer.add_histogram(f'Weights/{name}', param.data, epoch)
        
        # --- 打印进度 ---
        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if device == 'cuda' else 0
            print(f'Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | GPU: {gpu_mem:.2f}GB | Time: {elapsed:.1f}s')
        
        # --- 保存检查点 (每5个epoch) ---
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss)
    
    print("-" * 70)
    print(f"训练完成! 总用时: {time.time() - start_time:.1f}s")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return train_losses, val_losses


# ============================================
# 7. TensorBoard 启动
# ============================================

def start_tensorboard(log_dir=TENSORBOARD_DIR, port=6006):
    """启动 TensorBoard"""
    try:
        import subprocess
        import webbrowser
        
        # 检查是否已有 TensorBoard 在运行
        cmd = f"tensorboard --logdir={log_dir} --port={port} --host=127.0.0.1"
        
        # 后台启动
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"\n✓ TensorBoard 已启动!")
        print(f"  URL: http://localhost:{port}")
        print(f"  日志目录: {log_dir}")
        print(f"  在浏览器中打开 http://localhost:{port}")
        
        # 尝试打开浏览器
        time.sleep(2)
        webbrowser.open(f"http://localhost:{port}")
        
        return True
    except Exception as e:
        print(f"⚠ TensorBoard 启动失败: {e}")
        print("  手动启动: tensorboard --logdir=./runs/frame_lstm --port=6006")
        return False


# ============================================
# 8. 预测和测试
# ============================================

def predict_full_sequence(model, ground_motion, params, mean, std, device='cuda'):
    model.eval()
    model = model.to(device)
    
    seq_len = len(ground_motion)
    if seq_len > 2000:
        ground_motion = ground_motion[:2000]
        seq_len = 2000
    elif seq_len < 2000:
        ground_motion = np.pad(ground_motion, (0, 2000 - seq_len), 'constant')
    
    x = torch.FloatTensor(ground_motion).unsqueeze(0).unsqueeze(-1).to(device)
    p = torch.FloatTensor(params).unsqueeze(0).to(device)
    
    with torch.no_grad():
        y_norm = model(x, p).cpu().numpy().squeeze()
    y = y_norm * std + mean
    return y[:seq_len]


def test_random_samples(model, val_motions, val_responses, val_params, 
                        global_mean, global_std, device='cuda', num_samples=10):
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
        params = val_params[idx]
        pred = predict_full_sequence(model, motion, params, global_mean, global_std, device)
        
        min_len = min(len(true), len(pred))
        true = true[:min_len]
        pred = pred[:min_len]
        motion = motion[:min_len]
        
        t = np.arange(min_len) * TARGET_DT
        
        axes[i, 0].plot(t, motion, linewidth=0.8)
        axes[i, 0].set_ylabel('加速度 (g)')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_title(f'输入地震波 {i+1}')
        
        axes[i, 1].plot(t, true, linewidth=1.5, label='真实')
        axes[i, 1].plot(t, pred, linewidth=1.5, linestyle='--', label='预测')
        axes[i, 1].set_ylabel('位移 (mm)')
        axes[i, 1].set_title(f'响应对比 {i+1}')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].plot(t, pred - true, color='red', linewidth=0.8)
        axes[i, 2].axhline(y=0, color='black', linestyle='--')
        axes[i, 2].set_ylabel('误差 (mm)')
        axes[i, 2].set_title(f'误差 {i+1}')
        axes[i, 2].grid(True, alpha=0.3)
        
        info = f"层:{int(params[0])} 跨:{int(params[1])}"
        axes[i, 2].text(0.02, 0.95, info, transform=axes[i, 2].transAxes, fontsize=8)
        
        r2 = 1 - np.sum((pred - true)**2) / np.sum((true - np.mean(true))**2)
        r2_list.append(r2)
    
    axes[-1, 0].set_xlabel('时间 (s)')
    axes[-1, 1].set_xlabel('时间 (s)')
    axes[-1, 2].set_xlabel('时间 (s)')
    avg_r2 = np.mean(r2_list)
    plt.suptitle(f'随机 {n_samples} 条样本预测对比 (平均 R²={avg_r2:.4f})')
    plt.tight_layout()
    plt.savefig('frame_params_test_comparison.png', dpi=150)
    print(f"✓ 测试图已保存")
    print(f"平均 R²: {avg_r2:.4f}")
    plt.show()
    
    return r2_list


# ============================================
# 9. 主程序
# ============================================

def main():
    print("="*70)
    print("大规模框架结构 Seq2Seq 训练")
    print("交叉组合: 地震波 × 结构参数")
    print(f"目标样本数: {NUM_SAMPLES:,}")
    print("="*70)
    
    # ===== TensorBoard =====
    writer = None
    if USE_TENSORBOARD and MODE == 'train':
        # 创建日志目录
        os.makedirs(TENSORBOARD_DIR, exist_ok=True)
        writer = SummaryWriter(TENSORBOARD_DIR)
        print(f"✓ TensorBoard 日志目录: {TENSORBOARD_DIR}")
        
        # 记录超参数
        writer.add_text('Config', f"""
        NUM_SAMPLES: {NUM_SAMPLES}
        EPOCHS: {EPOCHS}
        BATCH_SIZE: {BATCH_SIZE}
        HIDDEN_SIZE: {HIDDEN_SIZE}
        NUM_LAYERS: {NUM_LAYERS}
        LEARNING_RATE: {LEARNING_RATE}
        DROPOUT: {DROPOUT}
        """)
        
        # 后台启动 TensorBoard
        start_tensorboard(TENSORBOARD_DIR, port=6006)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = 'cuda'
    else:
        device = 'cpu'
        print("⚠ 使用CPU")
    
    print(f"\n结构参数范围:")
    print(f"  楼层数: {PARAM_RANGES['num_stories']}")
    print(f"  跨数: {PARAM_RANGES['num_bays']}")
    print(f"  跨宽: {PARAM_RANGES['bay_width']} m")
    print(f"  层高: {PARAM_RANGES['story_height']} m")
    print(f"  节点质量: {PARAM_RANGES['mass_per_node']} kg")
    print(f"  阻尼比: {PARAM_RANGES['damping_ratio']}")
    print(f"\n模型配置:")
    print(f"  隐藏层大小: {HIDDEN_SIZE}")
    print(f"  LSTM层数: {NUM_LAYERS}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  数据加载线程: {NUM_WORKERS}")
    print(f"\n中断恢复:")
    print(f"  RESUME_TRAINING = {RESUME_TRAINING}")
    print("="*70)

    # ===== 数据生成 =====
    if MODE == 'train':
        print("\n[1] 生成大规模训练数据 (交叉组合)...")
        print(f"  目标: {NUM_SAMPLES:,} 条样本")
        try:
            motions, responses, params = generate_data_with_params(NUM_SAMPLES, force_regen=FORCE_REGEN)
        except Exception as e:
            print(f"数据生成失败: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("\n[1] 加载缓存数据...")
        if not os.path.exists(CACHE_FILE):
            print(f"错误：缓存文件不存在")
            print(f"提示：设置 FORCE_REGEN = True 重新生成数据")
            return
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        motions, responses, params = data['motions'], data['responses'], data['params']
        print(f"✓ 加载缓存: {len(motions):,} 条")
    
    if motions is None or len(motions) == 0:
        print("错误：数据加载失败")
        return
    
    print(f"  数据形状: motions {motions.shape}, responses {responses.shape}")
    
    global_mean = np.mean(responses)
    global_std = np.std(responses) + 1e-8
    print(f"  全局均值: {global_mean:.4f} mm")
    
    n = len(motions)
    indices = np.random.permutation(n)
    train_n = int(0.8 * n)
    
    train_motions = motions[indices[:train_n]]
    train_responses = responses[indices[:train_n]]
    train_params = params[indices[:train_n]]
    val_motions = motions[indices[train_n:]]
    val_responses = responses[indices[train_n:]]
    val_params = params[indices[train_n:]]
    
    print(f"  训练集: {len(train_motions):,} 条")
    print(f"  验证集: {len(val_motions):,} 条")
    
    print("\n[2] 创建数据集...")
    train_dataset = Seq2SeqDataset(train_motions, train_responses, train_params, global_mean, global_std)
    val_dataset = Seq2SeqDataset(val_motions, val_responses, val_params, global_mean, global_std)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR
    )
    
    print(f"  训练批次数: {len(train_loader):,}")
    
    print("\n[3] 创建模型...")
    model = Seq2SeqWithParams(
        param_dim=PARAM_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,
        seq_len=2000,
        dropout=DROPOUT
    )
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    if MODE == 'train':
        print("\n" + "="*70)
        print("[训练模式]")
        print("="*70)
        
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        model = model.to(device)
        print("预热GPU...")
        for batch in train_loader:
            X_batch = batch[0].to(device)
            P_batch = batch[1].to(device)
            with torch.no_grad():
                _ = model(X_batch, P_batch)
            break
        torch.cuda.synchronize()
        print("GPU预热完成")
        
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, 
            epochs=EPOCHS, lr=LEARNING_RATE, device=device,
            resume=RESUME_TRAINING, writer=writer
        )
        
        torch.save(model.state_dict(), MODEL_FILE)
        print(f"\n✓ 模型已保存: {MODEL_FILE}")
        
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('训练历史')
        plt.savefig('frame_training_history.png', dpi=150)
        plt.show()
    
    print("\n" + "="*70)
    print("[测试] 随机10条")
    print("="*70)
    
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model = model.to(device)
        print(f"✓ 加载模型")
    else:
        print(f"✗ 模型不存在")
        return
    
    test_random_samples(model, val_motions, val_responses, val_params,
                        global_mean, global_std, device, num_samples=10)
    
    # 关闭 TensorBoard
    if writer is not None:
        writer.close()
        print("✓ TensorBoard 已关闭")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
