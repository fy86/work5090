# -*- coding: utf-8 -*-
"""
infer_stage.py
Quant-GPT 多阶段自回归大模型 —— 统一推理与胜率评估引擎
功能:
  1. 兼容 Stage 1 ~ Stage 6 任意阶段产出的 Checkpoint 权重与 D1(322)/D2(642)/D3(778) 验证集/测试集
  2. 以序列前 N-1 列作为输入预测明日 H (1~63 分箱概率分布)，最后一列作为真实明日 H (true_h)
  3. 计算 >= 指定分箱 (如 token 38 对应 >=2%) 的累计概率和、期望分箱、最高概率分箱
  4. 与元数据 id0, id1, id2 严格对齐并导出为 CSV 文件 (支持导出完整 1~63 概率表)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

# 开启 Ampere+ 架构 TF32 矩阵乘法加速
torch.set_float32_matmul_precision('high')


# ==============================================================================
# 1. 全局词表宪法与推理配置 (与 train_stage.py 严格保持一致)
# ==============================================================================
class VocabSpec:
    GROUP_SIZE: int = 64
    NUM_EMB_GROUPS: int = 10
    VOCAB_SIZE: int = 640
    NUM_HEAD_GROUPS: int = 4

    SEP_STK_ID: int = 0            # 个股日线周期定界符 & 预测明日H触发符
    SEP_IDX_ID: int = 256          # 指数日线周期定界符
    SEP_M15_ID: int = 512          # 15分钟线单日定界符

    GT_2PCT_MIN_BIN: int = 39      # 涨幅 >= 2% 起始分箱 (1~63 坐标系)  🔴
    LIMIT_UP_MIN_BIN: int = 60     # 今日收盘涨停起始分箱 (1~63 坐标系，后处理过滤参考)
    MAX_SLOTS: int = 64


class InferConfig:
    # === [A] 模型权重与数据集路径配置 (可自由切换 Stage 1 / 2 / 3~6) ===
    base_dir: str = r"../tmp"

    # 对应关系参考:
    #   Stage 1 模型 -> ckpt_stage1.pth + d1_stk_322_val.parquet
    #   Stage 2 模型 -> ckpt_stage2.pth + d2_idx_642_val.parquet
    #   Stage 3+模型 -> ckpt_stage3.pth + d3_m15_778_val.parquet
    ckpt_file: str = "pth.pth" # "ckpt_stage1.pth"  #   🔴🔴
    val_parquet: str = "ds.parquet"
    output_csv: str = "pred.csv"

    # === [B] 目标分箱概率和配置 ===
    target_min_bin: int = 33                 # 主目标分箱: 计算 P(H >= 38) 即涨幅 >= 2% 的概率和 🔴 39,38
    extra_min_bins: list = [36, 39]          # 额外顺带计算的阈值分箱列表 (如 >=3%, >=5% 等，不需要可设为空 [])
    save_full_prob_table: bool = True        # True: 在 CSV 中保存 p_bin_01 ~ p_bin_63 完整 63 列概率表
    float_precision: int = 6                 # CSV 浮点数保留小数位

    # === [C] 模型骨干固定超参数 (严禁改动，自动从 ckpt 同步 rope_cfg) ===
    max_seq_len: int = 1024
    n_layer: int = 16
    n_head: int = 8
    n_embd: int = 512
    use_qk_norm: bool = True
    dropout: float = 0.0                     # 推理阶段固定为 0.0

    # 默认 RoPE 配置 (加载 ckpt 时会自动覆盖为训练时的实际配置)
    rope_mode: str = '2d'
    rope_day_dims: int = 32
    rope_day_base: float = 1000.0
    rope_slot_base: float = 100.0

    # === [D] 推理硬件参数 ===
    batch_size: int = 256
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = InferConfig()


# ==============================================================================
# 2. 模型骨干定义 (与 train_stage.py 100% 架构对齐，推理固定走 forward_h_only)
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def precompute_rope(max_seq_len: int, head_dim: int):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)


def precompute_rope_table(max_pos: int, rot_dim: int, base: float):
    inv_freq = 1.0 / (base ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def build_2d_positions(idx: torch.Tensor, max_day: int):
    G = VocabSpec.GROUP_SIZE
    g = idx // G
    is_sep = (idx % G == 0)

    day = (idx == VocabSpec.SEP_STK_ID).long().cumsum(dim=1) - 1
    day = day.clamp(0, max_day)

    slot = torch.where(
        g < 4, torch.where(is_sep, 0, g + 1),
        torch.where(
            g < 8, torch.where(is_sep, 5, g + 2),
            torch.where(g == 9, torch.where(is_sep, 27, 28), 0)
        )
    )

    is_m15 = (g == 8)
    c = is_m15.long().cumsum(dim=1)
    c_at_sep = torch.where(idx == VocabSpec.SEP_M15_ID, c, torch.zeros_like(c))
    k = c - torch.cummax(c_at_sep, dim=1).values
    slot = torch.where(is_m15, 10 + k, slot)

    return day, slot.clamp(0, VocabSpec.MAX_SLOTS - 1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, split_dim: int = 0) -> torch.Tensor:
    T = x.shape[2]
    if split_dim > 0:
        x_rot = torch.cat((_rotate_half(x[..., :split_dim]), _rotate_half(x[..., split_dim:])), dim=-1)
    else:
        x_rot = _rotate_half(x)
    cos_t = cos[:, :, :T, :].to(x.dtype)
    sin_t = sin[:, :, :T, :].to(x.dtype)
    return (x * cos_t) + (x_rot * sin_t)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: InferConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_qk_norm = config.use_qk_norm
        self.rope_split = config.rope_day_dims if config.rope_mode == '2d' else 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin, self.rope_split)
        k = apply_rope(k, cos, sin, self.rope_split)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, config: InferConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        hidden_dim = int(config.n_embd * 8 / 3)
        hidden_dim = 64 * ((hidden_dim + 64 - 1) // 64)
        self.mlp = SwiGLU(config.n_embd, hidden_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class PartitionedEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.groups = nn.ModuleList([
            nn.Embedding(VocabSpec.GROUP_SIZE, n_embd)
            for _ in range(VocabSpec.NUM_EMB_GROUPS)
        ])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat([g.weight for g in self.groups], dim=0)
        return F.embedding(idx, full_weight)


class PartitionedLMHead(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.groups = nn.ModuleList([
            nn.Linear(n_embd, VocabSpec.GROUP_SIZE, bias=False)
            for _ in range(VocabSpec.NUM_HEAD_GROUPS)
        ])

    def forward_h_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.groups[0](x)


class ModernGPT(nn.Module):
    def __init__(self, config: InferConfig):
        super().__init__()
        self.config = config
        self.wte = PartitionedEmbedding(config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = PartitionedLMHead(config.n_embd)

        head_dim = config.n_embd // config.n_head
        if config.rope_mode == '2d':
            day_dims = config.rope_day_dims
            slot_dims = head_dim - day_dims
            d_cos, d_sin = precompute_rope_table(config.max_seq_len, day_dims, config.rope_day_base)
            s_cos, s_sin = precompute_rope_table(VocabSpec.MAX_SLOTS, slot_dims, config.rope_slot_base)
            self.register_buffer("rope_day_cos", d_cos, persistent=False)
            self.register_buffer("rope_day_sin", d_sin, persistent=False)
            self.register_buffer("rope_slot_cos", s_cos, persistent=False)
            self.register_buffer("rope_slot_sin", s_sin, persistent=False)
        else:
            cos, sin = precompute_rope(config.max_seq_len, head_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def build_rope(self, idx: torch.Tensor):
        if self.config.rope_mode == '2d':
            day, slot = build_2d_positions(idx, max_day=self.rope_day_cos.size(0) - 1)
            cos = torch.cat([self.rope_day_cos[day], self.rope_slot_cos[slot]], dim=-1).unsqueeze(1)
            sin = torch.cat([self.rope_day_sin[day], self.rope_slot_sin[slot]], dim=-1).unsqueeze(1)
            return cos, sin
        return self.cos, self.sin

    def forward_last_h(self, idx: torch.Tensor) -> torch.Tensor:
        """极速推理通道: 仅对最后一个 Token (<sep_stk>) 过 ln_f 与 Head[0]，返回 [B, 64]"""
        x = self.wte(idx)
        cos, sin = self.build_rope(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        x_last = self.ln_f(x[:, -1, :])            # [B, n_embd]
        return self.lm_head.forward_h_only(x_last)  # [B, 64]


# ==============================================================================
# 3. 辅助解析与推理主引擎
# ==============================================================================
def extract_today_stk_close_bin(seq_tensor: torch.Tensor) -> np.ndarray:
    """
    从任意模态序列 (D1/D2/D3) 中自动提取“今日个股收盘价 C”的分箱值 (1~63 坐标系)：
    个股日线 C 位于 Group 3 (193~255)，序列中最后一个落在该区间的 Token 即为今日个股 C。
    """
    # is_stk_c = (seq_tensor >= 193) & (seq_tensor <= 255)  # 🔴🔴
    is_stk_c = (seq_tensor >= 129) & (seq_tensor <= 191)
    # 构造递增列索引，通过 argmax 找到每行最后一个匹配的列位置
    col_indices = torch.arange(seq_tensor.size(1), device=seq_tensor.device).unsqueeze(0)
    last_c_col = torch.where(is_stk_c, col_indices, torch.full_like(col_indices, -1)).argmax(dim=1)
    raw_c_token = seq_tensor[torch.arange(seq_tensor.size(0), device=seq_tensor.device), last_c_col]
    return (raw_c_token - 128).cpu().numpy().astype(np.int32) # 🔴 192 -> 128


def print_evaluation_report(df_res: pd.DataFrame, target_bin: int, stage_name: str):
    """打印全样本客观指标 vs 剔除今日涨停后的可交易实战指标"""
    prob_col = f"prob_ge_{target_bin}"
    true_h = df_res["true_h"].values
    prob_ge = df_res[prob_col].values
    is_limit_up = df_res["today_c_bin"].values >= VocabSpec.LIMIT_UP_MIN_BIN

    for title, mask in [
        ("全样本客观分布 (含今日涨停)", np.ones(len(df_res), dtype=bool)),
        (f"实战可交易分布 (剔除今日C>={VocabSpec.LIMIT_UP_MIN_BIN}涨停样本)", ~is_limit_up)
    ]:
        sub_true = true_h[mask]
        sub_prob = prob_ge[mask]
        n_total = len(sub_true)
        if n_total == 0:
            continue

        pos_mask = (sub_true >= target_bin)
        base_rate = pos_mask.mean()

        print(f"   ┌─ [{title}] 样本数: {n_total} | 真实 H>={target_bin} 基准率: {base_rate:.4f} ({pos_mask.sum()}条)")
        line_parts = []
        for th in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]:
            pred_pos = (sub_prob >= th)
            cnt = int(pred_pos.sum())
            tp = int((pred_pos & pos_mask).sum())
            prec = tp / max(1, cnt)
            rec = tp / max(1, int(pos_mask.sum()))
            line_parts.append(f"P@{th:.2f}: {prec:.4f} (数:{cnt:4d}, R:{rec:.3f})")
        print("   │  " + " | ".join(line_parts[:3]))
        print("   └─ " + " | ".join(line_parts[3:]))


@torch.no_grad()
def run_stage_inference(cfg: InferConfig):
    ckpt_path = os.path.join(cfg.base_dir, cfg.ckpt_file)
    val_path = os.path.join(cfg.base_dir, cfg.val_parquet)
    out_csv_path = os.path.join(cfg.base_dir, cfg.output_csv)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型权重文件: {ckpt_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"未找到验证集 Parquet 文件: {val_path}")

    print("\n" + "=" * 85)
    print(f">> 启动 Quant-GPT 统一推理引擎 | Device: {cfg.device}")
    print("=" * 85)

    # 1. 加载 Checkpoint 并自动同步 RoPE 位置编码配置
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        stage_name = ckpt.get('stage_name', 'UnknownStage')
        rope_cfg = ckpt.get('rope_cfg', {'rope_mode': '1d'})
        trained_seq_len = ckpt.get('seq_len', 'Unknown')
        best_metric = ckpt.get('best_metric', float('nan'))
        best_metric_name = ckpt.get('best_metric_name', 'metric')
    else:
        state_dict = ckpt
        stage_name = "RawStateDict"
        rope_cfg = {'rope_mode': cfg.rope_mode}
        trained_seq_len = 'Unknown'
        best_metric = float('nan')
        best_metric_name = 'none'

    # 同步 RoPE 参数确保与训练阶段 100% 一致
    cfg.rope_mode = rope_cfg.get('rope_mode', '2d')
    if cfg.rope_mode == '2d':
        cfg.rope_day_dims = rope_cfg.get('rope_day_dims', 32)
        cfg.rope_day_base = rope_cfg.get('rope_day_base', 1000.0)
        cfg.rope_slot_base = rope_cfg.get('rope_slot_base', 100.0)

    print(f"   >> 模型权重已加载: {ckpt_path}")
    print(f"      ├─ 来源阶段: {stage_name} (训练序列长度: {trained_seq_len} | {best_metric_name}: {best_metric:.4f})")
    print(f"      └─ 位置编码: {rope_cfg}")

    model = ModernGPT(cfg).to(cfg.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 2. 读取 Parquet 数据集 (分离 id0, id1, id2 元数据与 t0..tn 序列)
    df = pd.read_parquet(val_path, engine='pyarrow')
    seq_cols = sorted(
        [c for c in df.columns if c.startswith('t') and c[1:].isdigit()],
        key=lambda c: int(c[1:])
    )
    if not seq_cols:
        raise ValueError(f"在 {val_path} 中未找到 t0, t1, ... 序列列！")

    # 严格提取并对齐 id0, id1, id2 (若存在其他非序列列一并保留)
    expected_meta = ['id0', 'id1', 'id2']
    meta_cols = [c for c in expected_meta if c in df.columns] + [
        c for c in df.columns if c not in seq_cols and c not in expected_meta
    ]
    meta_df = df[meta_cols].astype(np.int32).copy()

    sequences = torch.tensor(df[seq_cols].values, dtype=torch.int16)
    num_samples, seq_len = sequences.shape
    print(f"   >> 数据集已加载  : {val_path} | 样本数: {num_samples} | 总列数: {seq_len} (输入前 {seq_len - 1} 列, 最后 1 列为真实明日 H)")

    # 校验倒数第二列是否为明日 H 预测触发符 <sep_stk> (0)
    trigger_tokens = sequences[:, -2]
    if not torch.all(trigger_tokens == VocabSpec.SEP_STK_ID):
        bad_cnt = int((trigger_tokens != VocabSpec.SEP_STK_ID).sum().item())
        print(f"   [警告] 序列倒数第2列 (t{seq_len - 2}) 存在 {bad_cnt} 条样本不是 <sep_stk>=0 触发符！")

    # 3. 批量推理计算明日 H 的 1~63 分箱概率分布
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    enable_amp = ('cuda' in cfg.device)

    all_probs_64 = np.zeros((num_samples, VocabSpec.GROUP_SIZE), dtype=np.float32)
    all_true_h = sequences[:, -1].numpy().astype(np.int32)
    today_c_bin = extract_today_stk_close_bin(sequences[:, :-1])

    total_nll_loss = 0.0

    for start_idx in tqdm(range(0, num_samples, cfg.batch_size), desc=f"[Infer {stage_name}]"):
        end_idx = min(start_idx + cfg.batch_size, num_samples)
        batch_seq = sequences[start_idx:end_idx].to(cfg.device, non_blocking=True).long()

        x_input = batch_seq[:, :-1]   # 前 N-1 列作为输入 (最后一位是 <sep_stk>=0)
        y_true_h = batch_seq[:, -1]   # 最后 1 列作为真实明日 H (1~63)

        with torch.amp.autocast(device_type=('cuda' if 'cuda' in cfg.device else 'cpu'),
                                dtype=amp_dtype, enabled=enable_amp):
            last_h_logits = model.forward_last_h(x_input).float()  # [B, 64]

        # 物理屏蔽 0 号定界符通道，确保 1~63 分箱概率和严格等于 1.0
        last_h_logits[:, 0] = -float('inf')

        # 累计交叉熵 Loss (仅对有效标签 1~63 计算)
        valid_y = (y_true_h >= 1) & (y_true_h < VocabSpec.GROUP_SIZE)
        if valid_y.any():
            batch_loss = F.cross_entropy(last_h_logits[valid_y], y_true_h[valid_y], reduction='sum')
            total_nll_loss += batch_loss.item()

        probs = F.softmax(last_h_logits, dim=-1)  # [B, 64], 其中 probs[:, 0] == 0.0
        all_probs_64[start_idx:end_idx] = probs.cpu().numpy()

    avg_last_h_loss = total_nll_loss / max(1, int(((all_true_h >= 1) & (all_true_h < 64)).sum()))

    # 4. 组装输出表: 元数据 (id0, id1, id2) + 核心统计指标 + (可选) 1~63 完整概率表
    out_dict = {}
    for col in meta_df.columns:
        out_dict[col] = meta_df[col].values

    # 今日收盘分箱与真实明日 H
    out_dict['today_c_bin'] = today_c_bin
    out_dict['true_h'] = all_true_h
    out_dict[f'true_ge_{cfg.target_min_bin}'] = (all_true_h >= cfg.target_min_bin).astype(np.int32)

    # 预测最高概率分箱 (1~63) 与 数学期望分箱 E[H] = sum(k * p_k)
    bins_vec = np.arange(VocabSpec.GROUP_SIZE, dtype=np.float32)  # 0..63
    out_dict['pred_max_bin'] = all_probs_64[:, 1:].argmax(axis=1) + 1
    out_dict['pred_exp_bin'] = np.round((all_probs_64 * bins_vec).sum(axis=1), cfg.float_precision)

    # 计算主目标分箱 (如 >= 38) 及附加阈值分箱的累计概率和
    all_thresholds = [cfg.target_min_bin] + [b for b in cfg.extra_min_bins if b != cfg.target_min_bin]
    for min_b in all_thresholds:
        assert 1 <= min_b < VocabSpec.GROUP_SIZE, f"分箱阈值 {min_b} 必须在 1~63 之间!"
        prob_sum = all_probs_64[:, min_b:].sum(axis=1)
        out_dict[f'prob_ge_{min_b}'] = np.round(prob_sum, cfg.float_precision)

    # 拼接明日 H 的 1~63 分箱完整预测概率表 (p_bin_01 ~ p_bin_63)
    if cfg.save_full_prob_table:
        for b in range(1, VocabSpec.GROUP_SIZE):
            out_dict[f'p_bin_{b:02d}'] = np.round(all_probs_64[:, b], cfg.float_precision)

    df_out = pd.DataFrame(out_dict)

    # 5. 打印验证集评估报告并保存 CSV
    print(f"\n>> [{stage_name}] 推理完成 | 验证集 Last-H CrossEntropy Loss: {avg_last_h_loss:.4f}")
    print_evaluation_report(df_out, cfg.target_min_bin, stage_name)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv_path)), exist_ok=True)
    df_out.to_csv(out_csv_path, index=False)
    print(f"\n>> 预测结果已成功对齐 (id0, id1, id2) 并导出至 CSV: {out_csv_path}")
    print(f"   └─ CSV 维度: {df_out.shape[0]} 行 x {df_out.shape[1]} 列 | 前 8 列预览: {list(df_out.columns[:8])}")

    return df_out


if __name__ == "__main__":
    run_stage_inference(cfg)
