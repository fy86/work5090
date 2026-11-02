# -*- coding: utf-8 -*-
"""
train_stage.py
Quant-GPT 多阶段自回归大模型 —— 通用单阶段训练引擎 (Template)
用法:
  需要运行多个 Stage 时，直接将本文件复制为 train_stage1.py, train_stage2.py ...
  仅修改下方 `StageConfig` 中的参数即可实现：
  [纯个股 / +指数 / +15m线] x [all_ohlc / all_h / last_k_h / last_h] x [验证早停 / 实盘全量裸跑]
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

# 开启 Ampere+ 架构 TF32 矩阵乘法加速
torch.set_float32_matmul_precision('high')


# ==============================================================================
# 1. 全局词表宪法与阶段超参数配置 (复制本文件后，仅需修改 StageConfig 即可立项新 Stage)
# ==============================================================================
class VocabSpec:
    """
    10 象限 640 词表物理分配宪法 (定死不变):
      Group 0 (0~63)   : 个股日线 H (0 为 <sep_stk> 周期定界符 & 明日H预测触发符, 1~63 为分箱)
      Group 1 (64~127) : 个股日线 O (64 为定界符, 65~127 为分箱)
      Group 2 (128~191): 个股日线 L (128 为定界符, 129~191 为分箱)
      Group 3 (192~255): 个股日线 C (192 为定界符, 193~255 为分箱)
      Group 4 (256~319): 指数日线 H (256 为 <sep_idx> 定界符, 257~319 为分箱)
      Group 5 (320~383): 指数日线 O
      Group 6 (384~447): 指数日线 L
      Group 7 (448~511): 指数日线 C
      Group 8 (512~575): 个股15分钟收盘 C (512 为 <sep_m15> 定界符, 513~575 为分箱)
      Group 9 (576~639): 预留扩展槽 (如成交量/换手率等)
    """
    GROUP_SIZE: int = 64
    NUM_EMB_GROUPS: int = 10
    VOCAB_SIZE: int = 640          # 10 * 64
    NUM_HEAD_GROUPS: int = 4       # 仅预测个股 H(G0), O(G1), L(G2), C(G3) 共 256 维

    SEP_STK_ID: int = 0            # 个股日线周期定界符 & 预测明日H触发符
    SEP_IDX_ID: int = 256          # 指数日线周期定界符
    SEP_M15_ID: int = 512          # 15分钟线单日定界符

    GT_2PCT_MIN_BIN: int = 38      # 涨幅 >= 2% 起始分箱 (1~63 坐标系)
    LIMIT_UP_MIN_BIN: int = 60     # 涨停起始分箱 (1~63 坐标系，仅在 infer_stage.py 后处理使用)

    # === 2D 层次化 RoPE 天内槽位表 (以 D3 单日完整排布为基准，D1/D2/D3 共用，保证跨阶段坐标一致) ===
    #   个股: <sep_stk>=0, H=1, O=2, L=3, C=4
    #   指数: <sep_idx>=5, H=6, O=7, L=8, C=9
    #   15m : <sep_m15>=10, 当日第 k 根收盘=10+k (11~26)
    #   预留 Group 9: <sep>=27, 数值=28 (占位，启用时按实际排布细化)
    MAX_SLOTS: int = 64


class StageConfig:
    # === [A] 阶段身份与文件路径配置 ===
    stage_name: str = "Stage1_Stock_AllOHLC"
    base_dir: str = r"../tmp"
    # 可选数据集: 'd1_stk_322.parquet' (纯个股) | 'd2_idx_642.parquet' (+指数) | 'd3_m15_778.parquet' (+15m)
    train_parquet: str = "ds_h_s1_60_100.parquet"
    val_parquet: str = "d1_stk_322_val.parquet"

    load_ckpt_path: str = ""                 # 上一阶段权重路径 (空字符串 "" 表示从头随机初始化训练)
    save_ckpt_path: str = "ckpt_stage1.pth"  # 当前阶段最佳权重保存文件名

    # === [B] 跨阶段词表克隆热启动指令 (在加载 load_ckpt_path 后、冻结前立即执行) ===
    # 格式: {源Group号: [目标Group号列表]}
    # 示例 - Stage 1 (同源初始化个股 O,L,C): {0: [1, 2, 3]}
    # 示例 - Stage 2 (个股克隆给指数 H->H, O->O, L->L, C->C): {0: [4], 1: [5], 2: [6], 3: [7]}
    # 示例 - Stage 3 (个股 C 克隆给 15m C): {3: [8]}
    # 示例 - Stage 4~6 (词表已成熟，无需克隆): {}
    clone_emb_map: dict = {0: [1, 2, 3]}

    # === [C] 手术刀级解冻开关 (不在列表中的组件将被物理锁死 requires_grad=False) ===
    # 1. 允许训练的输入词表象限 (0~9):
    #    Stage1: [0,1,2,3] | Stage2: [4,5,6,7] | Stage3: [8] | Stage4~6: [] (全锁死)
    train_emb_groups: list = [0, 1, 2, 3]

    # 2. 允许训练的 Transformer 层索引 (共 16 层: 0~15, 分4个Zone: 0..3, 4..7, 8..11, 12..15):
    #    Stage1: list(range(0, 16)) | Stage2: list(range(4, 16)) | Stage3/4: list(range(8, 16)) | Stage6: [14, 15]
    train_block_indices: list = list(range(0, 16))

    # 3. 允许训练的输出头象限 (0:H, 1:O, 2:L, 3:C):
    #    'all_ohlc' 阶段设为 [0, 1, 2, 3]; 只算 H 的阶段 ('all_h'/'last_k_h'/'last_h') 设为 [0]
    train_head_groups: list = [0, 1, 2, 3]

    # === [D] 损失函数模式 (Loss Mode) ===
    # 可选值:
    #   'all_ohlc' : 计算所有个股 H, O, L, C 的等权 Loss (输出 256 维)
    #   'all_h'    : 硬切换，仅计算全序列所有跨日个股 H 的 Loss (输出 64 维)
    #   'last_k_h' : 硬切换，仅计算序列最后 last_k_days 天的跨日个股 H 的 Loss (输出 64 维)
    #   'last_h'   : 单点决胜，仅计算序列最末尾最后一个 H (明日H) 的 Loss (仅最后一步输出 64 维)
    loss_mode: str = "all_ohlc"
    last_k_days: int = 8                     # 当 loss_mode == 'last_k_h' 时生效 (例如聚焦最后8天带15m前导的H)
    context_warmup_tokens: int = 1  # 25          # 'all_ohlc' / 'all_h' 下跳过序列前 N 个缺乏上下文的 Token 🔴🔴
    ignore_index: int = -100

    # === [E] 模型骨干固定超参数 (所有 Stage 严禁改动，确保权重 100% 兼容) ===
    max_seq_len: int = 1024                  # RoPE 预计算最大长度 (自适应 322 / 642 / 778 等任意序列)
    n_layer: int = 16                        # 16 层 (4 个功能特区 x 4 层)
    n_head: int = 8                          # 8 头 (head_dim = 512 // 8 = 64, 完美对齐 FlashAttention)
    n_embd: int = 512                        # 512 维特征表示 (~52M 参数量)
    use_qk_norm: bool = True                 # 强制开启 RMSNorm 的 QK-Norm，锁死多阶段切换时的注意力尺度

    # 位置编码模式 (无可学习参数，切换后 strict 加载不会报错，但注意力语义改变 -> 必须从 Stage1 重训):
    #   '1d': 传统一维连续 RoPE (按拉平序列下标旋转，D1/D2/D3 中"隔1天"的距离各不相同，仅作对照)
    #   '2d': 2D 层次化 RoPE (head_dim 拆分: 前 rope_day_dims 维按「交易日序号」旋转, 其余按「天内槽位」旋转)
    rope_mode: str = '2d'
    rope_day_dims: int = 32                  # 天轴维数 (偶数, < head_dim)，其余 head_dim - rope_day_dims 维给槽位轴
    rope_day_base: float = 1000.0            # 天轴 base: 跨度 64~250+ 天，偏低频
    rope_slot_base: float = 100.0            # 槽位轴 base: 跨度 0~26，偏高频以分辨相邻槽位

    # === [F] 正则化与优化器超参数 ===
    dropout: float = 0.15
    token_dropout: float = 0.08              # 仅对数值分箱 (idx % 64 != 0) 置零，100% 保护所有 <sep> 定界符
    weight_decay: float = 0.08
    lr: float = 1.5e-4  # 2.5e-4                       # 建议: Stage1 2.5e-4 -> Stage2/3 1e-4 -> Stage4 5e-5 -> Stage6 1.5e-5
    epochs: int = 2
    batch_size: int = 128

    # === [G] 验证早停 vs 实盘全量裸跑控制 ===
    use_val: bool = False        #              # False: 进入【实盘裸跑模式】，不加载验证集，按平滑训练 Loss 或步数停机   # 🔴🔴
    eval_interval_steps: int = 200    # 200 🔴🔴
    patience: int = 3
    anchor_on_last_h: bool = True            # True: 无论当前 Stage 训练什么 Loss，一律以验证集 Last-H Loss 作为保存最优模型的唯一标准
    max_train_steps: int = 1200
    target_train_loss: float = 2.36         # 实盘裸跑模式 (use_val=False) 下的目标停止训练 Loss  # 🔴🔴

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = StageConfig()


# ==============================================================================
# 2. 模型骨干定义 (10象限词表 + QK-Norm + SwiGLU + 4象限按需输出头)
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
    """2D RoPE 单轴查表: 返回 cos/sin [max_pos, rot_dim]，前向时按位置坐标 gather"""
    inv_freq = 1.0 / (base ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def build_2d_positions(idx: torch.Tensor, max_day: int):
    """
    由 token id 直接推导 2D 坐标 (与数据排布无关，D1/D2/D3 通用):
      day : 每遇到 <sep_stk>=0 进入新的一天 (序列末尾 [0, 明日H] 自动落在第 64 天)
      slot: 按 VocabSpec 天内槽位表映射
    idx: [B, T] -> (day [B, T], slot [B, T])
    """
    G = VocabSpec.GROUP_SIZE
    g = idx // G
    is_sep = (idx % G == 0)

    day = (idx == VocabSpec.SEP_STK_ID).long().cumsum(dim=1) - 1
    day = day.clamp(0, max_day)

    slot = torch.where(
        g < 4, torch.where(is_sep, 0, g + 1),                      # 个股 0..4
        torch.where(
            g < 8, torch.where(is_sep, 5, g + 2),                  # 指数 5..9
            torch.where(g == 9, torch.where(is_sep, 27, 28), 0)    # 预留组占位
        )
    )

    # 15m: 10 + 当日第 k 根 (k=0 为 <sep_m15>)
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
    """
    split_dim == 0 : 1D RoPE，cos/sin 形状 [1, 1, L, D]
    split_dim  > 0 : 2D RoPE，前 split_dim 维 (天轴) 与其余维 (槽位轴) 各自独立 rotate-half，
                     cos/sin 形状 [B, 1, T, D] (已按坐标拼接好)
    """
    T = x.shape[2]
    if split_dim > 0:
        x_rot = torch.cat((_rotate_half(x[..., :split_dim]), _rotate_half(x[..., split_dim:])), dim=-1)
    else:
        x_rot = _rotate_half(x)
    cos_t = cos[:, :, :T, :].to(x.dtype)
    sin_t = sin[:, :, :T, :].to(x.dtype)
    return (x * cos_t) + (x_rot * sin_t)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: StageConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_qk_norm = config.use_qk_norm
        self.rope_split = config.rope_day_dims if config.rope_mode == '2d' else 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # 使用 RMSNorm 的 QK-Norm，防止多阶段冻结/解冻与新词表注入时的注意力内积漂移
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

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, config: StageConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        hidden_dim = int(config.n_embd * 8 / 3)
        hidden_dim = 64 * ((hidden_dim + 64 - 1) // 64)  # 512 -> 1408
        self.mlp = SwiGLU(config.n_embd, hidden_dim)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_drop(self.attn(self.ln_1(x), cos, sin))
        x = x + self.resid_drop(self.mlp(self.ln_2(x)))
        return x


class PartitionedEmbedding(nn.Module):
    """
    10 象限物理分立词表 (10 x nn.Embedding(64, n_embd)):
    支持独立冻结任意象限，同时在前向传播时拼接为连续张量调用 F.embedding，零性能损耗。
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.groups = nn.ModuleList([
            nn.Embedding(VocabSpec.GROUP_SIZE, n_embd)
            for _ in range(VocabSpec.NUM_EMB_GROUPS)
        ])

    @torch.no_grad()
    def execute_clone_map(self, clone_map: dict):
        if not clone_map:
            return
        for src_g, dst_list in clone_map.items():
            for dst_g in dst_list:
                self.groups[dst_g].weight.copy_(self.groups[src_g].weight)
                print(f"   >> [词表热启动] 已将 Embedding Group[{src_g}] 权重精准克隆至 Group[{dst_g}]")

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat([g.weight for g in self.groups], dim=0)  # [640, n_embd]
        return F.embedding(idx, full_weight)


class PartitionedLMHead(nn.Module):
    """
    4 象限物理分立输出头 (4 x nn.Linear(n_embd, 64, bias=False)):
    groups[0]: 预测个股 H (0~63)
    groups[1]: 预测个股 O (64~127)
    groups[2]: 预测个股 L (128~191)
    groups[3]: 预测个股 C (192~255)
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.groups = nn.ModuleList([
            nn.Linear(n_embd, VocabSpec.GROUP_SIZE, bias=False)
            for _ in range(VocabSpec.NUM_HEAD_GROUPS)
        ])

    def forward_all_ohlc(self, x: torch.Tensor) -> torch.Tensor:
        # 拼接全部 4 个头输出 [B, T, 256]
        return torch.cat([head(x) for head in self.groups], dim=-1)

    def forward_h_only(self, x: torch.Tensor) -> torch.Tensor:
        # 仅调用 Group 0 输出个股 H 的 [B, T, 64] 或 [B, 1, 64]，彻底跳过 O,L,C 的矩阵乘法！
        return self.groups[0](x)


class ModernGPT(nn.Module):
    def __init__(self, config: StageConfig):
        super().__init__()
        self.config = config
        self.wte = PartitionedEmbedding(config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = PartitionedLMHead(config.n_embd)

        head_dim = config.n_embd // config.n_head
        if config.rope_mode == '2d':
            day_dims = config.rope_day_dims
            slot_dims = head_dim - day_dims
            assert day_dims % 2 == 0 and slot_dims % 2 == 0 and 0 < day_dims < head_dim, \
                f"rope_day_dims={day_dims} 非法 (head_dim={head_dim}, 两轴维数都须为正偶数)"
            d_cos, d_sin = precompute_rope_table(config.max_seq_len, day_dims, config.rope_day_base)
            s_cos, s_sin = precompute_rope_table(VocabSpec.MAX_SLOTS, slot_dims, config.rope_slot_base)
            # persistent=False: 不进入 state_dict，权重文件结构与 1D 版本完全一致
            self.register_buffer("rope_day_cos", d_cos, persistent=False)
            self.register_buffer("rope_day_sin", d_sin, persistent=False)
            self.register_buffer("rope_slot_cos", s_cos, persistent=False)
            self.register_buffer("rope_slot_sin", s_sin, persistent=False)
        elif config.rope_mode == '1d':
            cos, sin = precompute_rope(config.max_seq_len, head_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        else:
            raise ValueError(f"未知的 rope_mode: {config.rope_mode}")

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('mlp.w3.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def build_rope(self, idx: torch.Tensor):
        """返回供 16 层共用的 (cos, sin)。2D 模式按 token id 推导坐标后查表拼接为 [B, 1, T, head_dim]"""
        if self.config.rope_mode == '2d':
            day, slot = build_2d_positions(idx, max_day=self.rope_day_cos.size(0) - 1)
            cos = torch.cat([self.rope_day_cos[day], self.rope_slot_cos[slot]], dim=-1).unsqueeze(1)
            sin = torch.cat([self.rope_day_sin[day], self.rope_slot_sin[slot]], dim=-1).unsqueeze(1)
            return cos, sin
        return self.cos, self.sin

    def forward(self, idx: torch.Tensor, mode: str = 'all_ohlc') -> torch.Tensor:
        """
        mode:
          - 'all_ohlc' : 全序列通过 ln_f 并计算 256 维输出 [B, T, 256]
          - 'all_h' / 'last_k_h' : 全序列通过 ln_f，但仅通过 Head[0] 计算 64 维 H 输出 [B, T, 64]
          - 'last_h'   : 仅对最后一个时间步 x[:, -1:, :] 执行 ln_f 和 Head[0]，返回 [B, 1, 64]
        """
        x = self.wte(idx)

        # 结构保护型 Token Dropout：所有满足 idx % 64 == 0 的定界符 <sep> 严禁置零！
        if self.training and self.config.token_dropout > 0.0:
            is_numeric = (idx % VocabSpec.GROUP_SIZE != 0)
            drop_prob = torch.rand_like(idx, dtype=x.dtype) < self.config.token_dropout
            keep_mask = ~(is_numeric & drop_prob)
            x = x * keep_mask.unsqueeze(-1).to(x.dtype)

        x = self.drop(x)
        cos, sin = self.build_rope(idx)
        for block in self.blocks:
            x = block(x, cos, sin)

        if mode == 'last_h':
            x_last = self.ln_f(x[:, -1:, :])
            return self.lm_head.forward_h_only(x_last)  # [B, 1, 64]
        elif mode in ('all_h', 'last_k_h'):
            x = self.ln_f(x)
            return self.lm_head.forward_h_only(x)       # [B, T, 64]
        else:
            x = self.ln_f(x)
            return self.lm_head.forward_all_ohlc(x)     # [B, T, 256]


# ==============================================================================
# 3. 手术刀级参数冻结与优化器构建引擎
# ==============================================================================
def configure_stage_freezing_and_optimizer(model: ModernGPT, cfg: StageConfig):
    # 1. 先默认全部锁死
    for p in model.parameters():
        p.requires_grad = False

    # 2. 按配置解冻指定的输入词表象限
    for g_idx in cfg.train_emb_groups:
        model.wte.groups[g_idx].requires_grad_(True)

    # 3. 按配置解冻指定的 Transformer Block 层
    for b_idx in cfg.train_block_indices:
        model.blocks[b_idx].requires_grad_(True)

    # 4. 顶层 RMSNorm 始终跟随解冻 (确保输出分布尺度可微调)
    model.ln_f.requires_grad_(True)

    # 5. 按配置解冻指定的输出头象限
    for h_idx in cfg.train_head_groups:
        model.lm_head.groups[h_idx].requires_grad_(True)

    # 6. 统计参数并构建仅包含 requires_grad=True 参数的 AdamW
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or 'ln_' in name or 'q_norm' in name or 'k_norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': cfg.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=cfg.lr, fused=('cuda' in cfg.device))

    print(f"   >> [参数解冻报告] 总参数: {total_params / 1e6:.2f}M | 当前 Stage 激活训练: {trainable_params / 1e6:.2f}M ({100.0 * trainable_params / total_params:.1f}%)")
    print(f"      ├─ 激活词表组 (wte)   : {cfg.train_emb_groups}")
    print(f"      ├─ 激活网络层 (blocks): {cfg.train_block_indices[0]}..{cfg.train_block_indices[-1]} (共 {len(cfg.train_block_indices)} 层)" if cfg.train_block_indices else "      ├─ 激活网络层 (blocks): []")
    print(f"      └─ 激活输出头 (heads) : {cfg.train_head_groups} | Loss模式: '{cfg.loss_mode}'")

    return optimizer


def set_stage_train_mode(model: nn.Module, cfg: StageConfig):
    """
    进入训练模式，但将被冻结的 Transformer Block 切回 eval()：
    冻结层不再施加注意力/残差 Dropout，作为稳定的确定性特征提取器，避免向可训练上层注入额外噪声。
    (输入端的 Embedding Dropout / Token Dropout 保持开启，作为数据增强)
    """
    raw_model = getattr(model, '_orig_mod', model)
    model.train()
    trainable_blocks = set(cfg.train_block_indices)
    for b_idx, block in enumerate(raw_model.blocks):
        if b_idx not in trainable_blocks:
            block.eval()


# ==============================================================================
# 4. 数据集加载 (int32 元数据 + int16 序列显存驻留) 与多模式 Loss 函数
# ==============================================================================
class SequenceDataset(Dataset):
    """
    读取格式为 [id0, id1, id2, t0, t1, ..., tn] 的 Parquet 文件：
    - id0, id1 (yyyymmdd), id2 强制保存为 int32
    - t0 ~ tn 强制保存为 int16 (显存减半)
    """
    def __init__(self, parquet_file: str):
        df = pd.read_parquet(parquet_file, engine='pyarrow')
        self.seq_cols = sorted(
            [c for c in df.columns if c.startswith('t') and c[1:].isdigit()],
            key=lambda c: int(c[1:])
        )
        if not self.seq_cols:
            raise ValueError(f"在 {parquet_file} 中未找到 t0, t1, ... 序列列！")

        self.meta_cols = [c for c in df.columns if c not in self.seq_cols]
        self.meta_df = df[self.meta_cols].astype(np.int32).copy() if self.meta_cols else pd.DataFrame(index=df.index)
        self.seq_len = len(self.seq_cols)
        self.sequences = torch.tensor(df[self.seq_cols].values, dtype=torch.int16)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def compute_stage_loss(logits: torch.Tensor, y: torch.Tensor, cfg: StageConfig) -> torch.Tensor:
    """
    支持 4 种模式的通用自回归损失函数 (训练阶段纯客观拟合，不掺杂涨停过滤)：
    1. 'all_ohlc' : logits [B, T, 256], 目标为个股 H,O,L,C (0 < y < 256 且 y % 64 != 0)
    2. 'all_h'    : logits [B, T, 64],  目标仅为跨日个股 H (1 <= y < 64)
    3. 'last_k_h' : logits [B, T, 64],  目标仅为序列最后 K 个跨日个股 H
    4. 'last_h'   : logits [B, 1, 64],  目标仅为序列最末尾的 1 个 H (y[:, -1])
    """
    mode = cfg.loss_mode

    if mode == 'last_h':
        # logits: [B, 1, 64] -> [B, 64]
        last_logits = logits.squeeze(1).clone()
        last_logits[:, 0] = -1e4  # 屏蔽 0 号 <sep_stk> 通道
        last_target = y[:, -1]
        return F.cross_entropy(last_logits, last_target, ignore_index=cfg.ignore_index)

    B, T, V = logits.shape
    targets = y.clone()
    clean_logits = logits.clone()

    if mode == 'all_ohlc':
        # 仅保留个股 H, O, L, C 的数值分箱 (y < 256 且非定界符)，屏蔽指数(256~511)和15m(512~575)
        invalid_mask = (targets >= 256) | (targets % VocabSpec.GROUP_SIZE == 0)
        targets[invalid_mask] = cfg.ignore_index
        if cfg.context_warmup_tokens > 0 and T > cfg.context_warmup_tokens:
            targets[:, :cfg.context_warmup_tokens] = cfg.ignore_index

        # 净化输出空间中的 4 个定界符通道 (0, 64, 128, 192)
        for sep_id in (0, 64, 128, 192):
            clean_logits[:, :, sep_id] = -1e4

    elif mode == 'all_h':
        # 全序列中只有个股 H 落在 1 ~ 63 之间！一行代码精准锁定所有跨日个股 H
        invalid_mask = (targets <= 0) | (targets >= VocabSpec.GROUP_SIZE)
        targets[invalid_mask] = cfg.ignore_index
        if cfg.context_warmup_tokens > 0 and T > cfg.context_warmup_tokens:
            targets[:, :cfg.context_warmup_tokens] = cfg.ignore_index

        clean_logits[:, :, 0] = -1e4

    elif mode == 'last_k_h':
        # 仅保留最后 last_k_days 个跨日个股 H (1 <= y < 64)
        is_stk_h = (targets >= 1) & (targets < VocabSpec.GROUP_SIZE)
        # 从后往前累加计数个股 H 的出现位置
        h_rev_cumsum = torch.flip(is_stk_h.long(), dims=[1]).cumsum(dim=1)
        h_order_from_end = torch.flip(h_rev_cumsum, dims=[1])
        valid_last_k = is_stk_h & (h_order_from_end <= cfg.last_k_days)

        targets[~valid_last_k] = cfg.ignore_index
        clean_logits[:, :, 0] = -1e4

    else:
        raise ValueError(f"未知的 loss_mode: {mode}")

    return F.cross_entropy(
        clean_logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=cfg.ignore_index
    )


@torch.no_grad()
def evaluate_objective(model: nn.Module, val_data_gpu: torch.Tensor, cfg: StageConfig, autocast_ctx_fn):
    """
    客观验证集评估 (不做涨停剔除，客观衡量全样本拟合度与 Last-H 分布质量)：
    返回: (stage_loss, last_h_loss, base_rate, p50, r50, p65, p75, cnt65, cnt75)
    """
    model.eval()

    stage_loss_sum = 0.0
    stage_batch_cnt = 0

    last_h_loss_sum = 0.0
    total_samples = 0

    pos_actual = 0
    tp_50 = fp_50 = 0
    tp_65 = fp_65 = 0
    tp_75 = fp_75 = 0

    for i in range(0, len(val_data_gpu), cfg.batch_size):
        val_seq = val_data_gpu[i:i + cfg.batch_size].long()
        actual_b = len(val_seq)

        # 尾部残缺 Batch 零填充，严防触发 torch.compile 动态 shape 重编译
        if actual_b < cfg.batch_size:
            pad_seq = torch.zeros(
                (cfg.batch_size - actual_b, val_seq.size(1)),
                dtype=val_seq.dtype, device=val_seq.device
            )
            val_seq_input = torch.cat([val_seq, pad_seq], dim=0)
        else:
            val_seq_input = val_seq

        vx = val_seq_input[:, :-1]
        vy_full = val_seq_input[:, 1:]

        with autocast_ctx_fn():
            logits_full = model(vx, mode=cfg.loss_mode)
            val_logits = logits_full[:actual_b]
            vy = vy_full[:actual_b]
            b_loss = compute_stage_loss(val_logits, vy, cfg)

        if not torch.isnan(b_loss):
            stage_loss_sum += b_loss.item() * actual_b
            stage_batch_cnt += actual_b

        # 提取最后一个时间步 (预测明日个股 H) 的前 64 维 Logits
        last_h_logits = val_logits[:, -1, :VocabSpec.GROUP_SIZE].float()
        last_h_logits[:, 0] = -float('inf')
        last_h_target = vy[:, -1]

        l_loss = F.cross_entropy(last_h_logits, last_h_target, reduction='sum')
        last_h_loss_sum += l_loss.item()
        total_samples += actual_b

        # 全样本客观 P(H >= 38) 胜率统计
        probs = F.softmax(last_h_logits, dim=-1)
        prob_gt_2pct = probs[:, VocabSpec.GT_2PCT_MIN_BIN:].sum(dim=-1)
        true_gt_2pct = (last_h_target >= VocabSpec.GT_2PCT_MIN_BIN)

        pos_actual += true_gt_2pct.sum().item()

        p50 = (prob_gt_2pct >= 0.50)
        tp_50 += (p50 & true_gt_2pct).sum().item()
        fp_50 += (p50 & ~true_gt_2pct).sum().item()

        p65 = (prob_gt_2pct >= 0.65)
        tp_65 += (p65 & true_gt_2pct).sum().item()
        fp_65 += (p65 & ~true_gt_2pct).sum().item()

        p75 = (prob_gt_2pct >= 0.75)
        tp_75 += (p75 & true_gt_2pct).sum().item()
        fp_75 += (p75 & ~true_gt_2pct).sum().item()

    set_stage_train_mode(model, cfg)

    avg_stage_loss = stage_loss_sum / max(1, stage_batch_cnt)
    avg_last_h_loss = last_h_loss_sum / max(1, total_samples)
    base_rate = pos_actual / max(1, total_samples)

    prec_50 = tp_50 / max(1, tp_50 + fp_50)
    rec_50 = tp_50 / max(1, pos_actual)
    prec_65 = tp_65 / max(1, tp_65 + fp_65)
    prec_75 = tp_75 / max(1, tp_75 + fp_75)

    return (
        avg_stage_loss, avg_last_h_loss, base_rate,
        prec_50, rec_50, prec_65, prec_75,
        tp_65 + fp_65, tp_75 + fp_75
    )


# ==============================================================================
# 5. 阶段训练主引擎 (支持检查点接力、词表克隆、模块化冻结、验证早停与实盘裸跑)
# ==============================================================================
def _rope_cfg_dict(cfg: StageConfig) -> dict:
    if cfg.rope_mode == '2d':
        return {'rope_mode': '2d', 'rope_day_dims': cfg.rope_day_dims,
                'rope_day_base': float(cfg.rope_day_base), 'rope_slot_base': float(cfg.rope_slot_base)}
    return {'rope_mode': cfg.rope_mode}


def run_stage_training(cfg: StageConfig):
    train_path = os.path.join(cfg.base_dir, cfg.train_parquet)
    val_path = os.path.join(cfg.base_dir, cfg.val_parquet)
    load_path = os.path.join(cfg.base_dir, cfg.load_ckpt_path) if cfg.load_ckpt_path else ""
    save_path = os.path.join(cfg.base_dir, cfg.save_ckpt_path)

    print("\n" + "=" * 80)
    print(f">> 启动 [{cfg.stage_name}] 训练引擎 | Device: {cfg.device}")
    print("=" * 80)

    train_dataset = SequenceDataset(train_path)
    train_data_gpu = train_dataset.sequences.to(cfg.device)  # int16 常驻显存
    seq_len = train_dataset.seq_len
    assert seq_len <= cfg.max_seq_len, f"序列长度 {seq_len} 超过 max_seq_len {cfg.max_seq_len}!"

    print(f"   >> 训练集加载完毕: {len(train_dataset)} 条样本 | seq_len={seq_len} (int16 显存占用: {train_data_gpu.element_size() * train_data_gpu.nelement() / 1e6:.2f} MB)")

    model = ModernGPT(cfg).to(cfg.device)

    # 1. 加载上一阶段 Checkpoint 权重 (若指定)
    if load_path and os.path.exists(load_path):
        ckpt = torch.load(load_path, map_location=cfg.device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        prev_stage = ckpt.get('stage_name', 'Unknown') if isinstance(ckpt, dict) else 'RawStateDict'
        print(f"   >> [权重接力] 已成功加载上一阶段权重: {load_path} (来源: {prev_stage})")

        # RoPE 无参数，strict 加载无法发现位置编码不一致，需显式校验 (旧权重无该字段视为 '1d')
        prev_rope = ckpt.get('rope_cfg', {'rope_mode': '1d'}) if isinstance(ckpt, dict) else {'rope_mode': '1d'}
        cur_rope = _rope_cfg_dict(cfg)
        if prev_rope != cur_rope:
            raise ValueError(
                f"RoPE 配置不一致! 上一阶段: {prev_rope} | 当前: {cur_rope}\n"
                f"位置编码变更会改变注意力语义，请保持一致或从 Stage1 重新训练。"
            )
    elif cfg.load_ckpt_path:
        raise FileNotFoundError(f"未找到指定的上一阶段权重文件: {load_path}")
    else:
        print("   >> [从头初始化] 未指定上一阶段权重，当前为基座预训练起点。")
    print(f"   >> [位置编码] {_rope_cfg_dict(cfg)}")

    # 2. 执行跨象限词表克隆热启动
    model.wte.execute_clone_map(cfg.clone_emb_map)

    # 3. 执行手术刀级参数冻结与构建优化器
    optimizer = configure_stage_freezing_and_optimizer(model, cfg)

    # 4. 编译加速
    if 'cuda' in cfg.device:
        try:
            model = torch.compile(model)
            print("   >> [torch.compile] 算子融合编译加速已开启")
        except Exception as e:
            print(f"   >> [torch.compile] 跳过: {e}")

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    enable_amp = ('cuda' in cfg.device)
    scaler = torch.amp.GradScaler('cuda') if (enable_amp and amp_dtype == torch.float16) else None

    def get_autocast_ctx():
        return torch.amp.autocast(
            device_type=('cuda' if 'cuda' in cfg.device else 'cpu'),
            dtype=amp_dtype,
            enabled=enable_amp
        )

    val_data_gpu = None
    if cfg.use_val and os.path.exists(val_path):
        val_dataset = SequenceDataset(val_path)
        val_data_gpu = val_dataset.sequences.to(cfg.device)
        print(f"   >> 验证集已加载: {len(val_dataset)} 条样本 (客观全样本评估，涨停过滤后置到 infer_stage.py)")
    else:
        print(f"   >> [实盘全量裸跑模式] use_val={cfg.use_val}，将依据平滑训练 Loss <= {cfg.target_train_loss} 或步数上限 {cfg.max_train_steps} 停机。")

    train_steps_per_epoch = max(1, len(train_data_gpu) // cfg.batch_size)
    total_steps = min(cfg.max_train_steps, cfg.epochs * train_steps_per_epoch)
    warmup_steps = 0 #  int(total_steps * 0.08)   # 🔴🔴

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = min(1.0, float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
        return 0.05 + 0.5 * (1.0 - 0.05) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_metric = float('inf')
    smoothed_train_loss = float('inf')
    patience_counter = 0
    global_step = 0

    interval_loss_sum = torch.tensor(0.0, device=cfg.device)
    interval_step_count = 0

    def _save_stage_checkpoint(best_metric_name, best_metric_val, metrics_dict=None):
        raw_model = getattr(model, '_orig_mod', model)
        payload = {
            'stage_name': cfg.stage_name,
            'loss_mode': cfg.loss_mode,
            'rope_cfg': _rope_cfg_dict(cfg),
            'seq_len': seq_len,
            'best_metric_name': best_metric_name,   # 'val_last_h_loss' | 'val_stage_loss' | 'smoothed_train_loss'
            'best_metric': best_metric_val,
            'metrics': metrics_dict or {},
            'model_state_dict': raw_model.state_dict()
        }
        torch.save(payload, save_path)

    set_stage_train_mode(model, cfg)
    frozen_blocks = [i for i in range(cfg.n_layer) if i not in set(cfg.train_block_indices)]
    if frozen_blocks:
        print(f"   >> [冻结层 eval] Block {frozen_blocks} 训练期间保持 eval() (关闭内部 Dropout)")
    num_samples = len(train_data_gpu)

    for epoch in range(cfg.epochs):
        indices = torch.randperm(num_samples, device=cfg.device)
        steps_to_run = min(train_steps_per_epoch, total_steps - global_step)
        pbar = tqdm(range(steps_to_run), desc=f"[{cfg.stage_name}] Epoch {epoch + 1}/{cfg.epochs}")

        for batch_idx in pbar:
            global_step += 1
            start_idx = batch_idx * cfg.batch_size
            end_idx = start_idx + cfg.batch_size
            batch_indices = indices[start_idx:end_idx]

            # 从 int16 显存张量切片并转为 long 送入网络
            seq = train_data_gpu[batch_indices].long()
            x = seq[:, :-1]
            y = seq[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            with get_autocast_ctx():
                logits = model(x, mode=cfg.loss_mode)
                loss = compute_stage_loss(logits, y, cfg)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()

            interval_loss_sum += loss.detach()
            interval_step_count += 1

            if global_step % 10 == 0 or global_step == total_steps:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Mode': cfg.loss_mode,
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            # === 定期评估与保存 ===
            if global_step % cfg.eval_interval_steps == 0 or global_step == total_steps:
                smoothed_train_loss = (interval_loss_sum / max(1, interval_step_count)).item()
                interval_loss_sum.zero_()
                interval_step_count = 0

                if val_data_gpu is not None:
                    (val_stage_loss, val_last_h_loss, base_rate,
                     p50, r50, p65, p75, cnt65, cnt75) = evaluate_objective(
                        model, val_data_gpu, cfg, get_autocast_ctx
                    )
                    print(
                        f"\n[Step {global_step:05d}] Train({cfg.loss_mode}): {smoothed_train_loss:.4f} | "
                        f"Val({cfg.loss_mode}): {val_stage_loss:.4f} | Val Last-H Loss: {val_last_h_loss:.4f}\n"
                        f"   └─ [全样本客观分布 (基准率:{base_rate:.4f})] "
                        f"P@0.50: {p50:.4f} (R:{r50:.4f}) | "
                        f"P@0.65: {p65:.4f} (数:{cnt65}) | "
                        f"P@0.75: {p75:.4f} (数:{cnt75})"
                    )

                    target_metric = val_last_h_loss if cfg.anchor_on_last_h else val_stage_loss
                    metric_name = 'val_last_h_loss' if cfg.anchor_on_last_h else 'val_stage_loss'
                    if target_metric < best_val_metric:
                        best_val_metric = target_metric
                        patience_counter = 0
                        _save_stage_checkpoint(metric_name, target_metric, {
                            'val_stage_loss': val_stage_loss,
                            'val_last_h_loss': val_last_h_loss,
                            'base_rate': base_rate,
                            'P@0.50': p50, 'R@0.50': r50,
                            'P@0.65': p65, 'cnt@0.65': cnt65,
                            'P@0.75': p75, 'cnt@0.75': cnt75,
                        })
                        print(f"   ★ [{cfg.stage_name}] 最佳模型已保存 -> {save_path} ({metric_name}: {target_metric:.4f})")
                    else:
                        patience_counter += 1
                        if patience_counter >= cfg.patience:
                            print(f"\n>> Early Stopping 触发! 连续 {cfg.patience} 次验证未提升。")
                            break
                else:
                    # 实盘裸跑模式
                    print(f"\n[Step {global_step:05d} 实盘裸跑] 平滑 Train Loss ({cfg.loss_mode}): {smoothed_train_loss:.4f}")
                    # 仅在平滑训练 Loss 创新低时保存，避免后期震荡/过拟合覆盖最优权重
                    if smoothed_train_loss < best_val_metric:
                        best_val_metric = smoothed_train_loss
                        _save_stage_checkpoint('smoothed_train_loss', smoothed_train_loss,
                                               {'smoothed_train_loss': smoothed_train_loss, 'step': global_step})
                        print(f"   ★ [{cfg.stage_name}] 平滑 Train Loss 创新低，模型已保存 -> {save_path}")
                    if smoothed_train_loss <= cfg.target_train_loss:
                        print(f"\n★ 达成实盘裸跑目标 Loss ({smoothed_train_loss:.4f} <= {cfg.target_train_loss})！模型已锁定 -> {save_path}")
                        break

            if global_step >= total_steps:
                break

        if (val_data_gpu is not None and patience_counter >= cfg.patience) or global_step >= total_steps:
            break
        if val_data_gpu is None and smoothed_train_loss <= cfg.target_train_loss:
            break

    print(f"\n>> [{cfg.stage_name}] 训练结束，权重保存在: {save_path}")
    return model


# ==============================================================================
# 6. 内置三模态标准仿真数据生成器 (自动生成 D1=322列, D2=642列, D3=778列 Parquet)
# ==============================================================================
def generate_mock_multi_stage_parquets_if_needed(base_dir: str, num_days: int = 64, m15_days: int = 8):
    """
    若本地无数据，一次性生成三套严格符合 10 象限 640 词表规范与因果律的仿真 Parquet 数据集：
      - d1_stk_322.parquet     (纯个股日线, len=322)
      - d2_idx_642.parquet     (个股在前+指数在后 同日交织, len=642)
      - d3_m15_778.parquet     (前56天[个股+指数] + 后8天[个股+指数+16根15m收盘], len=778)
    """
    os.makedirs(base_dir, exist_ok=True)
    d1_train = os.path.join(base_dir, "d1_stk_322.parquet")
    d2_train = os.path.join(base_dir, "d2_idx_642.parquet")
    d3_train = os.path.join(base_dir, "d3_m15_778.parquet")

    if os.path.exists(d1_train) and os.path.exists(d2_train) and os.path.exists(d3_train):
        return

    print(f"\n>> 正在生成符合 10 象限词表规范的三阶段仿真 Parquet 数据集 (int32 元数据 + int16 序列)...")
    rng = np.random.default_rng(2026)

    for split_suffix, n_samples in [("", 2048), ("_val", 512)]:
        # 1. 生成 64 天个股原始分箱 (1~63) 与 64 天指数原始分箱 (1~63)
        stk_c = rng.integers(15, 62, size=(n_samples, num_days), dtype=np.int16)
        stk_o = np.clip(stk_c + rng.integers(-5, 6, size=(n_samples, num_days)), 1, 63).astype(np.int16)
        stk_h = np.clip(np.maximum(stk_o, stk_c) + rng.integers(0, 7, size=(n_samples, num_days)), 1, 63).astype(np.int16)
        stk_l = np.clip(np.minimum(stk_o, stk_c) - rng.integers(0, 7, size=(n_samples, num_days)), 1, 63).astype(np.int16)

        idx_c = rng.integers(20, 48, size=(n_samples, num_days), dtype=np.int16)
        idx_o = np.clip(idx_c + rng.integers(-3, 4, size=(n_samples, num_days)), 1, 63).astype(np.int16)
        idx_h = np.clip(np.maximum(idx_o, idx_c) + rng.integers(0, 5, size=(n_samples, num_days)), 1, 63).astype(np.int16)
        idx_l = np.clip(np.minimum(idx_o, idx_c) - rng.integers(0, 5, size=(n_samples, num_days)), 1, 63).astype(np.int16)

        # 生成最近 8 天每天 16 根 15m 收盘分箱 (1~63)
        m15_c = rng.integers(18, 55, size=(n_samples, m15_days, 16), dtype=np.int16)

        # 注入合成规律决定明日个股 H (1~63)：
        # 规律: 今日个股阳线 (C > O) + 今日指数温和看涨 (idx_c > 32) + 最后一根 15m 尾盘抢筹 (m15_c[:, -1, -1] > 36)
        cond_stk = (stk_c[:, -1] > stk_o[:, -1]) & (stk_c[:, -1] < VocabSpec.LIMIT_UP_MIN_BIN)
        cond_idx = (idx_c[:, -1] >= 32)
        cond_m15 = (m15_c[:, -1, -1] >= 35)
        bullish = cond_stk & (cond_idx | cond_m15)

        next_h = np.where(
            bullish,
            rng.integers(VocabSpec.GT_2PCT_MIN_BIN, 63, size=n_samples),
            rng.integers(8, VocabSpec.GT_2PCT_MIN_BIN + 3, size=n_samples)
        ).astype(np.int16)

        # 元数据列: 强制 int32, id1 为 yyyymmdd
        meta_dict = {
            'id0': np.arange(600000, 600000 + n_samples, dtype=np.int32),
            'id1': np.full(n_samples, 20261020, dtype=np.int32),
            'id2': stk_c[:, -1].astype(np.int32)  # 记录今日收盘分箱作为辅助参考
        }

        # --- 组装 D1 (纯个股日线, len = 322) ---
        d1_seq = np.zeros((n_samples, num_days * 5 + 2), dtype=np.int16)
        for d in range(num_days):
            b = d * 5
            d1_seq[:, b] = VocabSpec.SEP_STK_ID      # 0
            d1_seq[:, b + 1] = stk_h[:, d]           # Group 0: 1~63
            d1_seq[:, b + 2] = stk_o[:, d] + 64      # Group 1: 65~127
            d1_seq[:, b + 3] = stk_l[:, d] + 128     # Group 2: 129~191
            d1_seq[:, b + 4] = stk_c[:, d] + 192     # Group 3: 193~255
        d1_seq[:, -2] = VocabSpec.SEP_STK_ID
        d1_seq[:, -1] = next_h

        # --- 组装 D2 (个股在前 + 指数在后 同日交织, len = 642) ---
        d2_seq = np.zeros((n_samples, num_days * 10 + 2), dtype=np.int16)
        for d in range(num_days):
            b = d * 10
            # 个股 5 tokens
            d2_seq[:, b:b + 5] = d1_seq[:, d * 5:(d + 1) * 5]
            # 指数 5 tokens (Group 4~7: 256~511)
            d2_seq[:, b + 5] = VocabSpec.SEP_IDX_ID  # 256
            d2_seq[:, b + 6] = idx_h[:, d] + 256
            d2_seq[:, b + 7] = idx_o[:, d] + 320
            d2_seq[:, b + 8] = idx_l[:, d] + 384
            d2_seq[:, b + 9] = idx_c[:, d] + 448
        d2_seq[:, -2] = VocabSpec.SEP_STK_ID
        d2_seq[:, -1] = next_h

        # --- 组装 D3 (前56天同D2 + 后8天同日嵌入15m线, len = 778) ---
        first_days = num_days - m15_days  # 56
        d3_len = first_days * 10 + m15_days * 27 + 2  # 560 + 216 + 2 = 778
        d3_seq = np.zeros((n_samples, d3_len), dtype=np.int16)
        # 前 56 天直接复制 D2 的前 560 列
        d3_seq[:, :first_days * 10] = d2_seq[:, :first_days * 10]
        # 后 8 天: 每天 [个股(5) + 指数(5) + 15m(17)] = 27 tokens
        for md in range(m15_days):
            d = first_days + md
            b = first_days * 10 + md * 27
            d3_seq[:, b:b + 10] = d2_seq[:, d * 10:(d + 1) * 10]
            d3_seq[:, b + 10] = VocabSpec.SEP_M15_ID  # 512
            d3_seq[:, b + 11:b + 27] = m15_c[:, md, :] + 512  # Group 8: 513~575
        d3_seq[:, -2] = VocabSpec.SEP_STK_ID
        d3_seq[:, -1] = next_h

        # 保存为 Parquet
        for name_prefix, seq_arr in [("d1_stk_322", d1_seq), ("d2_idx_642", d2_seq), ("d3_m15_778", d3_seq)]:
            fpath = os.path.join(base_dir, f"{name_prefix}{split_suffix}.parquet")
            seq_cols = {f"t{k}": seq_arr[:, k] for k in range(seq_arr.shape[1])}
            pd.DataFrame({**meta_dict, **seq_cols}).to_parquet(fpath, engine='pyarrow', index=False)

    print("   >> 三套标准 Parquet 数据集生成完毕！")


if __name__ == "__main__":
    # 1. 若本地无数据，自动生成 D1(322), D2(642), D3(778) 仿真数据用于验证流水线
    generate_mock_multi_stage_parquets_if_needed(cfg.base_dir)

    # 2. 执行当前文件配置的 Stage 训练
    run_stage_training(cfg)
