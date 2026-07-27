"""
Exp4: Pre-LN vs Post-LN
对比 Pre-LN（当前默认）和 Post-LN（原实现）的训练收敛差异。
共用同一套超参数，仅 LN 位置不同。
"""
import csv
import os
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm

sys.path.append('C:/PythonProject/English2French')
from utils.log import Logger
from component.transformer_complete import (
    EncoderDecoder, Embedding, PositionalEncoding,
    MultiHeadAttention, FeedForward, Generator,
    LayerNorm, clones, attention
)

# ============================================================
# Paths
# ============================================================
EN_SP_PATH = 'C:/PythonProject/English2French/data/train.en.sp'
FR_SP_PATH = 'C:/PythonProject/English2French/data/train.fr.sp'
LOG_DIR = 'C:/PythonProject/English2French/log'
LOSS_DIR = 'C:/PythonProject/English2French/loss'
MODEL_DIR = 'C:/PythonProject/English2French/model'

# ============================================================
# Hyperparameters
# ============================================================
VOCAB_SIZE = 6000
PAD = 0
BOS = 1
EOS = 2
D_MODEL = 256
D_FF = 1024
HEAD = 4
N_LAYERS = 4
DROPOUT = 0.2
MAX_LEN = 20
BATCH_SIZE = 64
EPOCHS = 20
LOG_INTERVAL = 1000
VAL_SIZE = 1024 * 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Data Loading
# ============================================================

def load_data(en_path, fr_path, max_len=MAX_LEN):
    en_sentences, fr_sentences = [], []
    with open(en_path, 'r', encoding='utf-8') as f_en, \
         open(fr_path, 'r', encoding='utf-8') as f_fr:
        for en_line, fr_line in zip(f_en, f_fr):
            en_t = [int(x) for x in en_line.strip().split() if x]
            fr_t = [int(x) for x in fr_line.strip().split() if x]
            en_t = [BOS] + en_t[:max_len - 2] + [EOS]
            fr_t = [BOS] + fr_t[:max_len - 2] + [EOS]
            en_t = en_t + [PAD] * (max_len - len(en_t))
            fr_t = fr_t + [PAD] * (max_len - len(fr_t))
            en_sentences.append(en_t)
            fr_sentences.append(fr_t)
    return torch.tensor(en_sentences, dtype=torch.long), \
           torch.tensor(fr_sentences, dtype=torch.long)


print('Loading data...')
en_data, fr_data = load_data(EN_SP_PATH, FR_SP_PATH)
dataset = data.TensorDataset(en_data, fr_data)
train_dataset, val_dataset = data.random_split(
    dataset, [len(dataset) - VAL_SIZE, VAL_SIZE]
)
train_loader = data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)
print(f'Loaded {len(dataset)} pairs (train: {len(train_dataset)}, val: {len(val_dataset)})')

# ============================================================
# Post-LN Components (for ablation comparison)
# ============================================================

class SublayerConnectionPostLN(nn.Module):
    """Post-LN: norm(x + dropout(sublayer(x)))。"""
    def __init__(self, d_model, dropout=DROPOUT):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayerPostLN(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnectionPostLN(d_model, dropout), 2)

    def forward(self, x, mask=None):
        if mask is not None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer[1](x, lambda x: self.feed_forward(x))
        return x


class EncoderPostLN(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.self_attn.linears[0].out_features)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayerPostLN(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnectionPostLN(d_model, dropout), 3)

    def forward(self, x, encoder_output, source_mask=None, target_mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, encoder_output, encoder_output, source_mask))
        x = self.sublayer[2](x, lambda x: self.feed_forward(x))
        return x


class DecoderPostLN(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.self_attn.linears[0].out_features)

    def forward(self, x, encoder_output, source_mask=None, target_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)


# ============================================================
# Build Functions
# ============================================================

def build_preln():
    """Pre-LN Transformer — 当前默认，从 component 直接导入。"""
    from component.transformer_complete import (
        Encoder as EncoderPreLN, EncoderLayer as EncoderLayerPreLN,
        Decoder as DecoderPreLN, DecoderLayer as DecoderLayerPreLN,
        SublayerConnection as SublayerConnectionPreLN
    )
    src_embed = Embedding(VOCAB_SIZE, D_MODEL)
    src_pos = PositionalEncoding(D_MODEL, DROPOUT, max_len=MAX_LEN)
    tgt_embed = Embedding(VOCAB_SIZE, D_MODEL)
    tgt_pos = PositionalEncoding(D_MODEL, DROPOUT, max_len=MAX_LEN)

    attn = MultiHeadAttention(D_MODEL, HEAD, DROPOUT)
    ff = FeedForward(D_MODEL, D_FF, DROPOUT)
    enc_layer = EncoderLayerPreLN(D_MODEL, attn, ff, DROPOUT)
    encoder = EncoderPreLN(enc_layer, N_LAYERS)

    dec_self = MultiHeadAttention(D_MODEL, HEAD, DROPOUT)
    dec_src = MultiHeadAttention(D_MODEL, HEAD, DROPOUT)
    dec_ff = FeedForward(D_MODEL, D_FF, DROPOUT)
    dec_layer = DecoderLayerPreLN(D_MODEL, dec_self, dec_src, dec_ff, DROPOUT)
    decoder = DecoderPreLN(dec_layer, N_LAYERS)

    model = EncoderDecoder(src_embed, src_pos, encoder, tgt_embed, tgt_pos, decoder,
                           Generator(VOCAB_SIZE, D_MODEL))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def build_postln():
    """Post-LN Transformer — 使用本文件定义的 Post-LN 组件。"""
    src_embed = Embedding(VOCAB_SIZE, D_MODEL)
    src_pos = PositionalEncoding(D_MODEL, DROPOUT, max_len=MAX_LEN)
    tgt_embed = Embedding(VOCAB_SIZE, D_MODEL)
    tgt_pos = PositionalEncoding(D_MODEL, DROPOUT, max_len=MAX_LEN)

    attn = MultiHeadAttention(D_MODEL, HEAD, DROPOUT)
    ff = FeedForward(D_MODEL, D_FF, DROPOUT)
    enc_layer = EncoderLayerPostLN(D_MODEL, attn, ff, DROPOUT)
    encoder = EncoderPostLN(enc_layer, N_LAYERS)

    dec_self = MultiHeadAttention(D_MODEL, HEAD, DROPOUT)
    dec_src = MultiHeadAttention(D_MODEL, HEAD, DROPOUT)
    dec_ff = FeedForward(D_MODEL, D_FF, DROPOUT)
    dec_layer = DecoderLayerPostLN(D_MODEL, dec_self, dec_src, dec_ff, DROPOUT)
    decoder = DecoderPostLN(dec_layer, N_LAYERS)

    model = EncoderDecoder(src_embed, src_pos, encoder, tgt_embed, tgt_pos, decoder,
                           Generator(VOCAB_SIZE, D_MODEL))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ============================================================
# Mask Helpers
# ============================================================

def make_src_mask(src):
    return (src != PAD).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt):
    tgt_pad = (tgt != PAD).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    causal = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
    return tgt_pad & causal


# ============================================================
# Noam Scheduler
# ============================================================

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0
        self._set_lr()

    def _set_lr(self):
        step = max(self._step, 1)
        lr = self.d_model ** -0.5 * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def step(self):
        self.optimizer.step()
        self._step += 1
        self._set_lr()

    def zero_grad(self):
        self.optimizer.zero_grad()


# ============================================================
# Training Function
# ============================================================

def train_one(model, name, logger, train_csv, val_csv):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, D_MODEL, warmup_steps=4000)
    criterion = nn.NLLLoss(ignore_index=PAD)

    with open(train_csv, 'w', newline='', encoding='utf-8') as tf, \
         open(val_csv, 'w', newline='', encoding='utf-8') as vf:
        csv.writer(tf).writerow(['model', 'epoch', 'step', 'loss'])
        csv.writer(vf).writerow(['model', 'epoch', 'step', 'loss'])

        model.train()
        global_examples = 0

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_examples = 0
            running_loss = 0.0
            running_examples = 0

            pbar = tqdm(train_loader, desc=f'[{name}] Epoch {epoch}/{EPOCHS}',
                        ncols=100, unit='batch', leave=True)

            for src, tgt in pbar:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                bs = src.size(0)

                scheduler.zero_grad()
                src_mask = make_src_mask(src)
                tgt_input = tgt[:, :-1]
                tgt_mask = make_tgt_mask(tgt_input)
                output = model(src, tgt_input, src_mask, tgt_mask)

                loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scheduler.step()

                loss_val = loss.item()
                running_loss += loss_val * bs
                running_examples += bs
                epoch_loss += loss_val * bs
                epoch_examples += bs
                global_examples += bs

                pbar.set_postfix({
                    'loss': f'{loss_val:.4f}',
                    'avg': f'{running_loss / running_examples:.4f}',
                    'ex': global_examples
                })

                if running_examples >= LOG_INTERVAL:
                    avg_loss = running_loss / running_examples
                    logger.info(f'[{name}] Epoch {epoch}/{EPOCHS} | '
                                f'Global: {global_examples} | '
                                f'Avg Loss (last {int(running_examples)}): {avg_loss:.4f}')
                    csv.writer(tf).writerow([name, epoch, global_examples, f'{avg_loss:.4f}'])
                    tf.flush()
                    running_loss = 0.0
                    running_examples = 0

            # Flush remaining
            if running_examples > 0:
                avg_loss = running_loss / running_examples
                logger.info(f'[{name}] Epoch {epoch}/{EPOCHS} | '
                            f'Global: {global_examples} | '
                            f'Avg Loss (last {int(running_examples)}): {avg_loss:.4f}')
                csv.writer(tf).writerow([name, epoch, global_examples, f'{avg_loss:.4f}'])
                tf.flush()

            epoch_avg = epoch_loss / epoch_examples
            epoch_time = time.time() - epoch_start

            # Validation
            model.eval()
            val_win_loss = 0.0
            val_win_ex = 0
            val_total_loss = 0.0
            val_total_ex = 0
            with torch.no_grad():
                for src, tgt in val_loader:
                    src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                    src_mask = make_src_mask(src)
                    tgt_input = tgt[:, :-1]
                    tgt_mask = make_tgt_mask(tgt_input)
                    output = model(src, tgt_input, src_mask, tgt_mask)
                    loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
                    val_win_loss += loss.item() * src.size(0)
                    val_win_ex += src.size(0)
                    val_total_loss += loss.item() * src.size(0)
                    val_total_ex += src.size(0)

                    if val_win_ex >= LOG_INTERVAL:
                        val_avg = val_win_loss / val_win_ex
                        logger.info(f'[{name}] Epoch {epoch} Val | '
                                    f'Examples: {val_total_ex} | Avg: {val_avg:.4f}')
                        csv.writer(vf).writerow(
                            [name, epoch, global_examples + val_total_ex, f'{val_avg:.4f}']
                        )
                        vf.flush()
                        val_win_loss = 0.0
                        val_win_ex = 0

            if val_win_ex > 0:
                val_avg = val_win_loss / val_win_ex
                logger.info(f'[{name}] Epoch {epoch} Val | '
                            f'Examples: {val_total_ex} | Avg: {val_avg:.4f}')
                csv.writer(vf).writerow(
                    [name, epoch, global_examples + val_total_ex, f'{val_avg:.4f}']
                )
                vf.flush()

            val_total_avg = val_total_loss / val_total_ex
            logger.info(f'[{name}] === Epoch {epoch} done | '
                        f'Train: {epoch_avg:.4f} | Val: {val_total_avg:.4f} | '
                        f'Time: {epoch_time:.1f}s ===')
            tqdm.write(f'[{name}] Epoch {epoch}/{EPOCHS} — '
                       f'Train: {epoch_avg:.4f} | Val: {val_total_avg:.4f} | '
                       f'Time: {epoch_time:.1f}s')
            model.train()

            ckpt = f'{MODEL_DIR}/{name}_epoch{epoch}.pth'
            torch.save(model.state_dict(), ckpt)
            logger.info(f'[{name}] Saved to {ckpt}')


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(LOSS_DIR, exist_ok=True)

    tqdm.write('=' * 60)
    tqdm.write('Exp4: Pre-LN vs Post-LN')
    tqdm.write(f'Device: {DEVICE} | d_model={D_MODEL} | N={N_LAYERS} | '
               f'dropout={DROPOUT} | batch={BATCH_SIZE}')
    tqdm.write('=' * 60)

    for variant_name, build_fn in [('preln', build_preln), ('postln', build_postln)]:
        tqdm.write(f'\n--- Training {variant_name} ---')
        logger = Logger('C:/PythonProject/English2French', f'exp4_{variant_name}').get_logger()
        logger.info(f'Starting {variant_name} training')
        model = build_fn()
        n_params = sum(p.numel() for p in model.parameters())
        tqdm.write(f'  Params: {n_params:,}')
        logger.info(f'Params: {n_params:,}')

        train_one(
            model, f'exp4_{variant_name}', logger,
            f'{LOSS_DIR}/exp4_{variant_name}_train.csv',
            f'{LOSS_DIR}/exp4_{variant_name}_val.csv',
        )

    tqdm.write('\n' + '=' * 60)
    tqdm.write('Exp4 complete! Pre-LN vs Post-LN done.')
    tqdm.write(f'Train CSVs: {LOSS_DIR}/exp4_preln_train.csv / exp4_postln_train.csv')
    tqdm.write(f'Val CSVs:   {LOSS_DIR}/exp4_preln_val.csv / exp4_postln_val.csv')
    tqdm.write('=' * 60)


if __name__ == '__main__':
    main()
