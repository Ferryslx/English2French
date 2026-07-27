"""
Transformer Ablation Studies
Experiments:
  1. exp1_no_warmup — 移除 Noam warmup，恒定 lr=0.0003
  2. exp2_n1        — 层数 N=1
  3. exp3_d128      — d_model=128, d_ff=512, head=4
  5. exp5_dropout0  — dropout=0.0

每个实验独立的日志 + 损失 CSV + 模型保存。
exp0_baseline 为当前默认 Transformer 配置，作为消融实验的对比基线。
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
    EncoderDecoder, Encoder, EncoderLayer, Decoder, DecoderLayer,
    Embedding, PositionalEncoding, MultiHeadAttention, FeedForward, Generator
)

# ============================================================
# Hardcoded Paths
# ============================================================
EN_SP_PATH = 'C:/PythonProject/English2French/data/train.en.sp'
FR_SP_PATH = 'C:/PythonProject/English2French/data/train.fr.sp'
LOG_DIR = 'C:/PythonProject/English2French/log'
LOSS_DIR = 'C:/PythonProject/English2French/loss'
MODEL_DIR = 'C:/PythonProject/English2French/model'

# ============================================================
# Hyperparameters (defaults)
# ============================================================
VOCAB_SIZE = 6000
PAD = 0
BOS = 1
EOS = 2

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
# Transformer Builder (parameterized)
# ============================================================

def build_transformer(d_model, d_ff, head, n_layers, dropout):
    src_embed = Embedding(VOCAB_SIZE, d_model)
    src_pos = PositionalEncoding(d_model, dropout, max_len=MAX_LEN)
    tgt_embed = Embedding(VOCAB_SIZE, d_model)
    tgt_pos = PositionalEncoding(d_model, dropout, max_len=MAX_LEN)

    attn = MultiHeadAttention(d_model, head, dropout)
    ff = FeedForward(d_model, d_ff, dropout)
    enc_layer = EncoderLayer(d_model, attn, ff, dropout)
    encoder = Encoder(enc_layer, n_layers)

    dec_self_attn = MultiHeadAttention(d_model, head, dropout)
    dec_src_attn = MultiHeadAttention(d_model, head, dropout)
    dec_ff = FeedForward(d_model, d_ff, dropout)
    dec_layer = DecoderLayer(d_model, dec_self_attn, dec_src_attn, dec_ff, dropout)
    decoder = Decoder(dec_layer, n_layers)

    generator = Generator(VOCAB_SIZE, d_model)
    model = EncoderDecoder(src_embed, src_pos, encoder, tgt_embed, tgt_pos, decoder, generator)

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

def train_model(model, loader, val_loader, criterion, optimizer, model_name,
                loss_fp, val_fp, scheduler=None, logger=None):
    model.to(DEVICE)
    model.train()

    running_loss = 0.0
    running_examples = 0
    step_counter = 0
    global_examples = 0
    writer = csv.writer(loss_fp)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_examples = 0

        pbar = tqdm(loader, desc=f'[{model_name}] Epoch {epoch}/{EPOCHS}',
                    ncols=100, unit='batch', leave=True)

        for src, tgt in pbar:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            bs = src.size(0)

            if scheduler:
                scheduler.zero_grad()
            else:
                optimizer.zero_grad()

            src_mask = make_src_mask(src)
            tgt_input = tgt[:, :-1]
            tgt_mask = make_tgt_mask(tgt_input)
            output = model(src, tgt_input, src_mask, tgt_mask)

            loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if scheduler:
                scheduler.step()
            else:
                optimizer.step()

            loss_val = loss.item()
            running_loss += loss_val * bs
            running_examples += bs
            epoch_loss += loss_val * bs
            epoch_examples += bs
            global_examples += bs

            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'avg': f'{running_loss / running_examples:.4f}' if running_examples > 0 else '0',
                'ex': global_examples
            })

            if running_examples >= LOG_INTERVAL:
                avg_loss = running_loss / running_examples
                step_counter += 1
                logger.info(
                    f'[{model_name}] Epoch {epoch}/{EPOCHS} | '
                    f'Global examples: {global_examples} | '
                    f'Avg Loss (last {int(running_examples)} ex): {avg_loss:.4f}'
                )
                writer.writerow([model_name, epoch, global_examples, f'{avg_loss:.4f}'])
                loss_fp.flush()
                running_loss = 0.0
                running_examples = 0

        # End of epoch flush
        if running_examples > 0:
            avg_loss = running_loss / running_examples
            step_counter += 1
            logger.info(
                f'[{model_name}] Epoch {epoch}/{EPOCHS} | '
                f'Global examples: {global_examples} | '
                f'Avg Loss (last {int(running_examples)} ex): {avg_loss:.4f}'
            )
            writer.writerow([model_name, epoch, global_examples, f'{avg_loss:.4f}'])
            loss_fp.flush()
            running_loss = 0.0
            running_examples = 0

        epoch_avg = epoch_loss / epoch_examples
        epoch_time = time.time() - epoch_start

        # Validation
        model.eval()
        val_writer = csv.writer(val_fp)
        val_run_loss = 0.0
        val_run_ex = 0
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
                val_run_loss += loss.item() * src.size(0)
                val_run_ex += src.size(0)
                val_total_loss += loss.item() * src.size(0)
                val_total_ex += src.size(0)

                if val_run_ex >= LOG_INTERVAL:
                    val_win = val_run_loss / val_run_ex
                    logger.info(
                        f'[{model_name}] Epoch {epoch} Val | '
                        f'Examples: {val_total_ex} | '
                        f'Avg Loss (last {int(val_run_ex)} ex): {val_win:.4f}'
                    )
                    val_writer.writerow([model_name, epoch, global_examples + val_total_ex, f'{val_win:.4f}'])
                    val_fp.flush()
                    val_run_loss = 0.0
                    val_run_ex = 0

        if val_run_ex > 0:
            val_win = val_run_loss / val_run_ex
            logger.info(
                f'[{model_name}] Epoch {epoch} Val | '
                f'Examples: {val_total_ex} | '
                f'Avg Loss (last {int(val_run_ex)} ex): {val_win:.4f}'
            )
            val_writer.writerow([model_name, epoch, global_examples + val_total_ex, f'{val_win:.4f}'])
            val_fp.flush()

        val_avg = val_total_loss / val_total_ex
        logger.info(
            f'[{model_name}] === Epoch {epoch} done | '
            f'Train Loss: {epoch_avg:.4f} | Val Loss: {val_avg:.4f} | '
            f'Time: {epoch_time:.1f}s ==='
        )
        tqdm.write(
            f'[{model_name}] Epoch {epoch}/{EPOCHS} — '
            f'Train: {epoch_avg:.4f} | Val: {val_avg:.4f} | Time: {epoch_time:.1f}s'
        )
        model.train()

        # Save checkpoint
        ckpt = f'{MODEL_DIR}/{model_name}_epoch{epoch}.pth'
        torch.save(model.state_dict(), ckpt)
        logger.info(f'[{model_name}] Model saved to {ckpt}')

    return model


# ============================================================
# Experiment Configs
# ============================================================

EXPERIMENTS = [
    {
        'name': 'exp1_no_warmup',
        'd_model': 256, 'd_ff': 1024, 'head': 4, 'n_layers': 4, 'dropout': 0.2,
        'use_warmup': False, 'lr': 0.0003,
        'desc': '移除 Noam warmup，恒定 lr=0.0003',
    },
    {
        'name': 'exp2_n1',
        'd_model': 256, 'd_ff': 1024, 'head': 4, 'n_layers': 1, 'dropout': 0.2,
        'use_warmup': True, 'lr': 0,
        'desc': '层数 N=1',
    },
    {
        'name': 'exp3_d128',
        'd_model': 128, 'd_ff': 512, 'head': 4, 'n_layers': 4, 'dropout': 0.2,
        'use_warmup': True, 'lr': 0,
        'desc': 'd_model=128, d_ff=512, head=4',
    },
    {
        'name': 'exp5_dropout0',
        'd_model': 256, 'd_ff': 1024, 'head': 4, 'n_layers': 4, 'dropout': 0.0,
        'use_warmup': True, 'lr': 0,
        'desc': 'dropout=0.0',
    },
]

# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(LOSS_DIR, exist_ok=True)
    criterion = nn.NLLLoss(ignore_index=PAD)

    tqdm.write('=' * 60)
    tqdm.write('Transformer Ablation Studies')
    tqdm.write(f'Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}')
    tqdm.write('=' * 60)

    # ---- Run each experiment ----
    for cfg in EXPERIMENTS:
        name = cfg['name']
        tqdm.write(f'\n--- {name}: {cfg["desc"]} ---')

        # Logger for this experiment
        logger = Logger('C:/PythonProject/English2French', f'ablation_{name}').get_logger()
        logger.info(f'Starting experiment: {name} — {cfg["desc"]}')
        logger.info(f'Config: d_model={cfg["d_model"]}, d_ff={cfg["d_ff"]}, '
                    f'head={cfg["head"]}, n_layers={cfg["n_layers"]}, '
                    f'dropout={cfg["dropout"]}, warmup={cfg["use_warmup"]}')

        # Build model
        model = build_transformer(cfg['d_model'], cfg['d_ff'], cfg['head'],
                                  cfg['n_layers'], cfg['dropout'])
        n_params = sum(p.numel() for p in model.parameters())
        tqdm.write(f'  Params: {n_params:,}')
        logger.info(f'Params: {n_params:,}')

        # Optimizer
        if cfg['use_warmup']:
            optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
            scheduler = NoamScheduler(optimizer, cfg['d_model'], warmup_steps=4000)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                         betas=(0.9, 0.98), eps=1e-9)
            scheduler = None

        # CSVs
        train_csv = f'{LOSS_DIR}/{name}_train.csv'
        val_csv = f'{LOSS_DIR}/{name}_val.csv'
        with open(train_csv, 'w', newline='', encoding='utf-8') as tf, \
             open(val_csv, 'w', newline='', encoding='utf-8') as vf:
            csv.writer(tf).writerow(['model', 'epoch', 'step', 'loss'])
            csv.writer(vf).writerow(['model', 'epoch', 'step', 'loss'])
            train_model(model, train_loader, val_loader, criterion, optimizer,
                        name, tf, vf, scheduler=scheduler, logger=logger)

        tqdm.write(f'  Train CSV: {train_csv}')
        tqdm.write(f'  Val CSV:   {val_csv}')
        tqdm.write(f'  Done: {name}')

    tqdm.write('\n' + '=' * 60)
    tqdm.write('All ablation experiments complete!')
    tqdm.write(f'Models saved to: {MODEL_DIR}')
    tqdm.write(f'Loss CSVs in:   {LOSS_DIR}')
    tqdm.write(f'Logs in:        {LOG_DIR}')
    tqdm.write('=' * 60)


if __name__ == '__main__':
    main()
