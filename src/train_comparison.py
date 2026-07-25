"""
English-French Translation: Transformer vs GRU+Attention
- Uses existing EncoderDecoder from component/transformer_complete.py
- Defines GRU+Attention as a baseline control group
- Logs every 1000 examples, saves losses to CSV, saves final models
"""
import csv
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from utils.log import Logger

from component.transformer_complete import (
    EncoderDecoder, Encoder, EncoderLayer, Decoder, DecoderLayer,
    Embedding, PositionalEncoding, MultiHeadAttention, FeedForward, Generator
)

# ============================================================
# Hardcoded Paths (no os.path.join)
# ============================================================
EN_SP_PATH = '../data/train.en.sp'
FR_SP_PATH = '../data/train.fr.sp'
SP_MODEL_PATH = '../model/spm.model'
LOSS_DIR = '../loss'
LOSS_CSV_PATH = '../loss/training_losses.csv'
VAL_LOSS_CSV_PATH = '../loss/validation_losses.csv'
MODEL_DIR = '../model'

# ============================================================
# Hyperparameters
# ============================================================
VOCAB_SIZE = 6000
PAD = 0
BOS = 1
EOS = 2

# Transformer
D_MODEL = 256
D_FF = 1024
HEAD = 4
N_LAYERS = 4
DROPOUT = 0.2

# GRU+Attention
EMBED_SIZE = 256
HIDDEN_SIZE = 256
GRU_LAYERS = 2

# Training
MAX_LEN = 20
BATCH_SIZE = 64
EPOCHS = 20
LOG_INTERVAL = 1000  # log average loss every 1000 examples
VAL_SIZE = 1024 * 5  # validation set size

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Logging Setup (custom Logger from utils/log.py)
# ============================================================
logger = Logger('../', 'English2French').get_logger()

# ============================================================
# Data Loading
# ============================================================

def load_data(en_path, fr_path, max_len=MAX_LEN):
    """Load tokenized parallel data, add BOS/EOS, pad to max_len."""
    en_sentences, fr_sentences = [], []

    with open(en_path, 'r', encoding='utf-8') as f_en, \
         open(fr_path, 'r', encoding='utf-8') as f_fr:
        for en_line, fr_line in zip(f_en, f_fr):
            en_tokens = [int(x) for x in en_line.strip().split() if x]
            fr_tokens = [int(x) for x in fr_line.strip().split() if x]

            en_tokens = [BOS] + en_tokens[:max_len - 2] + [EOS]
            fr_tokens = [BOS] + fr_tokens[:max_len - 2] + [EOS]

            en_tokens = en_tokens + [PAD] * (max_len - len(en_tokens))
            fr_tokens = fr_tokens + [PAD] * (max_len - len(fr_tokens))

            en_sentences.append(en_tokens)
            fr_sentences.append(fr_tokens)

    en_tensor = torch.tensor(en_sentences, dtype=torch.long)
    fr_tensor = torch.tensor(fr_sentences, dtype=torch.long)
    return en_tensor, fr_tensor


logger.info('Loading data...')
en_data, fr_data = load_data(EN_SP_PATH, FR_SP_PATH)
dataset = data.TensorDataset(en_data, fr_data)

train_dataset, val_dataset = data.random_split(
    dataset, [len(dataset) - VAL_SIZE, VAL_SIZE]
)
train_dataloader = data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_dataloader = data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)

logger.info(
    f'Loaded {len(dataset)} sentence pairs '
    f'(train: {len(train_dataset)}, val: {len(val_dataset)}), '
    f'batch size {BATCH_SIZE}, {len(train_dataloader)} batches per epoch'
)

# ============================================================
# GRU + Attention Model (Baseline Control Group)
# ============================================================

class BahdanauAttention(nn.Module):
    """Bahdanau additive attention mechanism."""
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        decoder_hidden = decoder_hidden.unsqueeze(1)
        score = self.Va(torch.tanh(
            self.Wa(encoder_outputs) + self.Ua(decoder_hidden)
        )).squeeze(-1)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(score, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_size * 2 + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs, src_mask=None):
        embedded = self.dropout(self.embedding(x))

        context, _ = self.attention(hidden[-1], encoder_outputs, src_mask)
        context = context.unsqueeze(1)

        gru_input = torch.cat([embedded, context], dim=-1)
        output, hidden = self.gru(gru_input, hidden)

        output = output.squeeze(1)
        embedded_sq = embedded.squeeze(1)
        context_sq = context.squeeze(1)

        logit = self.fc_out(torch.cat([output, context_sq, embedded_sq], dim=-1))
        return F.log_softmax(logit, dim=-1), hidden


class GRUAttention(nn.Module):
    """GRU encoder-decoder with Bahdanau attention — baseline control group."""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.encoder = GRUEncoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = GRUDecoder(vocab_size, embed_size, hidden_size, num_layers, dropout)

    def forward(self, src, tgt_input, src_mask=None):
        encoder_outputs, hidden = self.encoder(src)
        tgt_len = tgt_input.size(1)

        logits = []
        dec_input = tgt_input[:, 0:1]
        for t in range(tgt_len):
            logit, hidden = self.decoder(dec_input, hidden, encoder_outputs, src_mask)
            logits.append(logit)
            if t < tgt_len - 1:
                dec_input = tgt_input[:, t + 1:t + 2]

        return torch.stack(logits, dim=1)


# ============================================================
# Transformer Builder
# ============================================================

def build_transformer():
    src_embed = Embedding(VOCAB_SIZE, D_MODEL)
    src_pos = PositionalEncoding(D_MODEL, DROPOUT, max_len=MAX_LEN)
    tgt_embed = Embedding(VOCAB_SIZE, D_MODEL)
    tgt_pos = PositionalEncoding(D_MODEL, DROPOUT, max_len=MAX_LEN)

    attn = MultiHeadAttention(D_MODEL, HEAD, DROPOUT)
    ff = FeedForward(D_MODEL, D_FF, DROPOUT)
    encoder_layer = EncoderLayer(D_MODEL, attn, ff, DROPOUT)
    encoder = Encoder(encoder_layer, N_LAYERS)

    dec_self_attn = MultiHeadAttention(D_MODEL, HEAD, DROPOUT)
    dec_src_attn = MultiHeadAttention(D_MODEL, HEAD, DROPOUT)
    dec_ff = FeedForward(D_MODEL, D_FF, DROPOUT)
    decoder_layer = DecoderLayer(D_MODEL, dec_self_attn, dec_src_attn, dec_ff, DROPOUT)
    decoder = Decoder(decoder_layer, N_LAYERS)

    generator = Generator(VOCAB_SIZE, D_MODEL)
    model = EncoderDecoder(src_embed, src_pos, encoder, tgt_embed, tgt_pos, decoder, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# ============================================================
# Mask Creation Helpers
# ============================================================

def make_src_mask(src):
    return (src != PAD).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt):
    tgt_pad = (tgt != PAD).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    causal = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
    return tgt_pad & causal


# ============================================================
# Training Function
# ============================================================

class NoamScheduler:
    """Transformer LR scheduler from 'Attention Is All You Need'.
    LR = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
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


def train_model(model, dataloader, val_dataloader, criterion, optimizer, model_name,
                loss_csv_fp, val_loss_csv_fp, scheduler=None):
    """
    Generic training loop for both Transformer and GRU+Attention.
    Logs average loss every LOG_INTERVAL examples and appends to CSV.
    """
    model.to(DEVICE)
    model.train()

    running_loss = 0.0
    running_examples = 0
    step_counter = 0
    global_examples = 0

    writer = csv.writer(loss_csv_fp)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_examples = 0
        batch_count = 0

        progress_bar = tqdm(
            dataloader,
            desc=f'[{model_name}] Epoch {epoch}/{EPOCHS}',
            ncols=100,
            unit='batch',
            leave=True
        )

        for src, tgt in progress_bar:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            batch_size = src.size(0)

            if model_name == 'transformer':
                scheduler.zero_grad()
            else:
                optimizer.zero_grad()

            if model_name == 'transformer':
                src_mask = make_src_mask(src)
                tgt_input = tgt[:, :-1]
                tgt_mask = make_tgt_mask(tgt_input)
                output = model(src, tgt_input, src_mask, tgt_mask)
            else:
                src_mask = (src != PAD).float()
                output = model(src, tgt[:, :-1], src_mask)

            loss = criterion(
                output.reshape(-1, VOCAB_SIZE),
                tgt[:, 1:].reshape(-1)
            )
            loss.backward()
            if model_name == 'transformer':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scheduler.step()
            else:
                optimizer.step()

            loss_val = loss.item()
            running_loss += loss_val * batch_size
            running_examples += batch_size
            epoch_loss += loss_val * batch_size
            epoch_examples += batch_size
            global_examples += batch_size
            batch_count += 1

            running_avg = running_loss / running_examples if running_examples > 0 else 0.0
            progress_bar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'avg': f'{running_avg:.4f}',
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
                loss_csv_fp.flush()
                running_loss = 0.0
                running_examples = 0

        # End of epoch
        if running_examples > 0:
            avg_loss = running_loss / running_examples
            step_counter += 1
            logger.info(
                f'[{model_name}] Epoch {epoch}/{EPOCHS} | '
                f'Global examples: {global_examples} | '
                f'Avg Loss (last {int(running_examples)} ex): {avg_loss:.4f}'
            )
            writer.writerow([model_name, epoch, global_examples, f'{avg_loss:.4f}'])
            loss_csv_fp.flush()
            running_loss = 0.0
            running_examples = 0

        epoch_avg = epoch_loss / epoch_examples
        epoch_time = time.time() - epoch_start

        # Validation (windowed, every LOG_INTERVAL examples, same granularity as training)
        model.eval()
        val_running_loss = 0.0
        val_running_examples = 0
        val_total_loss = 0.0
        val_total_examples = 0
        val_step = 0
        with torch.no_grad():
            for src, tgt in val_dataloader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                if model_name == 'transformer':
                    src_mask = make_src_mask(src)
                    tgt_input = tgt[:, :-1]
                    tgt_mask = make_tgt_mask(tgt_input)
                    output = model(src, tgt_input, src_mask, tgt_mask)
                else:
                    src_mask = (src != PAD).float()
                    output = model(src, tgt[:, :-1], src_mask)
                loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
                val_running_loss += loss.item() * src.size(0)
                val_running_examples += src.size(0)
                val_total_loss += loss.item() * src.size(0)
                val_total_examples += src.size(0)

                if val_running_examples >= LOG_INTERVAL:
                    val_step += 1
                    val_win_avg = val_running_loss / val_running_examples
                    logger.info(
                        f'[{model_name}] Epoch {epoch} Val | '
                        f'Examples: {val_total_examples} | '
                        f'Avg Loss (last {int(val_running_examples)} ex): {val_win_avg:.4f}'
                    )
                    val_writer = csv.writer(val_loss_csv_fp)
                    val_writer.writerow(
                        [f'{model_name}_val', epoch, global_examples + val_total_examples,
                         f'{val_win_avg:.4f}']
                    )
                    val_loss_csv_fp.flush()
                    val_running_loss = 0.0
                    val_running_examples = 0

        if val_running_examples > 0:
            val_win_avg = val_running_loss / val_running_examples
            logger.info(
                f'[{model_name}] Epoch {epoch} Val | '
                f'Examples: {val_total_examples} | '
                f'Avg Loss (last {int(val_running_examples)} ex): {val_win_avg:.4f}'
            )
            val_writer = csv.writer(val_loss_csv_fp)
            val_writer.writerow(
                [f'{model_name}_val', epoch, global_examples + val_total_examples,
                 f'{val_win_avg:.4f}']
            )
            val_loss_csv_fp.flush()

        val_avg_loss = val_total_loss / val_total_examples
        logger.info(
            f'[{model_name}] === Epoch {epoch} done | '
            f'Train Loss: {epoch_avg:.4f} | Val Loss: {val_avg_loss:.4f} | '
            f'Time: {epoch_time:.1f}s ==='
        )
        tqdm.write(
            f'[{model_name}] Epoch {epoch}/{EPOCHS} — '
            f'Train Loss: {epoch_avg:.4f} | Val Loss: {val_avg_loss:.4f} | '
            f'Time: {epoch_time:.1f}s'
        )

        model.train()

        save_path = f'{MODEL_DIR}/{model_name}_epoch{epoch}.pth'
        torch.save(model.state_dict(), save_path)
        logger.info(f'[{model_name}] Model saved to {save_path}')

    return model


# ============================================================
# Main
# ============================================================

def main():
    logger.info('=' * 60)
    logger.info('English-French Translation: Transformer vs GRU+Attention')
    logger.info(f'Device: {DEVICE} | Epochs: {EPOCHS} | MaxLen: {MAX_LEN} | '
                f'Batch: {BATCH_SIZE} | Log interval: {LOG_INTERVAL} examples')
    logger.info('=' * 60)

    os.makedirs(LOSS_DIR, exist_ok=True)

    tqdm.write('Building Transformer model...')
    logger.info('Building Transformer model...')
    transformer = build_transformer()
    tqdm.write(f'Transformer params: {sum(p.numel() for p in transformer.parameters()):,}')
    logger.info(f'Transformer params: {sum(p.numel() for p in transformer.parameters()):,}')

    tqdm.write('Building GRU+Attention model...')
    logger.info('Building GRU+Attention model...')
    gru_model = GRUAttention(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, GRU_LAYERS, DROPOUT)
    tqdm.write(f'GRU+Attention params: {sum(p.numel() for p in gru_model.parameters()):,}')
    logger.info(f'GRU+Attention params: {sum(p.numel() for p in gru_model.parameters()):,}')

    criterion = nn.NLLLoss(ignore_index=PAD)

    with open(LOSS_CSV_PATH, 'w', newline='', encoding='utf-8') as f, \
         open(VAL_LOSS_CSV_PATH, 'w', newline='', encoding='utf-8') as f_val:
        writer = csv.writer(f)
        writer.writerow(['model', 'epoch', 'step', 'loss'])
        val_writer = csv.writer(f_val)
        val_writer.writerow(['model', 'epoch', 'step', 'loss'])

        tqdm.write('\n' + '-' * 50)
        tqdm.write('Starting Transformer training...')
        tqdm.write('-' * 50)
        logger.info('Starting Transformer training...')

        transformer_optimizer = torch.optim.Adam(
            transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
        # NoamScheduler 在 __init__ 和每次 step() 中覆盖 lr，optimizer 的初始 lr 不用设
        transformer_scheduler = NoamScheduler(transformer_optimizer, D_MODEL, warmup_steps=4000)
        train_model(transformer, train_dataloader, val_dataloader, criterion, transformer_optimizer,
                    'transformer', f, f_val, scheduler=transformer_scheduler)

        tqdm.write('\n' + '-' * 50)
        tqdm.write('Starting GRU+Attention training...')
        tqdm.write('-' * 50)
        logger.info('Starting GRU+Attention training...')

        gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
        train_model(gru_model, train_dataloader, val_dataloader, criterion, gru_optimizer,
                    'gru', f, f_val)

    tqdm.write('\n' + '=' * 50)
    tqdm.write('Training complete!')
    tqdm.write(f'Train loss CSV: {LOSS_CSV_PATH}')
    tqdm.write(f'Val loss CSV:   {VAL_LOSS_CSV_PATH}')
    tqdm.write(f'Transformer final: {MODEL_DIR}/transformer_epoch{EPOCHS}.pth')
    tqdm.write(f'GRU final: {MODEL_DIR}/gru_epoch{EPOCHS}.pth')
    tqdm.write('=' * 50)
    logger.info('Training complete!')


if __name__ == '__main__':
    main()
