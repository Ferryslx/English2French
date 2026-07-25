import sentencepiece as spm

in_file='../data/eng-fra-v2.txt'
en_file='../data/english.en'
fr_file='../data/french.fr'
merged_file='../data/train.en-fr'
model_prefix='../model/spm'

with open(in_file,'r',encoding='utf-8') as f_in,\
    open(en_file,'w',encoding='utf-8') as f_en,\
    open(fr_file,'w',encoding='utf-8') as f_fr:

    for line in f_in:
        line=line.strip()
        if not line or "\t" not in line:
            continue
        en,fr=line.split('\t',1)
        en=en.strip()
        fr=fr.strip()

        if en and fr:
            f_en.write(en + "\n")
            f_fr.write(fr + "\n")

with open(en_file,'r',encoding='utf-8') as f_en,\
    open(fr_file,'r',encoding='utf-8') as f_fr,\
    open(merged_file,'w',encoding='utf-8') as f_merged:

    for en_line,fr_line in zip(f_en,f_fr):
        if en_line and fr_line:
            f_merged.write(en_line.strip()+'\n')
            f_merged.write(fr_line.strip()+'\n')


#训练SentencePiece模型
spm.SentencePieceTrainer.train(
input=merged_file,
    model_prefix=model_prefix,
    vocab_size=6000,
    model_type="unigram",
    character_coverage=1.0,
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
    normalization_rule_name="nfkc"
)


sp=spm.SentencePieceProcessor('../model/spm.model')

#批量编码
def encode_file(in_path, out_path, sp_model):
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ids = sp_model.encode(line, out_type=int)
            fout.write(" ".join(map(str, ids)) + "\n")

encode_file(en_file, "../data/train.en.sp", sp)
encode_file(fr_file, "../data/train.fr.sp", sp)