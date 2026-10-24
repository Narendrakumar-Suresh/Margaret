from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.Metaspace()
tok.train(
    ["all_text.txt"],
    trainers.BpeTrainer(vocab_size=1024, special_tokens=["<pad>", "<unk>"]),
)
tok.save("tokenizer.json")