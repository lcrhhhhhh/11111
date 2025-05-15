import os
from transformers import GPT2LMHeadModel, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from datasets import Dataset
import torch


def load_all_texts(data_dir="./data"):
    texts = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt"):
            path = os.path.join(data_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            except UnicodeDecodeError:
                with open(path, "r", encoding="gbk", errors="ignore") as f:
                    texts.append(f.read())
    return "\n".join(texts)

def merge_texts(data_dir="./data", output_file="merged_corpus.txt"):
    texts = load_all_texts(data_dir)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(texts)

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def generate_text(model, tokenizer, prompt="Hello", max_length=100):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.95)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


merge_texts("./data", "merged_corpus.txt")

# 步骤2：加载预训练模型
tokenizer = AutoTokenizer.from_pretrained("gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("gpt2-chinese-cluecorpussmall")
tokenizer.eos_token = tokenizer.pad_token
model.resize_token_embeddings(len(tokenizer))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


dataset = load_dataset()
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
    logging_dir='./logs',
    report_to="none",
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

print(generate_text(model, tokenizer, prompt="两柄木剑挥舞交斗，相互撞击", max_length=200))