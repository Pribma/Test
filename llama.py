import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, TrainingArguments

# 1. NASTAVENÍ KOMPRESE (Slon do batohu)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 2. NAČTENÍ ZBRANĚ A MLÝNKU (Tokenizer)
model_id = "meta-llama/Llama-2-7b-hf" # Příklad, organizátoři dají přesný název
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

# 3. NASAZENÍ ADAPTÉRU (LoRA)
lora_config = LoraConfig(
    r=8, # Velikost adaptéru
    target_modules=["q_proj", "v_proj"], # Kam se adaptér napíchne
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. TRÉNINK PŘES AUTOMATIKU
args = TrainingArguments(
    output_dir="./vysledky",
    per_device_train_batch_size=2, # Extrémně malé, ať nepraskne paměť!
    num_train_epochs=1, # U LLM se často trénuje jen 1-3 epochy!
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tvuj_dataset, # Data, která sis připravil
    args=args,
)

trainer.train() # TOHLE ODPÁLÍ VÝCVIK!
