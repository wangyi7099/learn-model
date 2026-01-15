from peft import LoraConfig
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from trl.models import clone_chat_template
from datasets import load_dataset
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

# Configure model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(
    device
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_name)
# Setup chat template

model, tokenizer, added_tokens = clone_chat_template(
    model, tokenizer, "Qwen/Qwen3-0.6B")

rank_dimension = 6
lora_alpha = 8
lora_dropout = 0.05

peft_config = LoraConfig(
    r=rank_dimension,  # Rank dimension - typically between 4-32
    lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
    lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
    # Bias type for LoRA. the corresponding biases will be updated during training.
    bias="none",
    target_modules="all-linear",  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)


training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="no",
    eval_steps=50,
    max_length=384
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()
