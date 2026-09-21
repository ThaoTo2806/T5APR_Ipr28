import os

# ============================================================
# DISABLE WANDB
# ============================================================

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from enum import Enum
from math import ceil
import gc

import torch

from datasets import (
    load_dataset,
    concatenate_datasets,
)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


from .configs import cache_dir, models_root


# ============================================================
# CONFIG
# ============================================================

ModelType = Enum(
    "ModelType",
    ["MULTI", "PYTHON", "JAVA", "C", "JAVASCRIPT"],
)

model_type = ModelType.MULTI

model_checkpoint = "Salesforce/codet5-small"

max_input_length = 512
max_target_length = 256

# Batch size PER GPU
train_batch_size = 8
eval_batch_size = 16

epochs = 1
lr = 1e-4

checkpoints_each_epoch = 5

# Streaming shuffle buffer.
# Bigger = better randomization but more RAM.
shuffle_buffer = 10000

# Number of examples processed by tokenizer at once.
tokenize_batch_size = 256


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("DEVICE")
print("=" * 70)

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():

    print("GPU count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(
            f"GPU {i}:",
            torch.cuda.get_device_name(i),
        )

    print(
        "Current GPU:",
        torch.cuda.current_device(),
    )

print()


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("LOADING CODET5")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint,
    cache_dir=cache_dir,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_checkpoint,
    cache_dir=cache_dir,
)

print("Model:", model_checkpoint)
print()


# ============================================================
# DATASETS
# ============================================================

dataset_names = {
    "Python": "h4iku/coconut_python2010_preprocessed",
    "Java": "h4iku/coconut_java2006_preprocessed",
    "JavaScript": "h4iku/coconut_javascript2010_preprocessed",
    "C": "h4iku/coconut_c2005_preprocessed",
}

if model_type is not ModelType.MULTI:

    dataset_names = {
        lang: name
        for lang, name in dataset_names.items()
        if lang.upper() == model_type.name
    }


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_function(prefix, examples):

    # --------------------------------------------------------
    # Clean removed code
    # --------------------------------------------------------

    rems = [
        ex.strip().replace(
            tokenizer.eos_token,
            tokenizer.unk_token,
        )
        for ex in examples["rem"]
    ]

    # --------------------------------------------------------
    # Clean added code
    # --------------------------------------------------------

    adds = [
        ex.strip().replace(
            tokenizer.eos_token,
            tokenizer.unk_token,
        )
        for ex in examples["add"]
    ]

    # --------------------------------------------------------
    # Clean context
    # --------------------------------------------------------

    contexts = [
        " ".join(ex.split()).replace(
            tokenizer.eos_token,
            tokenizer.unk_token,
        )
        for ex in examples["context"]
    ]

    # --------------------------------------------------------
    # Same input construction as original T5APR
    #
    # PREFIX REM :
    #
    # then append CONTEXT
    # --------------------------------------------------------

    inputs = [
        f"{prefix} {rem} :"
        for rem in rems
    ]

    inputs_contexts = [
        f"{src} {ctx}"
        for src, ctx in zip(inputs, contexts)
    ]

    # --------------------------------------------------------
    # Tokenize input
    # --------------------------------------------------------

    model_inputs = tokenizer(
        inputs_contexts,
        max_length=max_input_length,
        truncation=True,
        padding=False,
    )

    # --------------------------------------------------------
    # Tokenize target
    # --------------------------------------------------------

    labels = tokenizer(
        adds,
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# ============================================================
# FILTER + TOKENIZATION
# ============================================================
#
# IMPORTANT:
#
# We DON'T create:
#
#     filtered_dataset
#
# Instead:
#
#     streaming raw dataset
#             ↓
#         filter()
#             ↓
#         map(tokenize)
#
# Nothing containing millions of tokenized samples is stored
# on /kaggle/working.
# ============================================================

def make_streaming_dataset(prefix, dataset_name):

    print()
    print("=" * 70)
    print(f"LOADING STREAMING DATASET: {prefix}")
    print("=" * 70)

    # --------------------------------------------------------
    # streaming=True
    #
    # This is the key difference.
    #
    # Hugging Face does NOT materialize the entire dataset
    # into an Arrow file in /kaggle/working.
    # --------------------------------------------------------

    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
    )

    # --------------------------------------------------------
    # Shuffle stream
    # --------------------------------------------------------

    dataset = dataset.shuffle(
        seed=42,
        buffer_size=shuffle_buffer,
    )

    # --------------------------------------------------------
    # FILTER
    #
    # Only tokenize rem/add for checking length.
    #
    # We intentionally DON'T tokenize context here.
    # --------------------------------------------------------

    def filter_function(example):

        rem = example["rem"].strip()
        add = example["add"].strip()

        rem = rem.replace(
            tokenizer.eos_token,
            tokenizer.unk_token,
        )

        add = add.replace(
            tokenizer.eos_token,
            tokenizer.unk_token,
        )

        rem_ids = tokenizer(
            f"{prefix} {rem} :",
            add_special_tokens=True,
            truncation=False,
        ).input_ids

        add_ids = tokenizer(
            add,
            add_special_tokens=True,
            truncation=False,
        ).input_ids

        return (
            len(rem_ids) <= max_input_length
            and 2 < len(add_ids) <= max_target_length
        )

    dataset = dataset.filter(
        filter_function,
    )

    # --------------------------------------------------------
    # TOKENIZE LAZILY
    #
    # Only a small batch is held in RAM at a time.
    # --------------------------------------------------------

    dataset = dataset.map(
        lambda examples: preprocess_function(
            prefix,
            examples,
        ),
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=[
            "rem",
            "add",
            "context",
        ],
    )

    return dataset


# ============================================================
# BUILD MULTI-LANGUAGE STREAM
# ============================================================

print()
print("=" * 70)
print("BUILDING STREAMING DATASET")
print("=" * 70)

streaming_datasets = []

for prefix, dataset_name in dataset_names.items():

    ds = make_streaming_dataset(
        prefix,
        dataset_name,
    )

    streaming_datasets.append(ds)

    print(f"Streaming dataset ready: {prefix}")


# ============================================================
# CONCATENATE STREAMS
# ============================================================

print()
print("=" * 70)
print("CONCATENATING STREAMING DATASETS")
print("=" * 70)

#
# Unlike concatenate_datasets() on normal Arrow datasets,
# this does NOT materialize all samples on disk.
#
concatenated_dataset = concatenate_datasets(
    streaming_datasets,
)

print("Streaming dataset created.")
print("Features:", concatenated_dataset.features)
print()


# ============================================================
# IMPORTANT:
# ============================================================
#
# Streaming datasets don't know their final filtered length
# without scanning the whole dataset.
#
# We therefore use the filtered counts obtained previously
# from your datasets.
#
# Current counts from your previous successful run:
#
# Python       264842
# Java        1009268
# JavaScript   463027
# C            ~584k
#
# We use the exact total from your previous run:
#
#              2,324,030
#
# If you change the filtering logic, update this number.
# ============================================================

KNOWN_FILTERED_SAMPLES = {
    "Python": 264842,
    "Java": 1009268,
    "JavaScript": 463027,
    "C": 586893,
}

if model_type is ModelType.MULTI:

    estimated_samples = sum(
        KNOWN_FILTERED_SAMPLES.values()
    )

else:

    prefix = model_type.name.capitalize()

    estimated_samples = KNOWN_FILTERED_SAMPLES[
        prefix
    ]


print("=" * 70)
print("DATASET SIZE")
print("=" * 70)

print(
    "Estimated filtered samples:",
    estimated_samples,
)

print(
    "NOTE: This number is used only to calculate training steps."
)

print()


# ============================================================
# TRAINING
# ============================================================

model_name = model_checkpoint.split("/")[-1]

output_dir = (
    models_root
    / f"{model_name}-t5apr-{model_type.name.lower()}"
)


# ------------------------------------------------------------
# WORLD SIZE
# ------------------------------------------------------------

world_size = int(
    os.environ.get(
        "WORLD_SIZE",
        "1",
    )
)

print("=" * 70)
print("DISTRIBUTED TRAINING")
print("=" * 70)

print("WORLD_SIZE:", world_size)

if world_size > 1:
    print("Multi-GPU training enabled.")
else:
    print("Single-GPU training.")

print()


# ============================================================
# TRAINING STEPS
# ============================================================
#
# train_batch_size = batch per GPU
#
# Global batch:
#
#     batch_size × number_of_GPUs
#
# For 2 T4:
#
#     8 × 2 = 16
#
# Therefore:
#
#     optimizer steps ≈ samples / 16
#
# ============================================================

global_batch_size = (
    train_batch_size * world_size
)

steps_per_epoch = ceil(
    estimated_samples / global_batch_size
)

train_steps = (
    steps_per_epoch * epochs
)

save_steps = max(
    1,
    steps_per_epoch // checkpoints_each_epoch,
)


print("=" * 70)
print("TRAINING CONFIG")
print("=" * 70)

print("Model:", model_checkpoint)
print("Model type:", model_type.name)

print(
    "Estimated samples:",
    estimated_samples,
)

print(
    "Batch per GPU:",
    train_batch_size,
)

print(
    "Number of GPUs:",
    world_size,
)

print(
    "Global batch size:",
    global_batch_size,
)

print(
    "Epochs:",
    epochs,
)

print(
    "Steps per epoch:",
    steps_per_epoch,
)

print(
    "Total steps:",
    train_steps,
)

print(
    "Save steps:",
    save_steps,
)

print(
    "Learning rate:",
    lr,
)

print(
    "FP16:",
    True,
)

print(
    "Output:",
    output_dir,
)

print()


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

args = Seq2SeqTrainingArguments(

    output_dir=str(output_dir),

    learning_rate=lr,

    per_device_train_batch_size=train_batch_size,

    per_device_eval_batch_size=eval_batch_size,

    max_steps=train_steps,

    save_steps=save_steps,

    save_total_limit=5,

    predict_with_generate=True,

    fp16=True,

    lr_scheduler_type="constant",

    evaluation_strategy="no",

    logging_steps=100,

    dataloader_num_workers=0,#dataloader_num_workers=2,

    save_safetensors=True,

    report_to="none",

    # --------------------------------------------------------
    # Important for IterableDataset
    # --------------------------------------------------------

    remove_unused_columns=False,
    accelerator_config={
        "dispatch_batches": False,
    },

)


# ============================================================
# COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)


# ============================================================
# GENERATION
# ============================================================

model.config.max_length = max_target_length
model.config.min_length = 0
model.config.early_stopping = True
model.config.num_beams = 5


# ============================================================
# TRAINER
# ============================================================

trainer = Seq2SeqTrainer(

    model=model,

    args=args,

    train_dataset=concatenated_dataset,

    data_collator=data_collator,

    tokenizer=tokenizer,
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("START TRAINING")
print("=" * 70)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU",
)

print(
    "World size:",
    world_size,
)

print()


trainer.train(
    resume_from_checkpoint=False,
)


print()
print("=" * 70)
print("TRAINING FINISHED")
print("=" * 70)

print(
    "Model saved to:",
    output_dir,
)