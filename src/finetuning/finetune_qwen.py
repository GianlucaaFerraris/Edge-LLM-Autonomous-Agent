"""
finetune_qwen.py

Fine-tuning de Qwen2.5-7B-Instruct con QLoRA usando Unsloth.
Entrena sobre los 3 modos: english_tutor, engineering, agent.
Exporta a GGUF Q4_K_M listo para Ollama.

════════════════════════════════════════════════════════════
OPCIÓN A — Google Colab T4 (recomendado, gratis)
════════════════════════════════════════════════════════════
1. Ir a https://colab.research.google.com
2. Entorno de ejecución → Cambiar tipo → T4 GPU
3. En la primera celda:

    !pip install unsloth[colab-new] -q
    !pip install -U xformers --index-url https://download.pytorch.org/whl/cu121 -q

4. Subir finetune_dataset_clean.jsonl con el botón de archivos (carpeta izquierda)
5. Subir este script y correr:

    !python finetune_qwen.py

6. Descargar ./output/gguf/model-unsloth.Q4_K_M.gguf al terminar

Tiempo estimado en T4: ~45-90 min para 850 ejemplos, 3 epochs

════════════════════════════════════════════════════════════
OPCIÓN B — PC local con GPU NVIDIA (8GB VRAM mínimo)
════════════════════════════════════════════════════════════
    pip install unsloth torch transformers datasets trl
    python finetune_qwen.py

════════════════════════════════════════════════════════════
POST FINE-TUNING — Registrar en Ollama (en la Radxa/Jetson)
════════════════════════════════════════════════════════════
    # Copiar el .gguf a la SBC y ejecutar:
    ollama create asistente -f Modelfile.assistant
    ollama run asistente

    # Verificar que funciona:
    python test_finetuned.py
"""

import argparse
import json
import os
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME   = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
DATASET_PATH = "finetune_dataset_clean.jsonl"
OUTPUT_DIR   = "./output"
MAX_SEQ_LEN  = 2048

# LoRA — rank 16 es buen balance calidad/velocidad para 7B
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

# Entrenamiento
BATCH_SIZE   = 2      # bajar a 1 si hay OOM en Colab
GRAD_ACCUM   = 4      # batch efectivo = 2 * 4 = 8
EPOCHS       = 3
LR           = 2e-4
WARMUP_RATIO = 0.03


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cargar modelo base con Unsloth (4-bit para menor VRAM)
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    print("[1/5] Cargando modelo base en 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LEN,
        dtype          = None,     # auto: float16 en GPU, float32 en CPU
        load_in_4bit   = True,     # QLoRA — ~5GB VRAM
    )

    # Aplicar LoRA a las capas de atención y FFN
    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_RANK,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",   # atención
            "gate_proj", "up_proj", "down_proj",        # FFN
        ],
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",        # ahorra ~30% VRAM
        random_state   = 42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros entrenables: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cargar y formatear dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(path: str, tokenizer) -> tuple:
    print(f"[2/5] Cargando dataset desde {path}...")

    with open(path, "r", encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]

    print(f"  Total ejemplos: {len(raw)}")

    # Distribución por modo (inferida del system prompt)
    modes = {}
    for ex in raw:
        sp = ex["messages"][0]["content"]
        if "spoken English" in sp or "English tutor" in sp:
            m = "english_tutor"
        elif "retired engineer" in sp or "brilliant" in sp:
            m = "engineering"
        elif "TOOL_CALL" in sp or "herramientas" in sp:
            m = "agent"
        else:
            m = "unknown"
        modes[m] = modes.get(m, 0) + 1
    print(f"  Distribución: {modes}")

    # Convertir al formato ChatML de Qwen2.5
    # <|im_start|>system\n...<|im_end|>\n
    # <|im_start|>user\n...<|im_end|>\n
    # <|im_start|>assistant\n...<|im_end|>\n
    def to_chatml(example: dict) -> str:
        text = ""
        for msg in example["messages"]:
            text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        return text

    formatted = [{"text": to_chatml(ex)} for ex in raw]

    # Verificar longitud máxima — descartar si supera MAX_SEQ_LEN tokens
    too_long = 0
    filtered = []
    for ex in formatted:
        tokens = tokenizer(ex["text"], return_tensors="pt")["input_ids"].shape[1]
        if tokens <= MAX_SEQ_LEN:
            filtered.append(ex)
        else:
            too_long += 1
    if too_long:
        print(f"  ⚠ {too_long} ejemplos descartados por superar {MAX_SEQ_LEN} tokens")

    dataset = Dataset.from_list(filtered)
    split   = dataset.train_test_split(test_size=0.05, seed=42)
    print(f"  Train: {len(split['train'])} | Eval: {len(split['test'])}")
    return split["train"], split["test"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Configurar y lanzar entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
def train(model, tokenizer, train_ds, eval_ds):
    print("[3/5] Configurando entrenamiento...")

    args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        warmup_ratio                = WARMUP_RATIO,
        lr_scheduler_type           = "cosine",
        optim                       = "adamw_8bit",   # 8-bit optimizer, menos VRAM
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        eval_strategy               = "steps",
        eval_steps                  = 50,
        save_strategy               = "steps",
        save_steps                  = 100,
        save_total_limit            = 2,              # conservar solo los 2 mejores checkpoints
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = "none",         # sin wandb ni tensorboard
        seed                        = 42,
    )

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_ds,
        eval_dataset       = eval_ds,
        dataset_text_field = "text",
        max_seq_length     = MAX_SEQ_LEN,
        args               = args,
    )

    print("[4/5] Entrenando...")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu} | VRAM: {vram:.1f} GB")
    else:
        print("  ⚠ Sin GPU — el entrenamiento será muy lento en CPU")

    print(f"  Epochs: {EPOCHS} | Batch efectivo: {BATCH_SIZE * GRAD_ACCUM} | LR: {LR}")

    stats = trainer.train()

    print(f"\n  ✅ Loss final: {stats.training_loss:.4f}")
    print(f"  ✅ Tiempo:     {stats.metrics['train_runtime'] / 60:.1f} min")

    # Guardar adaptadores LoRA por separado (útil para merge posterior)
    lora_path = f"{OUTPUT_DIR}/lora_adapters"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"  ✅ Adaptadores LoRA guardados en {lora_path}/")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 5. Exportar a GGUF Q4_K_M (listo para Ollama)
# ─────────────────────────────────────────────────────────────────────────────
def export_gguf(model, tokenizer):
    print("[5/5] Exportando a GGUF Q4_K_M...")
    gguf_path = f"{OUTPUT_DIR}/gguf"

    # Q4_K_M = INT4 con grupos de cuantización mixta — mejor relación calidad/tamaño
    # Tamaño resultante: ~4.5 GB para 7B
    model.save_pretrained_gguf(
        gguf_path,
        tokenizer,
        quantization_method="q4_k_m",
    )

    # Buscar el archivo generado
    gguf_file = None
    for f in os.listdir(gguf_path):
        if f.endswith(".gguf"):
            gguf_file = f
            size_gb = os.path.getsize(os.path.join(gguf_path, f)) / 1e9
            print(f"  ✅ {gguf_path}/{f} — {size_gb:.2f} GB")
            break

    if not gguf_file:
        print(f"  ⚠ No se encontró .gguf en {gguf_path}/")

    print(f"""
══════════════════════════════════════════════════════════
  FINE-TUNING COMPLETADO
══════════════════════════════════════════════════════════

  Archivo GGUF:  {gguf_path}/{gguf_file or 'model-unsloth.Q4_K_M.gguf'}

  Próximos pasos (en la Radxa/Jetson):

  1. Copiar el .gguf a la SBC:
       scp {gguf_path}/{gguf_file or '*.gguf'} radxa@<IP>:~/models/

  2. Actualizar Modelfile.assistant (línea FROM):
       FROM ~/models/{gguf_file or 'model-unsloth.Q4_K_M.gguf'}

  3. Registrar en Ollama:
       ollama create asistente -f Modelfile.assistant

  4. Probar:
       python test_finetuned.py
       python assistant.py

══════════════════════════════════════════════════════════
""")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning de Qwen2.5-7B con QLoRA → GGUF Q4_K_M"
    )
    parser.add_argument("--dataset", default=DATASET_PATH,
                        help=f"Ruta al JSONL de entrenamiento (default: {DATASET_PATH})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Número de epochs (default: {EPOCHS})")
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE,
                        help=f"Batch size por GPU (default: {BATCH_SIZE})")
    args = parser.parse_args()

    # Sobreescribir globals con args
    global DATASET_PATH, EPOCHS, BATCH_SIZE
    DATASET_PATH = args.dataset
    EPOCHS       = args.epochs
    BATCH_SIZE   = args.batch

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] No encontré el dataset: {DATASET_PATH}")
        print(f"  Primero corré: python create_final_dataset.py")
        raise SystemExit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, tokenizer         = load_model()
    train_ds, eval_ds        = load_dataset(DATASET_PATH, tokenizer)
    model, tokenizer         = train(model, tokenizer, train_ds, eval_ds)
    export_gguf(model, tokenizer)


if __name__ == "__main__":
    main()