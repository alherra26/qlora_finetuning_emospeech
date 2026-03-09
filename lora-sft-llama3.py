# ==========================================
# 0. DEPENDENCY INSTALLATION (KAGGLE)
# ==========================================
import os
print("Installing required libraries...")
os.system("pip install -q -U bitsandbytes transformers accelerate datasets trl peft")

# ==========================================
# 1. SETUP AND IMPORTS
# ==========================================
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                          BitsAndBytesConfig, TrainingArguments, pipeline)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Optimizations and environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

print(f"Working on {device} with PyTorch version {torch.__version__}")

# Paths and model configuration
TRAIN_FILE = "/kaggle/input/datasets/alherra26/emospeech-train/EmoSPeech_phase_2_train_public.csv"
TEST_FILE = "/kaggle/input/datasets/alherra26/emospeech-v2/EmoSPeech_phase_2_test_public.csv"
MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "trained_weigths_4_EPOCH"

# ==========================================
# 2. DATA PREPARATION AND FILE RADAR
# ==========================================
print("--- KAGGLE FILE RADAR ---")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("-------------------------")

# Load training data
df = pd.read_csv(TRAIN_FILE, names=["sentiment", "text"], encoding="utf-8", encoding_errors="replace", sep=",")

X_train, X_eval = [], []

# Stratified split based on sentiment
for sentiment in ["anger", "joy", "neutral", "disgust", "fear", "sadness"]:
    train, test = train_test_split(df[df.sentiment==sentiment], train_size=0.7, test_size=0.3, random_state=42)
    X_train.append(train)
    X_eval.append(test)

X_train = pd.concat(X_train).sample(frac=1, random_state=10).reset_index(drop=True)
X_eval = pd.concat(X_eval).reset_index(drop=True)

# Prompt engineering function
def generate_prompt(data_point, is_train=True):
    base_prompt = f"""Tu objetivo es identificar la emoción entre las siguientes: Joy, Sadness, Fear, Anger, Disgust y Neutral emotion. Aquí tienes las definiciones de cada una:
        \n- Joy: Esta emoción se caracteriza por sentimientos de alegría, contentamiento y bienestar. Se manifiesta a través de sonrisas, risas y expresiones faciales que muestran placer.
        \n- Sadness: La tristeza se experimenta como un sentimiento de pena, pérdida o decepción. Se manifiesta con expresiones faciales como cejas fruncidas, labios caídos y ojos llorosos.
        \n- Fear: Esta emoción surge en situaciones de peligro o amenaza y se caracteriza por sentimientos de ansiedad y preocupación. Las expresiones faciales típicas del miedo incluyen ojos abiertos, cejas levantadas y boca abierta.
        \n- Anger: El enfado se experimenta como respuesta a la frustración, injusticia o provocación. Se manifiesta con expresiones faciales como cejas fruncidas, labios apretados y dientes apretados.
        \n- Disgust: Esta emoción surge en respuesta a estímulos desagradables o repulsivos y se caracteriza por sentimientos de aversión y repulsión. Las expresiones faciales típicas del disgusto incluyen narices arrugadas, labios superiores levantados y una expresión de desagrado.
        \n- Neutral: Este estado significa una falta de expresión emocional fuerte o respuesta, ni positiva ni negativa. Se caracteriza por una expresión facial relajada o impasible, mostrando ni alegría ni angustia.
        \n\nLa frase a clasificar es : '{data_point["text"]}'
        \n\nSolución: """
    
    if is_train:
        return base_prompt + data_point["sentiment"]
    return base_prompt

# Apply prompt generation
X_train = pd.DataFrame(X_train.apply(lambda x: generate_prompt(x, True), axis=1), columns=["text"])
X_eval = pd.DataFrame(X_eval.apply(lambda x: generate_prompt(x, False), axis=1), columns=["text"])

train_data = Dataset.from_pandas(X_train)
eval_data = Dataset.from_pandas(X_eval)

# ==========================================
# 3. MODEL (LLaMA 3) AND TOKENIZER LOADING
# ==========================================
from peft import prepare_model_for_kbit_training 

compute_dtype = getattr(torch, "float16")

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device,
    torch_dtype=compute_dtype,
    quantization_config=bnb_config, 
)

model.config.use_cache = False
model.config.pretraining_tp = 1
model.config.torch_dtype = torch.float16

for param in model.parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)

model = prepare_model_for_kbit_training(model)

# Load tokenizer
max_seq_length = 2048
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, max_seq_length=max_seq_length)
tokenizer.pad_token_id = tokenizer.eos_token_id

# ==========================================
# 4. PEFT / LoRA CONFIGURATION AND TRAINING
# ==========================================
from trl import SFTConfig, SFTTrainer

# LoRA config
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# SFTConfig
training_arguments = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    packing=False,
    max_length=max_seq_length,    
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    eval_steps=25,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    processing_class=tokenizer,
)

print("Applying memory corrections for GPU P100...")

for param in trainer.model.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)
    elif param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)

for buffer in trainer.model.buffers():
    if buffer.dtype == torch.bfloat16:
        buffer.data = buffer.data.to(torch.float16)

print("Starting training...")
trainer.train()

# Save the trained model and tokenizer
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training finished and model saved.")

# ==========================================
# 5. INFERENCE AND TESTING
# ==========================================
def predict(dataset, model_to_use, tokenizer_to_use):
    y_pred = []
    # Note: LLaMA pipeline sometimes does not use cache properly after training without reloading, 
    # but keeping the original logic
    pipe = pipeline(task="text-generation", model=model_to_use, tokenizer=tokenizer_to_use, 
                    max_new_tokens=5, temperature=0.0)
    
    for i in tqdm(range(len(dataset))):
        prompt = dataset.iloc[i]["text"]
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("Solución:")[-1].strip().lower()

        # Parse the predicted emotion
        if "anger" in answer: y_pred.append("anger")
        elif "disgust" in answer: y_pred.append("disgust")
        elif "fear" in answer: y_pred.append("fear")
        elif "joy" in answer: y_pred.append("joy")
        elif "sadness" in answer: y_pred.append("sadness")
        else: y_pred.append("neutral")    
    return y_pred

# Read test data and predict
df_test_data = pd.read_csv(TEST_FILE, encoding="utf-8", encoding_errors="replace")
df_test_data['text'] = df_test_data['text'].apply(lambda x: generate_prompt({"text": x}, False))

print("Starting inference on test data...")
y_EmoSpeech = predict(df_test_data[['text']], model, tokenizer)

# Save results
df_results = pd.DataFrame(y_EmoSpeech, columns=['label'])
df_results.to_csv('output_test.csv', index=False, sep=';')
print("Predictions file output_test.csv generated successfully!")