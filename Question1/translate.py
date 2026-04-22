import torch
from transformers import MarianMTModel, MarianTokenizer
import sacrebleu
from striprtf.striprtf import rtf_to_text

# Model name
model_name = "Helsinki-NLP/opus-mt-bn-en"

# Load tokenizer and model
print("Loading model and tokenizer...")
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_clean_lines(rtf_file):
    with open(rtf_file, 'r', encoding='utf-8') as f:
        text = rtf_to_text(f.read())
    lines = text.split('\n')
    # Remove header line (starting with #) and empty lines
    clean_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return clean_lines

print("Reading input and reference files...")
input_lines = get_clean_lines('input.rtf')
reference_lines = get_clean_lines('output.rtf')

print(f"Number of input statements: {len(input_lines)}")

translated_lines = []
print("Translating...")
for i, text in enumerate(input_lines):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translated_lines.append(decoded)
    if i == 0:
        first_statement_output = decoded

print("\n--- FIRST STATEMENT OUTPUT ---")
print(first_statement_output)
print("------------------------------\n")

# Save output to output.txt
with open('output.txt', 'w', encoding='utf-8') as f:
    for line in translated_lines:
        f.write(line + '\n')

print("Output saved to output.txt.")

# Compute BLEU score
print("Computing BLEU score...")
bleu = sacrebleu.corpus_bleu(translated_lines, [reference_lines])
print(f"BLEU score: {bleu.score}")
