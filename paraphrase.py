from openai import OpenAI
import os
import openai
from pathlib import Path

key =  ""
# Initialize OpenAI API
openai.api_key = key  # Replace this
client = OpenAI(api_key=key)

# Directories
input_dir = Path("Emanuals_Corpus")
output_dir = Path("Emanuals_Corpus_Paraphrased")
output_dir.mkdir(exist_ok=True)

# Parameters
max_chars_per_request = 8000  # Approximate safe limit
n_paraphrases = 3
ai_model = "gpt-4o-mini"

# Helper function to chunk text
def chunk_text(text, max_length):
    chunks = []
    while len(text) > max_length:
        split_point = text.rfind("\n", 0, max_length)
        if split_point == -1:
            split_point = max_length
        chunks.append(text[:split_point])
        text = text[split_point:]
    if text:
        chunks.append(text)
    return chunks

# Helper function to paraphrase text
def paraphrase_text(chunk):
    response = client.responses.create(
        model=ai_model,
        input=f"Paraphrase the following text in a natural, technical way. Keep the meaning the same:\n\n{chunk}"
    )
    return response.output[0].content[0].text

# Process files
for filepath in input_dir.rglob("*.txt"):
    relative_path = filepath.relative_to(input_dir)
    basename = relative_path.stem

    # Check if already paraphrased
    already_done = all(
        (output_dir / relative_path.parent / f"{basename}_paraphrase_{i+1}.txt").exists()
        for i in range(n_paraphrases)
    )
    if already_done:
        print(f"Skipping already paraphrased file: {filepath}")
        continue

    print(f"Processing: {filepath}")

    # Read text
    with open(filepath, "r", encoding="utf-8") as f:
        original_text = f.read()

    chunks = chunk_text(original_text, max_chars_per_request)

    # For each paraphrase
    for i in range(n_paraphrases):
        output_subdir = output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_file = output_subdir / f"{basename}_paraphrase_{i+1}.txt"

        with open(output_file, "a", encoding="utf-8") as out_f:
            for chunk in chunks:
                try:
                    paraphrased = paraphrase_text(chunk)
                    out_f.write(f"{paraphrased}\n\n")
                except Exception as e:
                    print(f"Error paraphrasing chunk: {e}")

print("All done!")
