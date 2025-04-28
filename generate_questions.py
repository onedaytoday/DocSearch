from openai import OpenAI
import os
import csv
from pathlib import Path

# OpenAI key setup
key = "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"
client = OpenAI(api_key=key)

# Directories
input_dir = Path("Emanuals_Corpus")
output_csv = Path("generated_questions.csv")

# Parameters
max_chars_per_request = 8000  # character limit
ai_model = "gpt-4o-mini"

# Helper to generate Q&A
def generate_question_answer(text):
    prompt = f"""Based on the following text, generate ONE question and a detailed answer.
Please format the response EXACTLY like this:

Question: <your question here>
Answer: <your detailed answer here>

Here is the text:
{text}
"""
    response = client.chat.completions.create(
        model=ai_model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

# Create CSV if it doesn't exist, and track already processed files
processed_files = set()
if output_csv.exists():
    with open(output_csv, "r", encoding="utf-8") as csvfile_read:
        reader = csv.DictReader(csvfile_read)
        for row in reader:
            processed_files.add(row["source file"])
else:
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile_write:
        writer = csv.writer(csvfile_write)
        writer.writerow(["source file", "question", "answer"])

# Open CSV for appending
with open(output_csv, "a", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    # Process files
    for filepath in input_dir.rglob("*.txt"):
        if filepath.name in processed_files:
            print(f"Skipping already processed file: {filepath}")
            continue

        print(f"Processing: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            original_text = f.read().strip()

        if not original_text:
            print(f"Skipping empty file: {filepath}")
            continue

        text_to_use = original_text[:max_chars_per_request]

        try:
            qa_text = generate_question_answer(text_to_use)
            # Expecting format:
            # Question: ...
            # Answer: ...

            # Simple parsing
            lines = qa_text.splitlines()
            question = ""
            answer = ""
            for line in lines:
                if line.strip().lower().startswith("question:"):
                    question = line.split(":", 1)[1].strip()
                elif line.strip().lower().startswith("answer:"):
                    answer = line.split(":", 1)[1].strip()

            if question and answer:
                writer.writerow([str(filepath.name), question, answer])
            else:
                print(f"Failed to parse Q&A for file: {filepath}")

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

print("All done!")
