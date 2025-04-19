import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from llm2vec import LLM2Vec
from transformers import AutoModel
from peft import PeftModel

import LLM2VecModel


def main():
    print("Cuda is ", torch.cuda.is_available())
    MODEL_PATH_OR_NAME_A = r"meta-llama/Llama-3.2-1B"
    MODEL_PATH_OR_NAME_B = r"./saved_models"
    BEST_MODEL_BASELINE = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"

    model_a = LLM2Vec.from_pretrained(MODEL_PATH_OR_NAME_A)
    model_b = LLM2Vec.from_pretrained(MODEL_PATH_OR_NAME_B)

    model_best = LLM2VecModel.LLM2VecModel(BEST_MODEL_BASELINE)

    result_best = titans_test(model_best)
    result_a = titans_test(model_a)
    result_b = titans_test(model_b)
    compare_similarity(result_a, result_b)
    vis_similiarty(abs(result_best - result_b) - abs(result_best - result_a))
    plt.show()


def calculate_cos_similarity(embeder, docs, queries):
    q_reps = embeder.encode(queries)
    d_reps = embeder.encode(docs)

    q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
    d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
    return torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))


def titans_test(a):
    prompts = [
        "How do creatures on Titan communicate?",
        "What is the lost city of Auravelle?",
        "Describe the properties of Nocturnium.",
        "How do deep-sea creatures communicate?",
        "Are there ancient cities under ice?",
        "Do any minerals glow naturally?",
        "What is Titan known for?",
        "What are the legends of Atlantis?",
        "How do solar panels absorb energy?",
        "What is the capital of France?",
        "How do birds migrate?",
        "What is quantum entanglement?"
    ]

    documents = [
        "Scientists believe that bioluminescent organisms in Titan’s methane seas use rhythmic flashes of light as a "
        "form of communication.",
        "The lost city of Auravelle is a rumored civilization buried beneath the Arctic ice, said to contain ancient "
        "structures resistant to extreme cold.",
        "Nocturnium is a rare mineral that absorbs moonlight and emits a faint glow only in complete darkness.",
        "Many deep-sea creatures use bioluminescence to attract prey or signal to others in their environment.",
        "Some theories suggest that lost civilizations could be buried beneath glaciers, but no definitive proof has "
        "been found.",
        "Some minerals, like fluorite and phosphorescent rocks, can emit light after being exposed to energy sources.",
        "Titan is Saturn’s largest moon, known for its thick atmosphere and methane lakes.",
        "Atlantis is a mythical island city described by Plato, said to have sunk into the ocean due to divine "
        "punishment.",
        "Solar panels use photovoltaic cells to convert sunlight into electricity, storing it for later use.",
        "Paris is the capital of France, known for its rich history and cultural landmarks.",
        "Many bird species migrate seasonally, using the Earth’s magnetic field and landmarks for navigation.",
        "Quantum entanglement is a phenomenon where two particles remain interconnected, instantly affecting each "
        "other regardless of distance."
    ]

    return calculate_cos_similarity(embeder=a, docs=documents, queries=prompts)


def compare_similarity(similarity_a, similarity_b):
    print(similarity_a)
    print(similarity_b)
    print(torch.argmax(similarity_a, dim=1))
    print(torch.argmax(similarity_b, dim=1))
    plot_matrix_heatmap(similarity_a)
    plot_matrix_heatmap(similarity_b)

    print(similarity_b - similarity_a)
    plot_matrix_heatmap(similarity_b - similarity_a)


def vis_similiarty(similarity):
    print(similarity)
    print(torch.argmax(similarity, dim=1))
    plot_matrix_heatmap(similarity)


def cloud_test(model):
    documents = [
        "Clouds are formed when water vapor in the atmosphere cools and condenses into visible water droplets or ice "
        "crystals.",
        "There are different types of clouds, such as cumulus, stratus, and cirrus, which vary in appearance and "
        "altitude.",
        "Cloud computing refers to the delivery of computing services like storage, databases, and software over the "
        "internet.",
        "Clouds can have a significant impact on weather patterns and climate conditions."
    ]

    prompts = [
        "What causes clouds to form?",
        "Explain the different types of clouds.",
        "What is cloud computing?",
        "How do clouds affect the weather?"
    ]

    return calculate_cos_similarity(embeder=model, docs=documents, queries=prompts)


def plot_matrix_heatmap(matrix):
    """
    Plots a heatmap of a given matrix where negative values are red
    and positive values are blue, with intensity based on magnitude.

    Parameters:
    - matrix: A 2D NumPy array or list of lists

    """

    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().detach().numpy()  # Convert to NumPy safely
    # Convert to numpy array in case it's not
    matrix = np.array(matrix)

    # Define a diverging colormap (blue for positive, red for negative)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Create the heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, center=0, linewidths=0.5, cbar=True)

    # Set title and show the plot
    plt.title("Matrix Heatmap")


if __name__ == '__main__':
    main()
