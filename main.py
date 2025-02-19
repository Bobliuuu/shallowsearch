import argparse
import os

import modal
from dotenv import load_dotenv

from benchmark.benchmark import Benchmark
from src.models import *

load_dotenv()

modal_app = modal.App("rootly-benchmark")


VERBOSE = False
PROMPT_FILEPATH = "static/prompt.txt"
PROMPT = None
with open(PROMPT_FILEPATH, "r") as fp:
    PROMPT = fp.read()
    if VERBOSE:
        print(PROMPT)


@modal_app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    # # Choose model to use
    # use_ollama = True
    # use_azure = False
    # # Otherwise use Groq + Deepseek-R1-Distill-LLama-70B

    # if use_ollama:
    #     model = DeepseekModel(config=OllamaModelConfig())
    # elif use_azure:
    #     model = AzureDeepseekModel(config=AzureModelConfig())
    # else:
    #     model = DeepseekModel(config=GroqModelConfig())

    if args.model == "gpt4o":
        model = GPT4oModel(eval_prompt=PROMPT)
        print("GPT-4o benchmark")
    elif args.model == "deepseek-70b-llama-groq":
        model = GroqDeepSeekLlama70BV2(eval_prompt=PROMPT)
        print("Groq DeepSeek 70B benchmark")
    elif args.model == "llama-3-3-70b-groq":
        model = GroqLlama3_3_70B(eval_prompt=PROMPT)
        print("Groq Llama 3.3 70B benchmark")

    else:
        raise NameError(f"Model argument {args.model} not recognized.")

    # benchmark = Benchmark(model=model, dataset_path="data/actual/dataset.csv", delimiter="|")
    benchmark = Benchmark(
        model=model, dataset_path="data/validation/actual_validation.csv", delimiter="|"
    )
    benchmark.run_benchmark()

    # benchmark = ApacheBenchmark(
    #     model=model,
    #     dataset_path="data/actual/rootly.csv",
    #     delimiter=","
    # )
    # results = benchmark.run_benchmark(max_samples=100)
