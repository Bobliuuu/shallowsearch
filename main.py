from benchmark.benchmark import Benchmark
from benchmark.model_class import Model, ModelPrediction

import argparse
import modal

from groq import Groq

# from ollama import chat
from openai import OpenAI
from typing import Optional, List
from dataclasses import dataclass
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# from azure.ai.inference import ChatCompletionsClient
# from azure.core.credentials import AzureKeyCredential
import os
# from benchmark.apache_model import ApacheBenchmark

load_dotenv()

modal_app = modal.App("rootly-benchmark")


PROMPT_FILEPATH = "static/prompt.txt"
PROMPT = None
with open(PROMPT_FILEPATH, "r") as fp:
    PROMPT = fp.read()
    print(PROMPT)


@dataclass
class GroqModelConfig:
    model: str = "groq"
    model_name: str = "deepseek-r1-distill-llama-70b"
    temperature: float = 0.6
    max_tokens: Optional[int] = 4096
    top_p: float = 0.95


@dataclass
class OllamaModelConfig:
    model: str = "ollama"
    model_name: str = "deepseek-r1:1.5b"
    temperature: float = 0.6
    num_predict: Optional[int] = 4096
    top_p: float = 0.95


@dataclass
class AzureModelConfig:
    model: str = "azure"
    model_name: str = "deepseek-r1"
    temperature: float = 0.6
    max_tokens: Optional[int] = 2048
    endpoint: str = "https://deepseek-r1-rebzw.eastus2.models.ai.azure.com/"


@dataclass
class GPT4oModelConfig:
    model: str = "gpt-4o"
    model_name: str = "gpt-4o"
    temperature: float = 0.6
    num_predict: Optional[int] = 4096
    top_p: float = 0.95


class DeepseekOutput(BaseModel):
    error_type: Optional[str] = Field(
        description="Type of error: see list of categories above."
    )
    severity: Optional[str] = Field(
        description="Severity of the error: one of ['error', 'warn', 'notice']"
    )
    description: Optional[str] = Field(
        description="One-line specific description of the log line."
    )
    solution: Optional[str] = Field(
        description="One-line specific solution to the log line, if error or warning."
    )


class BaseLanguageModel:
    def get_structured_response(self, system_prompt: str) -> DeepseekOutput:
        """Base method to be implemented by subclasses"""
        raise NotImplementedError

    @staticmethod
    def _parse_response(response: str) -> DeepseekOutput:
        """Parse the model's response into structured output"""
        print("--------------------------------")
        print(response)
        print("--------------------------------\n\n")

        output = {"error_type": "", "severity": "", "description": "", "solution": ""}

        try:
            # Split into lines and clean up
            lines = [line.strip() for line in response.split("\n") if line.strip()]

            if not lines:
                print(f"Warning: Empty response from model:\n{response}")
                return DeepseekOutput(**output)

            # Track if we've found any valid content
            found_content = False
            temp_output = dict(output)

            for line in lines:
                if ":" not in line:
                    continue

                # Handle numbered lists with or without periods
                line = (
                    line.replace("1. ", "")
                    .replace("2. ", "")
                    .replace("3. ", "")
                    .replace("4. ", "")
                )

                # Split only on first colon to preserve any colons in the content
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue

                key, value = parts[0].lower().strip(), parts[1].strip()
                value = value.strip("[]")  # Remove brackets if present

                if "severity" in key and value.strip():
                    temp_output["severity"] = value.lower()
                    found_content = True
                elif "error_type" in key and value.strip():
                    temp_output["error_type"] = value.lower()
                    found_content = True
                elif "description" in key and value.strip():
                    temp_output["description"] = value
                    found_content = True
                elif "solution" in key and value.strip():
                    temp_output["solution"] = value
                    found_content = True

            if found_content:
                return DeepseekOutput(**temp_output)

            print(
                f"Warning: Could not parse any valid fields from response:\n{response}"
            )

        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Problematic response:\n{response}")

        return DeepseekOutput(**output)


class GroqLanguageModel(BaseLanguageModel):
    def __init__(self, config: GroqModelConfig):
        self.config = config
        self.client = Groq()
        self.max_retries = 3  # Maximum number of retry attempts

    def get_structured_response(self, system_prompt: str) -> DeepseekOutput:
        """Invokes the Groq model and returns structured output."""
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": system_prompt}],
                    temperature=self.config.temperature
                    + (attempt * 0.1),  # Slightly increase temperature on retries
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    stream=False,
                    stop=None,
                )
                response = completion.choices[0].message.content.strip()
                result = self._parse_response(response)
                # Check if we got valid content
                if (
                    str(result.description)
                    and str(result.solution)
                    and result.error_type in ["fatal", "runtime", "no_error", "warning"]
                    and result.severity in ["notice", "warn", "error"]
                ):
                    return result

                print(f"Attempt {attempt + 1}: Invalid or empty response, retrying...")

            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    print("Max retries reached, returning empty response")
                    return self._parse_response("")
                print("Retrying...")
                continue

        # If we get here, all retries failed
        print("All retry attempts failed")
        return self._parse_response("")


class OllamaLanguageModel(BaseLanguageModel):
    def __init__(self, config: OllamaModelConfig):
        self.config = config
        self.max_retries = 3  # Maximum number of retry attempts

    def get_structured_response(self, system_prompt: str) -> DeepseekOutput:
        """Invokes the Ollama model and returns structured output."""
        for attempt in range(self.max_retries):
            try:
                response = chat(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": system_prompt}],
                    options={
                        "temperature": self.config.temperature + (attempt * 0.1),
                        "num_predict": self.config.num_predict,
                        "top_p": self.config.top_p,
                    },
                )
                result = self._parse_response(response["message"]["content"].strip())

                # Check if we got valid content
                if (
                    str(result.description)
                    and str(result.solution)
                    and result.error_type in ["fatal", "runtime", "no_error", "warning"]
                    and result.severity in ["notice", "warn", "error"]
                ):
                    return result

                print(f"Attempt {attempt + 1}: Invalid or empty response, retrying...")

            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    print("Max retries reached, returning empty response")
                    return self._parse_response("")
                print("Retrying...")
                continue

        # If we get here, all retries failed
        print("All retry attempts failed")
        return self._parse_response("")


class GPT4oLanguageModel(BaseLanguageModel):
    pass


class AzureDeepseekModel(Model):
    """Azure-hosted DeepSeek model implementation"""

    def __init__(self, config: AzureModelConfig = None):
        if config is None:
            config = AzureModelConfig()

        self.config = config

        # Initialize Azure client
        api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL")
        if not api_key:
            raise Exception(
                "AZURE_INFERENCE_CREDENTIAL environment variable must be set"
            )

        self.client = ChatCompletionsClient(
            endpoint=config.endpoint, credential=AzureKeyCredential(api_key)
        )

    def get_prediction_metrics(self) -> List[str]:
        """Return metrics this model predicts"""
        return ["error_type", "severity", "description", "solution"]

    def predict(self, text: str) -> ModelPrediction:
        """Make a prediction using the Azure-hosted model"""
        _prompt = """
        You are an expert at analyzing Apache error logs. Given a log line, classify its error type and severity, and provide a description and solution.

        IMPORTANT - OUTPUT FORMAT:
        You must respond in this exact format with these exact labels:
        1. severity: [one of: notice, warn, error]
        2. error_type: [one of: fatal, runtime, no_error, warning]
        3. description: [your detailed description]
        4. solution: [your detailed solution]

        Special cases to note:
        - Messages containing "SSL support unavailable" should be classified as notice, not error
        - Messages ending with "done..." are notices indicating normal completion
        - Messages starting with "child init" followed by numbers are errors requiring investigation

        For solutions:
        - Start with action verbs (Check, Verify, Ensure)
        - Include specific files and paths
        - For notices, use "No action required"
        - Keep solutions concise and actionable

        Analyze this log line:
        {text}
        """

        payload = {
            "messages": [{"role": "user", "content": _prompt.format(text=text)}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        response = self.client.complete(payload)
        output = response.choices[0].message.content.strip()

        # Parse the response into structured format
        _output = BaseLanguageModel._parse_response(output)

        return ModelPrediction(
            input=text,
            error_type=_output.error_type,
            severity=_output.severity,
            description=_output.description,
            solution=_output.solution,
        )


class DeepseekModel(Model):
    def __init__(self, config: Optional[BaseLanguageModel] = None):
        if config is None:
            config = GroqModelConfig()

        if isinstance(config, GroqModelConfig):
            self.model = GroqLanguageModel(config)
        elif isinstance(config, OllamaModelConfig):
            self.model = OllamaLanguageModel(config)
        else:
            raise ValueError(f"Unsupported model config type: {type(config)}")

    def get_prediction_metrics(self) -> list:
        """Return a list of metrics this model predicts"""
        return ["error_type", "severity", "description", "solution"]

    def predict(self, text: str) -> ModelPrediction:
        """Make a prediction using the selected model"""
        _prompt = """
        You are an expert at analyzing Apache error logs. Given a log line, classify its error type and severity, and provide a description and solution.

        IMPORTANT - OUTPUT FORMAT:
        You must respond in this exact format with these exact labels:
        1. severity: [one of: notice, warn, error]
        2. error_type: [one of: fatal, runtime, no_error, warning]
        3. description: [your detailed description]
        4. solution: [your detailed solution]

        DO NOT use JSON format. DO NOT add any other text or explanations. DO NOT add markdown formatting in the labels.
        Any deviation from this format will result in parsing errors.

        For solutions, follow these guidelines:
        - Start with action verbs like "Check", "Verify", "Ensure", or "Review"
        - For configuration issues, mention specific files (e.g., .htaccess, apache2.conf)
        - For permission issues, include specific commands (e.g., chmod, chown)
        - For no_error types, use "No action is required" or "No specific action is required"
        - Keep solutions concise but specific
        - Include file paths when relevant
        - For PHP/code issues, mention functions or methods to check

        Special cases to note:
        - Messages containing "SSL support unavailable" should be classified as notice, not error
        - Messages like "done..." are notices indicating normal completion
        - Messages starting with "child init" followed by numbers are errors requiring investigation

        Analysis steps:
        1. First, identify the log format components:
           - Timestamp if present (e.g., [Thu Sep 21 11:10:55.123456 2023])
           - Module/component (e.g., [rewrite:warn], [client], [php])
           - Client IP if present (e.g., [client 172.16.0.1])
           - Error code if present (e.g., AH00671, AH01630)

        2. Look for key indicators:
           - Error codes starting with "AH" are Apache-specific
           - Module names indicate the source (php, rewrite, core, etc.)
           - Words like "warning", "error", "notice" suggest severity
           - File paths often indicate the affected component

        Common patterns in Apache logs:
        1. PHP warnings:
           - "undefined array key" issues when code attempts to access non-existent array keys
           - "php warning where the code is attempting to access an undefined array"
           - Common with HTTP_USER_AGENT and 'host' key access attempts
           - Often seen in WordPress themes and plugins
           - Usually indicates missing input validation

        2. Access denials:
           - "client was denied access to the serverstatus resource due to server configuration"
           - "was denied access to the serverstatus" often indicates configuration restrictions
           - Access attempts to server-status pages are commonly blocked
           - Often related to .htaccess rules or Directory directives
           - May indicate security measures working as intended

        3. Normal operations:
           - "apache server has successfully completed its configuration and is resuming normal"
           - Command line startup messages (/usr/sbin/apache2)
           - Configuration loading messages (workers2.properties)
           - Server state changes (graceful restart, shutdown)
           - Process management messages (child processes, scoreboard)

        4. Runtime errors:
           - Invalid URI paths (often containing encoded characters)
           - Directory traversal attempts (../../)
           - Configuration mismatches between SSL and hostnames
           - Client access restrictions based on IP or configuration
           - File permission issues affecting rewrites or access

        Severity classification guidelines:
        1. severity: Must be one of ["notice", "warn", "error"]
        - "notice":
          * Normal operational messages (configuration loaded, server startup)
          * Informational state changes
          * Successful configuration readings
          * Process management information
          * Normal HTTP operations

        - "warn":
          * PHP warnings about undefined variables or array keys
          * Missing but non-critical configuration elements
          * Deprecated feature usage
          * Non-critical file access issues
          * Potential security concerns that were properly handled

        - "error":
          * Access denied messages
          * Invalid URI or file paths
          * SSL/TLS configuration issues
          * Critical file permission problems
          * Security violation attempts
          * Process creation failures

        Error type classification guidelines:
        2. error_type: Must be one of ["fatal", "runtime", "no_error", "warning"]
        - "no_error":
          * Server startup and shutdown messages
          * Normal configuration loading
          * Process management information
          * Successful operations logs

        - "warning":
          * PHP variable/array access issues
          * Deprecated feature notices
          * Non-critical configuration warnings
          * Handled security warnings

        - "runtime":
          * Access denied errors
          * File permission issues
          * URI parsing problems
          * Configuration mismatches
          * Resource access failures

        - "fatal":
          * Server unable to start
          * Critical configuration errors
          * SSL certificate failures
          * Process creation failures
          * Resource exhaustion errors

        Description and Solution Guidelines:
        3. description: 
           - Be specific about what the log indicates
           - Include relevant error codes
           - Mention affected components
           - Note any security implications
           - Indicate if it's normal behavior or an issue

        4. solution:
           - For errors/warnings, provide actionable steps
           - Reference specific configuration files if relevant
           - Mention security implications if any
           - Suggest validation or error handling if needed
           - If no_error, state that no action is needed

        FINAL REMINDER - YOUR RESPONSE MUST:
        I. Start immediately with "severity:" (no preamble)
        II. Use exactly these labels:
            1. severity: [one of: notice, warn, error]
            2. error_type: [one of: fatal, runtime, no_error, warning]
            3. description: [your detailed description]
            4. solution: [your detailed solution]
        III. Include all four fields in this exact order
        IV. Not include any other text or formatting

        Analyze this log line:
        {text}
        """
        _output = self.model.get_structured_response(f"{_prompt}\n\n{text}")
        _pred = ModelPrediction(
            input=text,
            error_type=_output.error_type,
            severity=_output.severity,
            description=_output.description,
            solution=_output.solution,
        )
        return _pred


class GPT4oModel(Model):
    def __init__(
        self, config: Optional[BaseLanguageModel] = None, eval_prompt: str = None
    ):
        # temperature: float = 0.6
        # num_predict: Optional[int] = 4096
        # top_p: float = 0.95

        self.model = OpenAI(api_key=os.getenv("X_OPENAI_API_KEY"))
        self.eval_prompt = eval_prompt

    def get_prediction_metrics(self) -> list:
        """Return a list of metrics this model predicts"""
        return ["error_type", "severity", "description", "solution"]

    def predict(self, text: str) -> ModelPrediction:
        """Make a prediction using the selected model"""

        _prompt = self.eval_prompt + f"\n{text}"
        print(_prompt)

        completion = self.model.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": _prompt}],
            top_p=1.0,
            temperature=1.0,
        )
        completion_text = completion.choices[0].message.content

        print("completion_text:", completion_text)
        # print(completion_text)

        formatted_output = BaseLanguageModel._parse_response(completion_text)
        print("formatted_output", formatted_output)

        print(formatted_output.error_type)

        # raise KeyError

        # _output = self.model.get_structured_response(f"{_prompt}\n\n{text}")
        _pred = ModelPrediction(
            input=text,
            error_type=formatted_output.error_type,
            severity=formatted_output.severity,
            description=formatted_output.description,
            solution=formatted_output.solution,
        )
        return _pred


class GroqV2(Model):
    def __init__(
        self, config: Optional[BaseLanguageModel] = None, eval_prompt: str = None
    ):
        # temperature: float = 0.6
        # num_predict: Optional[int] = 4096
        # top_p: float = 0.95

        self.model = Groq(
            api_key=os.getenv("X_GROQ_API_KEY"),
        )
        self.eval_prompt = eval_prompt

    def get_prediction_metrics(self) -> list:
        """Return a list of metrics this model predicts"""
        return ["error_type", "severity", "description", "solution"]

    def predict(self, text: str) -> ModelPrediction:
        """Make a prediction using the selected model"""

        _prompt = self.eval_prompt + f"\n{text}"
        print(_prompt)

        completion = self.model.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": _prompt}],
            top_p=1.0,
            temperature=1.0,
        )
        completion_text = completion.choices[0].message.content

        print("completion_text:", completion_text)
        # print(completion_text)

        formatted_output = BaseLanguageModel._parse_response(completion_text)
        print("formatted_output", formatted_output)

        print(formatted_output.error_type)

        # raise KeyError

        # _output = self.model.get_structured_response(f"{_prompt}\n\n{text}")
        _pred = ModelPrediction(
            input=text,
            error_type=formatted_output.error_type,
            severity=formatted_output.severity,
            description=formatted_output.description,
            solution=formatted_output.solution,
        )
        return _pred


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
        model = GPT4oModel(eval_propt=PROMPT)
        print("GPT-4o benchmark")
    elif args.model == "deepseek-70b-llama-groq":
        model = GroqV2(eval_prompt=PROMPT)
        print("GroqV2 benchmark")
    elif args.model == "modal":

        @modal_app.local_entrypoint()
        def main():
            print(square.remote(10))
            print("---")
    else:
        print(f"Model argument {args.model} not recognized.")

        # resp = square(10)
        # print(resp)

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
