"""
Benchmarking module for the log parsing model.
"""
from typing import List, Optional
import csv
from transformers import AutoModel, AutoTokenizer
import torch

from .model_class import Model, ModelPrediction, _error_types, _severities


class Benchmark:
    def __init__(self, model: Optional[Model] = None, dataset_path: Optional[str] = None,
                 delimiter: str = ","):
        self.model = model
        self.dataset_path = dataset_path
        self.delimiter = delimiter

        # Get metrics:
        self.metrics = self.model.get_prediction_metrics()

        # Load dataset as a list of ModelPrediction:
        self.dataset = self.load_dataset(dataset_path, delimiter=self.delimiter)

    @staticmethod
    def load_dataset(path: str, delimiter=",") -> List[ModelPrediction]:
        """
        Loads the dataset (from a CSV file) into a list of ModelPrediction.
        """
        print(f"\nLoading dataset from {path} with delimiter '{delimiter}'")
        _dicts = []
        with open(path, 'r', encoding='utf-8') as csvfile:
            rows = csv.DictReader(csvfile, delimiter=delimiter)
            print(f"CSV headers: {rows.fieldnames}")
            
            # Ensure all dictionary keys are strings and store rows
            _dicts = [{str(k).strip(): str(v).strip() if v is not None else None 
                      for k, v in row.items()} 
                     for row in rows]
            
            # Reverse the order of rows
            _dicts.reverse()
            
            print(f"Loaded {len(_dicts)} rows from CSV")
            print("First row:", _dicts[0] if _dicts else "No data")
            
            try:
                _dataset = [ModelPrediction.from_dict(_dict) for _dict in _dicts]
                print(f"Successfully converted {len(_dataset)} rows to ModelPrediction objects")
                return _dataset
            except Exception as e:
                print(f"Error converting dictionary to ModelPrediction: {e}")
                if _dicts:
                    print(f"Sample dict that caused error: {_dicts[0]}")
                raise

    @staticmethod
    def load_dicts(path: str, delimiter=",") -> List[dict]:
        _dicts = []
        with open(path, 'r', encoding='utf-8') as csvfile:
            rows = csv.DictReader(csvfile, delimiter=delimiter)
            _dicts = [{k: v for k, v in row.items()} for row in rows]
        return _dicts

    @staticmethod
    def get_similarity_loss(output: str, label: str) -> float:
        similarity_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

        # Tokenize the input strings
        inputs1 = tokenizer(f"{output}", return_tensors="pt", padding=True, truncation=True)
        inputs2 = tokenizer(f"{label}", return_tensors="pt", padding=True, truncation=True)

        # Generate embeddings for both strings
        with torch.no_grad():
            outputs1 = similarity_model(**inputs1)
            outputs2 = similarity_model(**inputs2)

        # Extract the embeddings (last hidden state of the first token [CLS])
        embedding1 = outputs1.last_hidden_state[:, 0, :]
        embedding2 = outputs2.last_hidden_state[:, 0, :]

        # Compute cosine similarity
        cosine_sim = torch.nn.CosineSimilarity(dim=1)
        similarity_score = cosine_sim(embedding1, embedding2).numpy()[0]
        similarity_loss = (1 - similarity_score) / 2
        return similarity_loss

    @staticmethod
    def _get_loss(pred, label, metric: str) -> float:
        _class_metrics = ["error_type", "severity"]
        if metric in _class_metrics:
            if metric == "error_type":
                if pred not in _error_types:
                    # Unrecognized pred, set to other:
                    pred = "other"
            elif metric == "severity":
                if pred not in _severities:
                    # Unrecognized pred, set to other:
                    pred = "other"
            loss = pred != label
            return loss
        # Otherwise, use semantic similarity function:
        loss = Benchmark.get_similarity_loss(pred, label)
        return loss

    def run_benchmark(self) -> tuple:
        """
        Runs the benchmark on the provided dataset.
        Returns a tuple of the losses (dict) and model predictions (list).
        """
        print("\nStarting benchmark run...")
        _prediction_metrics = self.model.get_prediction_metrics()
        print(f"Model metrics to evaluate: {_prediction_metrics}")
        
        losses = {_metric: 0 for _metric in _prediction_metrics}
        predictions = []

        print(f"\nProcessing {len(self.dataset)} examples...")
        try:
            for i, label in enumerate(self.dataset, 1):
                print(f"\nExample {i}/{len(self.dataset)}:")
                print(f"Input: {label.input[:100]}...")  # Show first 100 chars
                
                # Get prediction:
                label_dict = label.to_dict()
                prediction = self.model.predict(label.input)
                predictions.append(prediction)
                
                # Calculate losses:
                prediction_dict = prediction.to_dict()
                print("Prediction results:")
                for _metric in _prediction_metrics:
                    _pred = prediction_dict[_metric]
                    _label = label_dict[_metric]
                    _loss = Benchmark._get_loss(_pred, _label, _metric)
                    losses[_metric] += _loss
                    print(f"  {_metric}: predicted='{_pred}', actual='{_label}', loss={_loss}")

        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user. Processing partial results...")
        finally:
            # Calculate and display results for processed examples
            data_length = len(predictions)
            print(f"\nBenchmark complete. Processed {data_length}/{len(self.dataset)} examples.")
            
            if data_length > 0:
                print("Final average losses:")
                for metric, loss in losses.items():
                    avg_loss = loss / data_length
                    print(f"  {metric}: {avg_loss:.4f}")
            else:
                print("No examples were processed completely.")

            return losses, predictions, data_length
