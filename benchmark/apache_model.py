"""
Apache log parsing model class focused on severity classification and solutions.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass


@dataclass
class ApacheModelConfig:
    """Configuration for Apache log parsing models"""
    model: str = "groq"
    model_name: str = "mixtral-8x7b-32768"  # Default to Mixtral
    temperature: float = 0.2  # Lower temperature for more consistent outputs
    max_tokens: Optional[int] = 1024
    top_p: float = 0.9


class ApacheLogPrediction(BaseModel):
    """Prediction output for Apache log entries"""
    input: Optional[str] = Field(description="Original log entry")
    severity: str = Field(description="Severity level: notice, warn, or error")
    description: str = Field(description="Brief description of the log entry")
    solution: str = Field(description="Recommended solution or action")


class ApacheLogModel(ABC):
    """Base class for Apache log parsing models"""
    
    @abstractmethod
    def get_prediction_metrics(self) -> List[str]:
        """Return list of metrics this model predicts"""
        return ["severity", "description", "solution"]

    @abstractmethod
    def predict(self, log_entry: str) -> ApacheLogPrediction:
        """
        Analyze an Apache log entry and return structured prediction
        
        Args:
            log_entry: Raw log entry string
            
        Returns:
            ApacheLogPrediction containing severity, description and solution
        """
        pass


class ApacheBenchmark:
    """Benchmark class for Apache log parsing models"""
    
    def __init__(self, model: ApacheLogModel, dataset_path: str, delimiter: str = ","):
        self.model = model
        self.dataset_path = dataset_path
        self.delimiter = delimiter
        self.metrics = self.model.get_prediction_metrics()

    def _calculate_severity_accuracy(self, predicted: str, actual: str) -> float:
        """Calculate accuracy score for severity prediction"""
        return 1.0 if predicted.lower() == actual.lower() else 0.0

    def run_benchmark(self, max_samples: Optional[int] = None) -> dict:
        """
        Run benchmark on the dataset
        
        Args:
            max_samples: Optional limit on number of samples to process
            
        Returns:
            Dictionary containing benchmark metrics
        """
        import csv
        import time
        
        results = {
            "severity_accuracy": 0.0,
            "total_samples": 0,
            "processing_time": 0.0,
            "predictions": []
        }
        
        start_time = time.time()
        
        try:
            with open(self.dataset_path, 'r') as f:
                reader = csv.DictReader(f, delimiter=self.delimiter)
                for i, row in enumerate(reader):
                    if max_samples and i >= max_samples:
                        break
                        
                    # Make prediction
                    prediction = self.model.predict(row['input'])
                    
                    # Calculate metrics
                    severity_score = self._calculate_severity_accuracy(
                        prediction.severity, 
                        row['severity']
                    )
                    
                    results["severity_accuracy"] += severity_score
                    results["total_samples"] += 1
                    
                    # Store prediction
                    results["predictions"].append({
                        "input": row['input'],
                        "actual_severity": row['severity'],
                        "predicted_severity": prediction.severity,
                        "description": prediction.description,
                        "solution": prediction.solution
                    })
                    
                    # Print progress
                    if i % 10 == 0:
                        print(f"Processed {i} samples...")
                        
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
        finally:
            # Calculate final metrics
            if results["total_samples"] > 0:
                results["severity_accuracy"] /= results["total_samples"]
            results["processing_time"] = time.time() - start_time
            
            # Print summary
            print(f"\nBenchmark complete:")
            print(f"Samples processed: {results['total_samples']}")
            print(f"Severity accuracy: {results['severity_accuracy']:.2%}")
            print(f"Processing time: {results['processing_time']:.2f}s")
            
        return results 