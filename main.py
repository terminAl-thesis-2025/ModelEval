import requests
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import time
import json
import logging
from typing import Dict, List
from contextlib import contextmanager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('settings/ollama_evaluator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OllamaAPIError(Exception):
    """Custom exception for Ollama API errors"""
    pass


class OllamaEvaluator:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.base_url = self.config['base_url'].rstrip('/')
        self.models = self.config['models']
        self.system_prompt = self.config.get('system_prompt', '')
        self.output_dir = Path("model_outputs")
        self.output_dir.mkdir(exist_ok=True)

        # Simple session without retries
        self.session = requests.Session()
        self._verify_api_connection()

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            required_fields = ['base_url', 'models']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field '{field}' in config")

            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise Exception(f"Failed to load config: {str(e)}")

    def _verify_api_connection(self) -> None:
        try:
            version_response = self.session.get(f"{self.base_url}/api/version")
            version_response.raise_for_status()
            logger.info(f"Connected to Ollama API version: {version_response.json().get('version')}")

            # Check available models
            tags_response = self.session.get(f"{self.base_url}/api/tags")
            tags_response.raise_for_status()
            available_models = [model['name'] for model in tags_response.json().get('models', [])]

            # Verify our models exist
            for model_key, model_info in self.models.items():
                model_name = model_info['name']
                if model_name not in available_models:
                    logger.warning(f"Model '{model_name}' is not currently available in Ollama")

        except requests.exceptions.RequestException as e:
            raise OllamaAPIError(f"Failed to connect to Ollama API: {str(e)}")

    def generate_response(self, model: str, prompt: str) -> Dict:
        url = f"{self.base_url}/api/generate"

        data = {
            "model": self.models[model]['name'],
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.get('temperature', 0.7),
                "num_predict": self.config.get('num_predict', 100),
                "stop": self.config.get('stop_tokens', ["\n\n"]),
                "num_ctx": self.config.get('context_length', 2048),
            }
        }

        response = self.session.post(
            url,
            json=data,
            timeout=self.config.get('timeout', 30)
        )
        response.raise_for_status()
        response_data = response.json()

        if 'response' not in response_data:
            raise OllamaAPIError("No response field in API response")

        # Try to parse the response as JSON if it's a string
        raw_response = response_data['response']
        try:
            if isinstance(raw_response, str):
                parsed_response = json.loads(raw_response)
            else:
                parsed_response = raw_response
        except json.JSONDecodeError:
            # If it's not valid JSON, keep it as is
            parsed_response = raw_response

        return {
            'model': model,
            'prompt': prompt,
            'response': parsed_response,  # Use the parsed response
            'eval_duration': response_data.get('eval_duration', 0),
            'eval_count': response_data.get('eval_count', 0),
            'total_duration': response_data.get('total_duration', 0)
        }

    def write_output(self, model: str, prompt: str, response: Dict) -> None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{model}_results.jsonl"

            output_data = {
                "timestamp": timestamp,
                "model": model,
                "model_description": self.models.get(model, {}).get('description', 'Unknown model'),
                "prompt": prompt,
                "response": response
            }

            # Write with proper formatting
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False)
                f.write("\n")

        except IOError as e:
            logger.error(f"Failed to write output: {str(e)}")
            raise

    # You might also want to add a method to clear previous results:
    def clear_previous_results(self, model: str) -> None:
        """Clear previous results for a given model before starting new evaluation."""
        output_file = self.output_dir / f"{model}_results.jsonl"
        if output_file.exists():
            output_file.unlink()

    def debug_output(self, model: str) -> None:
        """Debug method to verify JSON output format."""
        output_file = self.output_dir / f"{model}_results.jsonl"
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Parse and pretty print the line
                        data = json.loads(line)
                        print(f"\nLine {line_num}:")
                        print(json.dumps(data, indent=2, ensure_ascii=False))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")

    # Usage in evaluate_models:
    def evaluate_models(self, prompts: List[str]) -> None:
        total_iterations = len(self.models) * len(prompts)
        failed_evaluations = []

        with tqdm(total=total_iterations, desc="Evaluating models") as pbar:
            for model in self.models:
                self.clear_previous_results(model)

                for prompt in prompts:
                    try:
                        response = self.generate_response(model, prompt)
                        self.write_output(model, prompt, response)
                    except Exception as e:
                        error_msg = f"Error evaluating {model} with prompt '{prompt[:50]}...': {str(e)}"
                        logger.error(error_msg)
                        failed_evaluations.append({
                            "model": model,
                            "prompt": prompt,
                            "error": str(e)
                        })
                    finally:
                        pbar.update(1)

                # Debug output after each model is complete
                logger.info(f"\nVerifying output format for {model}:")
                self.debug_output(model)

        if failed_evaluations:
            self._report_failures(failed_evaluations)

    def _report_failures(self, failures: List[Dict]) -> None:
        report_path = self.output_dir / "failed_evaluations.json"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(failures, f, indent=2, ensure_ascii=False)
            logger.warning(f"Some evaluations failed. See {report_path} for details.")
        except IOError as e:
            logger.error(f"Failed to write failure report: {str(e)}")

def main():
    try:
        with open('settings/prompts.json', 'r') as f:
            prompt_data = json.load(f)

        prompts = [q for category in prompt_data['prompts']
                   for q in category['questions']]

        evaluator = OllamaEvaluator()
        evaluator.evaluate_models(prompts)

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()