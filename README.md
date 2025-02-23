# ModelEval

A Python-based evaluation tool for comparing different Ollama models using predefined prompts. The tool allows systematic testing of multiple language models and captures their responses in a structured format.

## Project Structure

```
ModelEval/
├── LICENSE
├── main.py
├── model_outputs/
├── README.md
├── requirements.txt
└── settings/
    ├── config.json
    ├── config_template.json
    ├── ollama_evaluator.log
    ├── prompts.json
    └── prompts_template.json
```

## Prerequisites

- Python 3.8+
- Ollama running locally or accessible via network

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. Configure your settings:
   - Copy `settings/config_template.json` to `settings/config.json`
   - Copy `settings/prompts_template.json` to `settings/prompts.json`
   - Modify the configuration files according to your needs

### Configuration Files

#### config.json
Contains settings for:
- Ollama API endpoint
- Model configurations
- System prompt
- Generation parameters

## Usage

Run the evaluation:

```bash
python main.py
```

The tool will:
1. Load the configured models and prompts
2. Execute evaluations for each model-prompt combination
3. Save results in the `model_outputs` directory as JSONL files
4. Generate error reports if any evaluations fail

## Output Format

Results are stored in JSONL format with the following structure:
```json
{
    "timestamp": "YYYYMMDD_HHMMSS",
    "model": "model_name",
    "model_description": "model description",
    "prompt": "input prompt",
    "response": {
        "model": "model_name",
        "prompt": "input prompt",
        "response": "model response",
        "eval_duration": 0,
        "eval_count": 0,
        "total_duration": 0
    }
}
```

## Error Handling

- Failed evaluations are logged to `settings/ollama_evaluator.log`
- A summary of failures is saved to `model_outputs/failed_evaluations.json`

## Dependencies

This project uses the following packages:
- certifi==2025.1.31 (MPL-2.0 License)
- charset-normalizer==3.4.1 (MIT License)
- idna==3.10 (BSD-3-Clause License)
- requests==2.32.3 (Apache-2.0 License)
- tqdm==4.67.1 (MPL-2.0, MIT Licenses)
- urllib3==2.3.0 (MIT License)

All dependencies are compatible with this project's MIT License. See each package's documentation for their full license terms.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

For information about third-party dependency licenses, please see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
