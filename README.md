# CoFi-JIT

## Defect Categories

The tool recognizes the following defect types:

1. CHANGE_IDENTIFIER
2. CHANGE_NUMERAL
3. SWAP_BOOLEAN_LITERAL
4. CHANGE_MODIFIER
5. DIFFERENT_METHOD_SAME_ARGS
6. OVERLOAD_METHOD_MORE_ARGS
7. OVERLOAD_METHOD_DELETED_ARGS
8. CHANGE_CALLER_IN_FUNCTION_CALL
9. SWAP_ARGUMENTS
10. CHANGE_OPERATOR
11. CHANGE_UNARY_OPERATOR
12. CHANGE_OPERAND
13. MORE_SPECIFIC_IF
14. LESS_SPECIFIC_IF
15. ADD_THROWS_EXCEPTION
16. DELETE_THROWS_EXCEPTION
17. CLEAN (no defect)

## Requirements

- Python 3.x
- Required packages:
  - openai
  - google-generativeai
  - numpy
  - scikit-learn
  - argparse
  - json
  - csv
  - re
  - os
  - time
  - random

## Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Before running the tool, ensure you have:

1. Valid API keys for:
   - OpenAI (configured in `client = OpenAI()`)
   - Google Generative AI (configured in `genai.configure()`)
2. Proper endpoint URLs for both services

## Usage

### Command Line Arguments

```bash
python demo.py --model <mode> [--retrieve <method>] [--example_num <number>]
```

**Options:**
- `--model`: Required. Choose between:
  - `Classification` - Only classify defects
  - `Repair` - Only repair defects
  - `c_and_r` - Combined classification and repair
- `--retrieve`: Optional. Choose retrieval method:
  - `random` - Random example retrieval
  - `semantic` - Semantic similarity-based retrieval
- `--example_num`: Optional. Number of examples to use (default: 1)

### Input Files

The tool expects the following JSON files in the working directory:

1. `train_data.json` - Training dataset
2. `test_data.json` - Test dataset
3. `history_data.json` - For storing classification results
4. Similarity files (e.g., `sim_semantic_*.txt`) for semantic retrieval

### Output Files

The tool generates several output files:

1. `category_stats_log.txt` - Detailed classification statistics
2. `prompts.txt` - Generated prompts for analysis
3. `matched_answers.csv` - Correctly repaired samples
4. `mismatched_answers.csv` - Incorrectly repaired samples
5. `re_results.csv` - Repair accuracy results
6. `ans.txt` - Raw repair outputs

## Examples

1. **Classification with semantic retrieval:**
   ```bash
   python demo.py --model Classification --retrieve semantic --example_num 3
   ```

2. **Repair with random retrieval:**
   ```bash
   python demo.py --model Repair --retrieve random --example_num 2
   ```

3. **Combined classification and repair:**
   ```bash
   python demo.py --model c_and_r --example_num 1
   ```

## Notes

- The tool implements rate limiting and retry mechanisms for API calls
- Code normalization is performed to handle formatting differences
- Extensive logging is provided for debugging and analysis
- The classification process maintains state between runs using `history_data.json`

## Limitations

- Requires properly formatted input JSON files
- API keys and endpoints need to be manually configured
- Performance depends on the quality and quantity of training examples
- Some defect types may be harder to detect than others
