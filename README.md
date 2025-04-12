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

### Pre-process
First, preprocess the dataset to get the similarity scores for each code in the test set.
```bash
python preprocess.py
```

### Command Line Arguments
First, preprocess the dataset to get the similarity scores for each code in the test set.
```bash
python demo.py --model <mode> [--retrieve <method>] [--example_num <number>]
```
Then, you will get the "sim_token.txt" and "sim_semantic.txt", representing the similarities based on semantic.

**Options:**
- `--model`: Required. Choose between:
  - `Classification` - Only classify defects
  - `Repair` - Only repair defects
  - `c_and_r` - Combined classification and repair
- `--retrieve`: Optional. Choose retrieval method:
  - `random` - Random example retrieval
  - `semantic` - Semantic similarity-based retrieval
- `--example_num`: Optional. Number of examples to use (default: 1)

### Replication of RQ1
```bash
python demo.py --mode Repair --retrieve semantic --example_num 3
```

```bash
python demo.py --model Repair --retrieve random --example_num 5
```

### Replication of RQ2
```bash
python demo.py --mode Prediction --example_num 3
```

### Replication of RQ3
```bash
python demo.py --mode Classification --retrieve semantic --example_num 3
```

### Replication of pipeline performance
```bash
python demo.py --mode Classification --retrieve semantic --example_num 3
```

### Input Files

The tool expects the following JSON files in the working directory:

1. `train_data.json` - Training dataset
2. `test_data.json` - Test dataset
3. `history_data.json` - For storing classification results
4. Similarity files (e.g., `sim_semantic_*.txt`) for semantic retrieval


## Notes

- The tool implements rate limiting and retry mechanisms for API calls
- Code normalization is performed to handle formatting differences
- Extensive logging is provided for debugging and analysis
- The classification process maintains state between runs using `history_data.json`
