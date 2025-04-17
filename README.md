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
   - Doubao-1.5-pro (configured in `client = OpenAI()`)
   - Gemini-2.0-flash (configured in `genai.configure()`)
2. Proper endpoint URLs for both services

## Usage

### Pre-process
Preprocess the dataset to obtain the similarity scores for each piece of code in the test set. Alternatively, you can directly use the file we obtained, `sim_semantic_multiv_cross_oasis_1.5.txt`, and skip this step.
```bash
python preprocess.py
```
Then, you will get the `sim_semantic_multiv_cross_oasis_1.5.txt` representing the similarities based on semantics.
### Command Line Arguments
The command format and parameter descriptions are as follows:
```bash
python demo.py --mode <mode> [--retrieve <method>] [--example_num <number>]
```
**Options:**
- `--mode`: Required. Choose between:
  - `Classification` - Only classify defects
  - `Repair` - Only repair defects
  - `c_and_r` - Combined classification and repair
- `--retrieve`: Optional. Choose retrieval method:
  - `random` - Random example retrieval
  - `semantic` - Semantic similarity-based retrieval
- `--example_num`: Optional. Number of examples to use (default: 1)

### Replication of RQ1
To reproduce the semantic-base results in RQ1, please run the following command:
```bash
python demo.py --mode Repair --retrieve semantic --example_num 5
```
Then you can get a file named `semantic_repair_results.csv`, which records the defect repair accuracy.

To reproduce the random-base results in RQ1, please run the following command:
```bash
python demo.py --model Repair --retrieve random --example_num 5
```
Then you can get a file named `random_repair_results.csv`, which records the defect repair accuracy.
The successfully repaired code can be found in `matched_answers.csv`.
### Replication of RQ2
To reproduce the results in RQ2, please run the following command:
```bash
python demo.py --mode Prediction --example_num 3
```
Then you can get a file named `prediction_result.txt`, which records the results of defect prediction.
### Replication of RQ3
To reproduce the results in RQ3, please run the following command:
```bash
python demo.py --mode Classification --retrieve semantic --example_num 3
```
Then you can get a file named `semantic_categorization_result.txt`, which records the results of defect prediction.
### Replication of pipeline performance
```bash
python demo.py --mode c_and_r --example_num 5
```
