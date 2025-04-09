from sentence_transformers import SentenceTransformer, util
import time
import difflib
import json
import re
from tqdm import tqdm


def read_data(file_address):
    try:
        with open(file_address, "r", encoding='utf-8') as file:
            data = json.load(file)  # 一次性解析整个文件
            ids = [sample.get("id") for sample in data]
            fix_codes = [sample.get("FixCode") for sample in data]
            bug_codes = [sample.get("BugCode") for sample in data]
            bugtypes = [sample.get("bugType") for sample in data]
            clean_codes = [sample.get("CleanCode") for sample in data]
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return [], [], [], []
    except FileNotFoundError:
        print(f"File not found: {file_address}")
        return [], [], [], []
    except IOError as e:
        print(f"Error reading file: {e}")
        return [], [], [], []

    print('Number of samples is ' + str(len(ids)))
    print('Number of fix codes is ' + str(len(fix_codes)))
    print('Number of bug codes is ' + str(len(bug_codes)))
    print('Number of bug types is ' + str(len(bugtypes)))

    return ids, fix_codes, bug_codes, bugtypes, clean_codes


def get_data():
    training_bug, training_fix, training_labels, training_cleans, test_bug, test_fix, test_labels, test_cleans = [], [], [], [], [], [], [], []
    train_file_address = 'train_data.json'
    cur_ids, cur_fixs, cur_bugs, cur_labels, cur_cleans = read_data(train_file_address)
    training_bug += cur_bugs
    training_fix += cur_fixs
    training_labels += cur_labels
    training_cleans += cur_cleans
    test_file_address = 'val_data.json'
    cur_ids, cur_fixs, cur_bugs, cur_labels, cur_cleans = read_data(test_file_address)
    test_bug += cur_bugs
    test_fix += cur_fixs
    test_labels += cur_labels
    test_cleans += cur_cleans
    return training_bug, training_fix, training_labels, training_cleans, test_bug, test_fix, test_labels, test_cleans


def tokenize(code_str):
    code_str = str(code_str)
    code_str = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', code_str)
    code_str = re.sub(r'[\.\,\;\:\(\)\{\}\[\]]', ' ', code_str)
    code_str = re.sub(r'\s+', ' ', code_str)
    tokens = re.findall(r'[a-z]+|[A-Z][a-z]*|[0-9]+|[^\w\s]+', code_str)
    for i in range(len(tokens)):
        if i > 0 and tokens[i - 1].islower() and tokens[i].isupper():
            tokens[i] = tokens[i].lower()
    return tokens


def count_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common_elements = set1.intersection(set2)
    return len(common_elements)


def pre_process_samples_token(test_codes, training_codes):
    test_codes_embeddings, training_codes_embeddings = [], []
    st = time.time()
    for i in range(len(test_codes)):
        test_code = test_codes[i]
        code1_emb = tokenize(test_code)
        test_codes_embeddings.append(code1_emb)
    ed = time.time()
    print('Test code embedding generate finish!')
    print(str(ed - st))
    for i in range(len(training_codes)):
        train_code = training_codes[i]
        code1_emb = tokenize(train_code)
        training_codes_embeddings.append(code1_emb)
    print('Training code embedding generate finish!')
    with open('sim_token.txt', 'w') as fp:
        for i in range(len(test_codes)):
            test_code_embedding = test_codes_embeddings[i]
            sim_scores = []
            for j in range(len(training_codes)):
                train_code_embedding = training_codes_embeddings[j]
                score = count_common_elements(test_code_embedding, train_code_embedding)
                sim_scores.append(score)
            sorted_indexes = [i for i, v in sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)]
            for val in sorted_indexes[:10]:
                fp.write(str(val) + " ")
            fp.write('\n')


model = SentenceTransformer("Kwaipilot/OASIS-code-embedding-1.5B")


def generate_diff(clean, bug):
    diff = difflib.unified_diff(clean.splitlines(),
                                bug.splitlines(),
                                lineterm='')
    return '\n'.join(diff)


def generate_cross_embedding(main_code, diff_code, diff_feature):
    # Combine main code with diff feature
    final_input = f"{main_code} [SEP] {diff_feature}"
    return model.encode(final_input, convert_to_tensor=True)


def pre_process_samples_semantic(test_cleans, test_bug, training_cleans, training_bug, training_fix):
    # Initialize lists to store embeddings
    test_embeddings, training_embeddings = [], []

    # Generate embeddings for test codes (clean code + diff)
    st = time.time()
    for i in tqdm(range(len(test_cleans)), desc='Generating Test Code Embeddings'):
        clean_code = test_cleans[i]
        bug_code = test_bug[i]

        # Generate diff between clean and bug code
        diff_feature = generate_diff(clean_code, bug_code)

        # Generate enhanced embedding (clean code + diff)
        test_emb = generate_cross_embedding(clean_code, bug_code, diff_feature)
        test_embeddings.append(test_emb)

    ed = time.time()
    print('Test code embedding generation finish!')
    print(f'Time taken: {str(ed - st)} seconds')

    # Generate embeddings for training codes (fix code + diff)
    for i in tqdm(range(len(training_fix)), desc='Generating Training Code Embeddings'):
        fix_code = training_fix[i]
        bug_code = training_bug[i]
        clean_code = training_cleans[i]

        # Generate diff between fix and bug code
        diff_feature = generate_diff(clean_code, bug_code)

        # Generate enhanced embedding (fix code + diff)
        train_emb = generate_cross_embedding(fix_code, bug_code, diff_feature)
        training_embeddings.append(train_emb)

    print('Training code embedding generation finish!')

    # Calculate similarity scores and write results to file
    with open('sim_semantic_multiv_cross_oasis_1.5.txt', 'w') as fp:
        for i in tqdm(range(len(test_embeddings)), desc='Calculating Similarities'):
            test_embedding = test_embeddings[i]
            sim_scores = []

            for j in range(len(training_embeddings)):
                train_embedding = training_embeddings[j]
                hits = util.semantic_search(test_embedding, train_embedding)[0]
                top_hit = hits[0]
                score = top_hit['score']
                sim_scores.append(score)

            # Sort indexes by similarity scores in descending order
            sorted_indexes = [index for index, value in sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)]

            # Write the top 10 similar indices to the file
            for val in sorted_indexes[:10]:
                fp.write(str(val) + " ")
            fp.write('\n')


if __name__ == '__main__':
    training_bug, training_fix, training_labels, training_cleans, test_bug, test_fix, test_labels, test_cleans = get_data()
    # pre_process_samples_token(test_bug, training_bug)
    pre_process_samples_semantic(test_bug, training_bug)

