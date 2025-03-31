import csv
import openai
from openai import OpenAI
from openai import OpenAIError, RateLimitError, APIError
import json
import os
import re
import numpy as np
import time
import random
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
import google.generativeai as genai

client = OpenAI(
    api_key="f4cba9ca-5cbc-4997-997b-76399faa1c59",
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

genai.configure(
    api_key='sk-TB5xahA0SAB8WZAD0KxZYQAf7IsFSb3gLK2aDp1ZNi7efr2p',
    transport="rest",
    client_options={"api_endpoint": "https://api.openai-proxy.org/google"},
)


def completion_with_backoff(**kwargs):
    return client.completions.create(**kwargs)


def chat_completion_with_backoff(**kwargs):
    max_retries = 5  # 最大重试次数
    retry_delay = 2  # 初始重试延迟时间（秒）

    for attempt in range(max_retries):
        try:
            # 尝试调用 API
            response = client.chat.completions.create(**kwargs)
            return response
        except RateLimitError as e:
            # 处理速率限制错误
            print(f"速率限制错误: {e}. 尝试 {attempt + 1}/{max_retries}...")
            time.sleep(retry_delay * (attempt + 1))  # 指数退避
        except APIError as e:
            # 处理 API 错误
            print(f"API 错误: {e}. 尝试 {attempt + 1}/{max_retries}...")
            time.sleep(retry_delay * (attempt + 1))  # 指数退避
        except OpenAIError as e:
            # 处理其他 OpenAI 错误
            print(f"OpenAI 错误: {e}")
            raise  # 如果是未知错误，直接抛出
        except Exception as e:
            # 处理其他未知错误
            print(f"未知错误: {e}")
            raise  # 如果是未知错误，直接抛出

    # 如果重试次数用尽仍未成功，抛出异常
    raise Exception(f"重试 {max_retries} 次后仍未成功，请检查网络或 API 配置。")


prompt_pred = """
**Task:**  
You are an intelligent assistant specialized in Java. Your task is to determine whether the modified code introduces a certain type of specific defect or confirm that no defect has been introduced. 

**Defect Categories:**  
Evaluate the modified code based on the following predefined defect categories:  

1.CHANGE_IDENTIFIER: An identifier in an expression was replaced with another one of the same type.
2.CHANGE_NUMERAL: A numeric literal was replaced with another one.
3.SWAP_BOOLEAN_LITERAL: A Boolean literal (True/False) was replaced with its opposite.
4.CHANGE_MODIFIER: A variable, function, or class was declared with incorrect or missing modifiers.
5.DIFFERENT_METHOD_SAME_ARGS: This pattern checks whether the wrong function was called. Functions with similar names and the same signature are usual pitfall for developers.
6.OVERLOAD_METHOD_MORE_ARGS: An overloaded function with fewer arguments was called.
7.OVERLOAD_METHOD_DELETED_ARGS: An overloaded function with more arguments was called.
8.CHANGE_CALLER_IN_FUNCTION_CALL: The caller object in a function call was replaced with another one of the same type.
9.SWAP_ARGUMENTS: This pattern checks whether a function was called with two of its arguments swapped. When multiple arguments of a function are of the same type, if developers do not accurately remember what each argument represents then they can easily swap two such arguments without realizing it.
10.CHANGE_OPERATOR: When using logical operators in logical expressions, errors in the use of logical operators may occur due to reasons such as copy-pasting or similar variable names. For example, mistakenly writing "==" as "!=". Such errors can be identified by checking the logical judgment relationships of variables in the logical expressions and comparing them with the expected business logic.
11.CHANGE_UNARY_OPERATOR: This pattern is used to check whether a unary Boolean operator has been accidentally replaced by another unary Boolean operator of the same type. For example, developers might often forget to use the "!" operator in Boolean expressions or mistakenly add the "!" operator.
12.CHANGE_OPERAND: Check whether one of the operands in a operation in the program is incorrect. For example, the developer might have confused an operand in a operation with a similarly named operand from another expression.When checking for this type of defect, you should thoroughly read the context around the code modification and, by integrating your Java knowledge, determine whether the usage of the operands is reasonable.  
13.MORE_SPECIFIC_IF: An extra condition (&& operand) was added to an if statement’s condition.
14.LESS_SPECIFIC_IF: An extra condition (|| operand) was added to an if statement’s condition.
15.ADD_THROWS_EXCEPTION: Do NOT classify any defect into this category.The category related to exception declarations.
16.DELETE_THROWS_EXCEPTION: Do NOT classify any defect into this category.These category related to exception declarations.

**Examples:** 
"""

prompt_binary = """
**Task:**  
Your task is to identify what kind of defect has been introduced by the modified code.

**Defect Categories:**  
Evaluate the modified code based on the following predefined defect categories:  

1.CHANGE_IDENTIFIER: An identifier in an expression was replaced with another one of the same type.
2.CHANGE_NUMERAL: A numeric literal was replaced with another one.
3.SWAP_BOOLEAN_LITERAL: A Boolean literal (True/False) was replaced with its opposite.
4.CHANGE_MODIFIER: A variable, function, or class was declared with incorrect or missing modifiers.
5.DIFFERENT_METHOD_SAME_ARGS: This pattern checks whether the wrong function was called. Functions with similar names and the same signature are usual pitfall for developers.
6.OVERLOAD_METHOD_MORE_ARGS: An overloaded function with fewer arguments was called.
7.OVERLOAD_METHOD_DELETED_ARGS: An overloaded function with more arguments was called.
8.CHANGE_CALLER_IN_FUNCTION_CALL: The caller object in a function call was replaced with another one of the same type.
9.SWAP_ARGUMENTS: This pattern checks whether a function was called with two of its arguments swapped. When multiple arguments of a function are of the same type, if developers do not accurately remember what each argument represents then they can easily swap two such arguments without realizing it.
10.CHANGE_OPERATOR: When using logical operators in logical expressions, errors in the use of logical operators may occur due to reasons such as copy-pasting or similar variable names. For example, mistakenly writing "==" as "!=". Such errors can be identified by checking the logical judgment relationships of variables in the logical expressions and comparing them with the expected business logic.
11.CHANGE_UNARY_OPERATOR: This pattern is used to check whether a unary Boolean operator has been accidentally replaced by another unary Boolean operator of the same type. For example, developers might often forget to use the "!" operator in Boolean expressions or mistakenly add the "!" operator.
12.CHANGE_OPERAND: Check whether one of the operands in a operation in the program is incorrect. For example, the developer might have confused an operand in a operation with a similarly named operand from another expression.When checking for this type of defect, you should thoroughly read the context around the code modification and, by integrating your Java knowledge, determine whether the usage of the operands is reasonable.  
13.MORE_SPECIFIC_IF: An extra condition (&& operand) was added to an if statement’s condition.
14.LESS_SPECIFIC_IF: An extra condition (|| operand) was added to an if statement’s condition.
15.ADD_THROWS_EXCEPTION: The category related to exception declarations.
16.DELETE_THROWS_EXCEPTION: These category related to exception declarations.

**Examples:** 
"""

requirement = """
**Instructions:**  
- Carefully analyze the modified code in the context of the original, with a particular focus on the parts where the code has been modified.  
- Use the provided bug categories and the accompanying examples to determine whether a bug has been introduced.  
- Consider the overall functionality and logical correctness of the code.  
- Use your Java knowledge to make a well-reasoned judgment.  

**Response Format:**  
- Return your judgment result in plain text without any additional explanations, symbols or formatting.
- If no defect has been introduced, return CLEAN.
- If you are quite certain that a defect has been introduced, return only one category of defect that you think is the most likely.
"""

requirement_binary = """
**Instructions:**  
- Carefully analyze the modified code in the context of the original, with a particular focus on the parts where the code has been modified.  
- Use the provided bug categories and the accompanying examples to determine whether a bug has been introduced.  
- Consider the overall functionality and logical correctness of the code.  
- Use your Java knowledge to make a well-reasoned judgment.  

**Response Format:**  
- Only reply with the defect name.
- Return your judgment result in plain text without any additional explanations, symbols or formatting.
- If you are quite certain that a defect has been introduced, return only one category of defect that you think is the most likely.
"""

prompt_change_identifier_used = "Your primary role is to identify and fix bugs in Java code, particularly focusing on " \
                                "issues related to incorrect identifier usage. Review the provided Java code snippet, " \
                                "identify any issues related to incorrect identifier usage (e.g., using a different " \
                                "identifier than intended due to copy-pasting or similar naming), correct the code by " \
                                "replacing the incorrect identifier with the intended one, and return only the " \
                                "corrected code without any additional text or explanations."

prompt_change_numeric_literal = "Your primary role is to identify and fix bugs in Java code, particularly focusing on " \
                                "issues related to incorrect numeric literal usage. Analyze Java code snippets, " \
                                "detect and correct errors where a numeric literal is mistakenly replaced with " \
                                "another numeric value, and provide precise, corrected code without additional " \
                                "explanations or comments."

prompt_change_boolean_literal = "Your primary role is to identify and fix bugs in Java code, particularly focusing on " \
                                "issues related to incorrect Boolean literal usage. Review the provided Java code " \
                                "snippet, identify any issues related to incorrect Boolean literal usage (e.g., " \
                                "`true` replaced with `false` or vice versa), correct the code by replacing the " \
                                "incorrect Boolean literal with the intended value, and return only the corrected " \
                                "code without any additional text or explanations."

prompt_change_modifier = "Your primary role is to identify and fix bugs in Java code, particularly focusing on issues " \
                         "related to incorrect modifier usage in variable, function, or class declarations.Review the " \
                         "provided Java code snippet, identify any issues related to incorrect or missing modifiers " \
                         "in variable, function, or class declarations, correct the code by adding or replacing the " \
                         "appropriate modifiers, and return only the corrected code without any additional text or " \
                         "explanations."

prompt_wrong_function_name = "Your primary task is to detect and correct issues related to incorrect function names, " \
                             "particularly when a function with the same parameter list but the wrong name is called. " \
                             "Review the provided Java code snippet, identify instances where a function with the " \
                             "correct parameter list but the wrong name is called, replace the incorrect function " \
                             "name with the correct one, and return only the corrected code without any additional " \
                             "explanations or comments."

prompt_same_function_more_args = "Regarding 'Same Function More Args', it determines if an overloaded function with " \
                                 "more arguments is erroneously called, which can confuse developers. Inspect the " \
                                 "code and correct any occurrences, following the example structure.Return only the " \
                                 "corrected code without any additional explanation or commentary."

prompt_same_function_less_args = "For 'Same Function Less Args', the aim is to find if an overloaded function with " \
                                 "fewer arguments is wrongly called, as developers may forget an argument. Scan the " \
                                 "Java code and fix these errors as shown in the example, and return only the " \
                                 "corrected code without any additional text or explanations."

prompt_same_function_change_caller = "Your primary role is to identify and fix bugs in Java code, particularly " \
                                     "focusing on issues related to incorrect caller object usage in function calls. " \
                                     "Review the provided Java code snippet, identify any issues related to incorrect " \
                                     "caller object usage in function calls (e.g., using the wrong object due to " \
                                     "copy-pasting or similar variable names), correct the code by replacing the " \
                                     "incorrect caller object with the intended one, and return only the corrected " \
                                     "code without any additional text or explanations."

prompt_same_function_swap_args = "Your primary role is to identify and fix bugs in Java code, particularly focusing " \
                                 "on issues related to incorrect argument order in function calls. Review the " \
                                 "provided Java code snippet, identify any issues related to swapped arguments in " \
                                 "function calls (e.g., incorrect order of arguments of the same type), correct the " \
                                 "code by swapping the arguments back to their intended order, and return only the " \
                                 "corrected code without any additional text or explanations."

prompt_change_binary_operator = "Your primary role is to identify and fix bugs in Java code, particularly focusing on " \
                                "issues related to incorrect binary operator usage. Review the provided Java code " \
                                "snippet, identify any issues related to incorrect binary operator usage (e.g., " \
                                "using the wrong comparison or arithmetic operator), correct the code by replacing " \
                                "the incorrect binary operator with the intended one, and return only the corrected " \
                                "code without any additional text or explanations."

prompt_change_unary_operator = "Your primary role is to identify and fix bugs in Java code, particularly focusing on " \
                               "issues related to unary operator usage.Review the provided Java code snippet, " \
                               "identify any issues related to unary operator usage, correct the code by fixing the " \
                               "identified issues, and return only the corrected code without any additional text or " \
                               "explanations."

prompt_change_operand = "Your primary task is to analyze and correct errors in binary operations, particularly " \
                        "focusing on cases where one of the operands might be incorrect. When provided with a code " \
                        "snippet containing a bug, you will analyze the code, identify the incorrect operand in the " \
                        "binary operation. Errors in the code are usually limited to a single location, " \
                        "so avoid making excessive modifications. Return only the corrected code without any " \
                        "additional explanation or commentary."

prompt_more_specific_if = "For 'More Specific If', it checks if an extra '&& operand' condition has been added in an " \
                          "if statement's condition. Analize the Java code and correct it according to the example. " \
                          "Return only the corrected code without any additional text or explanations."

prompt_less_specific_if = "Regarding 'Less Specific If', it verifies if an extra '|| operand' condition should be " \
                          "added in an if statement's condition. Inspect the Java code and correct it following the " \
                          "example. Return only the corrected code without any additional text or explanations."

prompt_missing_throws_exception = "The 'Missing Throws Exception' bug checks if a throws clause needs to be added in " \
                                  "a function declaration. Examine the code and add it if necessary.Return only the " \
                                  "corrected code without any additional explanation or commentary."

prompt_delete_throws_exception = "For 'Delete Throws Exception', it checks if a throws clause has been wrongly " \
                                 "deleted from a function declaration. Analyze the Java code and correct it as per " \
                                 "the example.Return only the corrected code without any additional explanation or " \
                                 "commentary."

prompt_clean = "The provided code is already free of known bugs."

MAX_LENGTH = 57344


def load_data(file_path):
    """加载历史数据"""
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:  # 检查文件是否存在且不为空
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                # 如果文件内容不是有效的 JSON，返回空数据
                return {'category_stats': {}, 'clean_indices': [], 'y_true': [], 'y_pred': []}
    # 如果文件不存在或为空，返回空数据
    return {'category_stats': {}, 'clean_indices': [], 'y_true': [], 'y_pred': []}


def save_data(data, file_path):
    """保存数据到文件"""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def update_category_stats(old_stats, new_stats):
    """更新 category_stats"""
    for category, stats in new_stats.items():
        if category in old_stats:
            old_stats[category]['tp'] += stats['tp']
            old_stats[category]['fp'] += stats['fp']
            old_stats[category]['fn'] += stats['fn']
        else:
            old_stats[category] = stats
    return old_stats


def update_clean_indices(old_indices, new_indices):
    return old_indices + new_indices


def test_count(test_bug, test_labels, test_category, training_labels, sim_file, l_length, l_count, l_range):
    data_file = 'history_data.json'  # 保存历史数据的文件

    all_categories = ['DIFFERENT_METHOD_SAME_ARGS', 'OVERLOAD_METHOD_MORE_ARGS', 'CHANGE_UNARY_OPERATOR',
                      'OVERLOAD_METHOD_DELETED_ARGS', 'CHANGE_NUMERAL', 'CHANGE_IDENTIFIER', 'SWAP_BOOLEAN_LITERAL',
                      'CHANGE_CALLER_IN_FUNCTION_CALL', 'CHANGE_MODIFIER', 'CHANGE_OPERATOR', 'CHANGE_OPERAND',
                      'LESS_SPECIFIC_IF', 'MORE_SPECIFIC_IF', 'SWAP_ARGUMENTS', 'DELETE_THROWS_EXCEPTION',
                      'ADD_THROWS_EXCEPTION', 'CLEAN']

    # 初始化每种类型的计数
    category_stats = {
        category: {'tp': 0, 'fp': 0, 'fn': 0}
        for category in all_categories
    }

    clean_indices = []  # 用于记录被初步筛选掉应该跳过的序号
    y_true = []  # 记录真实的标签
    y_pred = []  # 记录预测的标签

    # 加载历史数据
    history_data = load_data(data_file)
    old_category_stats = history_data.get('category_stats', {})
    old_clean_indices = history_data.get('clean_indices', [])
    old_y_true = history_data.get('y_true', [])
    old_y_pred = history_data.get('y_pred', [])
    print(len(old_y_true))

    with open(sim_file, 'r', encoding='utf-8') as fpp:
        lines = fpp.readlines()

        for i in range(len(test_bug)):
            if i in old_clean_indices:
                continue
            sim_ids = lines[i].split(" ")

            # 统计示例的类别
            category_count = {category: 0 for category in all_categories}

            for cnt in range(l_range):
                index = int(sim_ids[cnt])
                category = training_labels[index]
                category_count[category] += 1

            max_count = max(category_count.values())
            most_common_types = [category for category, count in category_count.items() if count == max_count]
            true_label = test_labels[i]

            # 记录真实标签和预测标签
            if test_category in most_common_types and max_count >= l_count and len(most_common_types) <= l_length:
                if true_label == test_category:
                    category_stats[test_category]['tp'] += 1
                else:
                    category_stats[true_label]['fn'] += 1
                    category_stats[test_category]['fp'] += 1
                clean_indices.append(i)
                y_true.append(true_label)
                y_pred.append(test_category)

    # 更新历史数据
    updated_category_stats = update_category_stats(old_category_stats, category_stats)
    updated_clean_indices = update_clean_indices(old_clean_indices, clean_indices)
    updated_y_true = old_y_true + y_true
    updated_y_pred = old_y_pred + y_pred

    # 保存更新后的数据
    save_data({
        'category_stats': updated_category_stats,
        'clean_indices': updated_clean_indices,
        'y_true': updated_y_true,
        'y_pred': updated_y_pred
    }, data_file)


def test_classify(test_bug, test_labels, test_clean, training_bug, training_labels, training_clean,
                  example_num, old_category_stats, y_true, y_pred):
    # 初始化每个类别的 TP、FP、TN、FN
    cat_l = ['DIFFERENT_METHOD_SAME_ARGS', 'OVERLOAD_METHOD_MORE_ARGS', 'CHANGE_UNARY_OPERATOR',
             'OVERLOAD_METHOD_DELETED_ARGS', 'CHANGE_NUMERAL', 'CHANGE_IDENTIFIER', 'SWAP_BOOLEAN_LITERAL',
             'CHANGE_CALLER_IN_FUNCTION_CALL', 'CHANGE_MODIFIER', 'CHANGE_OPERATOR', 'CHANGE_OPERAND',
             'LESS_SPECIFIC_IF', 'MORE_SPECIFIC_IF', 'SWAP_ARGUMENTS', 'DELETE_THROWS_EXCEPTION',
             'ADD_THROWS_EXCEPTION', 'CLEAN']

    # rate_limit_per_minute = 1000
    # delay = 60.0 / rate_limit_per_minute

    sim_file = 'sim_semantic_multiv_cross_oasis_1.5.txt'

    with open('category_stats_log.txt', 'w', encoding='utf-8') as log_file, open('prompts.txt', 'w',
                                                                                 encoding='utf-8') as fp1, open(
        sim_file, 'r', encoding='utf-8') as fpp:
        lines = fpp.readlines()
        num = len(test_bug)
        for i in range(num):
            if i in skip:
                continue
            test_clean_code = test_clean[i]
            test_code = test_bug[i]
            sim_ids = lines[i].split(" ")

            new_prompt = prompt_binary

            for cnt in range(example_num):
                index = int(sim_ids[cnt])
                example_prompt = "\n#Example{}:\n".format(cnt)
                example_prompt += "\n#The code before modification\n" + training_clean[index]
                example_prompt += "\n#The code after modification\n" + training_bug[index]
                example_prompt += "\n#The judgment is:\n" + training_labels[index]

                if len(new_prompt) + len(example_prompt) > MAX_LENGTH:
                    break
                new_prompt += example_prompt

            new_prompt += "\n**The code to be analyzed.**\n"
            new_prompt += "#The code before modification\n" + test_clean_code + "\n"
            new_prompt += "#The code after modification\n" + test_code + "\n"

            new_prompt += requirement_binary

            fp1.write(f"Prompt for test sample {i}:\n{new_prompt}\n\n" + ",".join(sim_ids))

            # response = chat_completion_with_backoff(
            #     model="ep-20250208180008-r4mcz",
            #     messages=[
            #         {"role": "system", "content": "You are an intelligent assistant specialized in Java."},
            #         {"role": "user", "content": f"{new_prompt}"},
            #     ]
            # )
            # cur_ans = response.choices[0].message.content
            # 使用 genai 生成内容
            model = genai.GenerativeModel('gemini-2.0-flash')
            # 设置 messages
            response = model.generate_content(f"{new_prompt}")
            # 去除空格和换行符
            cur_ans = response.text.replace(" ", "").replace("\n", "")

            if cur_ans not in cat_l:
                print(i)
                print(cur_ans)

            true_label = test_labels[i]
            y_true.append(true_label)
            y_pred.append(cur_ans)
            for category in cat_l:
                if cur_ans == category and true_label == category:
                    old_category_stats[category]['tp'] += 1
                elif cur_ans == category and true_label != category:
                    old_category_stats[category]['fp'] += 1
                elif cur_ans != category and true_label == category:
                    old_category_stats[category]['fn'] += 1

            log_file.write(f"After test sample {i}:\n")
            log_file.write(json.dumps(old_category_stats, indent=4) + '\n\n')

    with open('category_stats_log.txt', 'a', encoding='utf-8') as log_file:
        # 将标签转换为二分类任务的形式（CLEAN -> 0, 其他 -> 1）
        b_y_preds = [0 if i == 'CLEAN' else 1 for i in y_pred]
        b_y_trues = [0 if i == 'CLEAN' else 1 for i in y_true]

        # 计算二分类任务的评估指标
        if len(np.unique(b_y_trues)) == 2:  # 确保有两个类别
            b_recall = recall_score(b_y_trues, b_y_preds)
            b_precision = precision_score(b_y_trues, b_y_preds)
            b_f1 = f1_score(b_y_trues, b_y_preds)
        else:
            b_recall = b_precision = b_f1 = None  # 如果只有一个类别，设置为 None

        # 计算每个类别的 precision 和 recall
        precision_recall = {}
        for category, stats in old_category_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision_recall[category] = {'precision': precision, 'recall': recall}

        # 将结果写入日志文件
        log_file.write("Final precision and recall:\n")
        log_file.write(json.dumps(precision_recall, indent=4) + '\n')

        # 写入二分类任务的评估结果
        log_file.write("\nBinary classification metrics:\n")
        log_file.write(f"Precision: {b_precision}\n")
        log_file.write(f"Recall: {b_recall}\n")
        log_file.write(f"F1 Score: {b_f1}\n")


MAX_LENGTH_64K = 65535


def test_classify_repair(test_bug, test_fix, test_labels, test_clean, training_bug, training_fix, training_labels,
                         training_clean,
                         category, example_num):
    sim_file = 'sim_semantic_oasis.txt'
    acc = []
    with open('category_stats_log.txt', 'w', encoding='utf-8') as log_file, open('prompts.txt', 'w',
                                                                                 encoding='utf-8') as fp1, open(
        sim_file, 'r', encoding='utf-8') as fpp:
        lines = fpp.readlines()

        for i in range(len(test_bug)):
            if test_labels[i] != category:
                continue
            test_clean_code = test_clean[i]
            test_code = test_bug[i]
            sim_ids = lines[i].split(" ")
            new_prompt = prompt_pred

            for cnt in range(example_num):
                index = int(sim_ids[cnt])
                example_prompt = "\n#Example{}:\n".format(cnt)
                example_prompt += "\n#The code before modification\n" + training_clean[index]
                example_prompt += "\n#The code after modification\n" + training_bug[index]
                example_prompt += "\n#The judgment is:\n" + training_labels[index]

                if len(new_prompt) + len(example_prompt) > MAX_LENGTH:
                    break
                new_prompt += example_prompt

            new_prompt += "\n#The code to be analyzed.\n"
            new_prompt += "#The code before modification\n" + test_clean_code + "\n"
            new_prompt += "#The code after modification\n" + test_code + "\n"
            new_prompt += "#Please check the modified code in combination with the description and examples of the " \
                          "above bug types, and also consider the overall functionality of the code. Use your Java " \
                          "knowledge to comprehensively and deeply determine whether there are bugs in this " \
                          "modification. Finally, only return the bug type that you think is the most likely. If you " \
                          "think the modification has not introduced any bugs, then return CLEAN. \n"
            new_prompt += "#only return the type you think is most likely. Do not add any extra symbols or " \
                          "explanation.\n"
            new_prompt += "#Your judgement is:"
            fp1.write(f"Prompt for test sample {i}:\n{new_prompt}\n\n" + ",".join(sim_ids))

            # time.sleep(delay)
            # response = chat_completion_with_backoff(
            #     model="ep-20250208180008-r4mcz",
            #     messages=[
            #         {"role": "system", "content": "You are an intelligent assistant specialized in Java."},
            #         {"role": "user", "content": f"{new_prompt}"},
            #     ]
            # )
            # cur_ans = response.choices[0].message.content
            model = genai.GenerativeModel('gemini-2.0-flash')
            # 设置 system 和 messages
            response = model.generate_content(f"{new_prompt}")
            # 去除空格和换行符
            cur_ans = response.text.replace(" ", "").replace("\n", "")

            true_label = test_labels[i]
            print(cur_ans)

            if cur_ans == true_label:
                rate_limit_per_minute = 1000
                delay = 60.0 / rate_limit_per_minute
                time.sleep(delay)

                prompt_lists = {
                    "DIFFERENT_METHOD_SAME_ARGS": prompt_wrong_function_name,
                    "OVERLOAD_METHOD_MORE_ARGS": prompt_same_function_more_args,
                    "CHANGE_UNARY_OPERATOR": prompt_change_unary_operator,
                    "OVERLOAD_METHOD_DELETED_ARGS": prompt_same_function_less_args,
                    "CHANGE_NUMERAL": prompt_change_numeric_literal,
                    "CHANGE_IDENTIFIER": prompt_change_identifier_used,
                    "SWAP_BOOLEAN_LITERAL": prompt_change_boolean_literal,
                    "CHANGE_CALLER_IN_FUNCTION_CALL": prompt_same_function_change_caller,
                    "CHANGE_MODIFIER": prompt_change_modifier,
                    "CHANGE_OPERATOR": prompt_change_binary_operator,
                    "CHANGE_OPERAND": prompt_change_operand,
                    "LESS_SPECIFIC_IF": prompt_less_specific_if,
                    "MORE_SPECIFIC_IF": prompt_more_specific_if,
                    "SWAP_ARGUMENTS": prompt_same_function_swap_args,
                    "DELETE_THROWS_EXCEPTION": prompt_delete_throws_exception,
                    "ADD_THROWS_EXCEPTION": prompt_missing_throws_exception,
                    "CLEAN": prompt_clean
                }
                new_prompt = prompt_lists.get(category, prompt_clean)
                for cnt in range(example_num):
                    index = int(sim_ids[cnt])
                    re_ex_prompt = ("\n#Example{}:\n".format(cnt))
                    # new_prompt += ("\n#The code before introducing bug:\n" + training_clean[index])
                    re_ex_prompt += ("\n#The code after introducing bug:\n" + training_bug[index])
                    re_ex_prompt += ("\n# The fixed code is:\n" + training_fix[index])

                    if len(new_prompt) + len(re_ex_prompt) > MAX_LENGTH_64K:
                        break
                    new_prompt += re_ex_prompt

                new_prompt += "\n#Code to Be Fixed:\n"
                # new_prompt += "#Code Before Introducing the Bug\n" + test_clean_code + "\n"
                new_prompt += "#Code After Introducing the Bug\n" + test_code + "\n"
                new_prompt += "#The fixed code is:\n"

                fp1.write(f"Prompt for test sample {i}:\n{new_prompt}\n\n" + ",".join(sim_ids))

                response = chat_completion_with_backoff(
                    model="ep-20250208173934-k5vl6",
                    messages=[
                        {"role": "system",
                         "content": "You are an intelligent assistant specialized in Java code repair. Your primary "
                                    "role is to analyze and fix bugs in Java code snippets provided by users. You are "
                                    "equipped with extensive knowledge of Java programming, common coding errors, "
                                    "and best practices for debugging and code optimization."},
                        {"role": "user", "content": f"{new_prompt}"},
                    ]
                )
                cur_ans = response.choices[0].message.content
                ans = test_fix[i]
                temp_ans = cur_ans.replace("```java", "").replace("```", "")
                temp_ans = normalize_code_regex(temp_ans)
                tt_ans = normalize_code_regex(ans)
                acc.append(temp_ans == tt_ans)
                print(sum(acc))

        print(sum(acc))


def normalize_code_regex(code):
    # 去除多余的空格和换行符
    code = re.sub(r'\s+', ' ', code).strip()
    return code


def test_sample_retrieve(test_bug, test_fix, test_labels, test_clean, training_bug, training_fix, training_clean,
                         category, example_num):
    acc = []
    sim_file = 'sim_semantic_multiv_cross_oasis_1.5.txt'
    with open('matched_answers.csv', 'a', newline='', encoding="utf-8") as cr_csv, \
            open(sim_file, 'r', encoding="utf-8") as fpp, \
            open('mismatched_answers.csv', 'a', newline='', encoding="utf-8") as wr_csv:

        wr_csvwriter = csv.writer(wr_csv)
        wr_csvwriter.writerow(['Test Code', 'Expected Fix', 'Actual Fix'])
        cr_csvwriter = csv.writer(cr_csv)
        cr_csvwriter.writerow(['id', 'Test Code', 'Expected Fix', 'Actual Fix'])
        lines = fpp.readlines()

        for i in range(len(test_bug)):
            if test_labels[i] != category:
                continue
            test_clean_code = test_clean[i]
            test_code = test_bug[i]
            ans = test_fix[i]
            sim_ids = lines[i].split(" ")
            prompt_lists = {
                "DIFFERENT_METHOD_SAME_ARGS": prompt_wrong_function_name,
                "OVERLOAD_METHOD_MORE_ARGS": prompt_same_function_more_args,
                "CHANGE_UNARY_OPERATOR": prompt_change_unary_operator,
                "OVERLOAD_METHOD_DELETED_ARGS": prompt_same_function_less_args,
                "CHANGE_NUMERAL": prompt_change_numeric_literal,
                "CHANGE_IDENTIFIER": prompt_change_identifier_used,
                "SWAP_BOOLEAN_LITERAL": prompt_change_boolean_literal,
                "CHANGE_CALLER_IN_FUNCTION_CALL": prompt_same_function_change_caller,
                "CHANGE_MODIFIER": prompt_change_modifier,
                "CHANGE_OPERATOR": prompt_change_binary_operator,
                "CHANGE_OPERAND": prompt_change_operand,
                "LESS_SPECIFIC_IF": prompt_less_specific_if,
                "MORE_SPECIFIC_IF": prompt_more_specific_if,
                "SWAP_ARGUMENTS": prompt_same_function_swap_args,
                "DELETE_THROWS_EXCEPTION": prompt_delete_throws_exception,
                "ADD_THROWS_EXCEPTION": prompt_missing_throws_exception,
                "CLEAN": prompt_clean
            }
            new_prompt = prompt_lists.get(category, prompt_clean)

            for cnt in range(example_num):
                index = int(sim_ids[cnt])
                new_prompt += ("\n#Example{}:\n".format(cnt))
                new_prompt += ("\n#The code after introducing bug:\n" + training_bug[index])
                new_prompt += ("\n# The fixed code is:\n" + training_fix[index])

            new_prompt += "\n#Code to Be Fixed:\n"
            new_prompt += "#Code After Introducing the Bug\n" + test_code + "\n"
            new_prompt += "#The fixed code is:\n"

            response = chat_completion_with_backoff(
                model="ep-20250208180008-r4mcz",
                messages=[
                    {"role": "system",
                     "content": "You are an intelligent assistant specialized in Java code repair. Your primary role "
                                "is to analyze and fix bugs in Java code snippets provided by users. You are equipped "
                                "with extensive knowledge of Java programming, common coding errors, "
                                "and best practices for debugging and code optimization."},
                    {"role": "user", "content": f"{new_prompt}"},
                ]
            )
            cur_ans = response.choices[0].message.content
            t_ans = cur_ans.replace("```java", "").replace("```", "")
            t_ans = normalize_code_regex(t_ans)
            tt_ans = normalize_code_regex(ans)

            acc.append(t_ans == tt_ans)
            if t_ans != tt_ans:
                wr_csvwriter.writerow([test_code, ans, cur_ans.replace("```java", "").replace("```", "")])
            if t_ans == tt_ans:
                cr_csvwriter.writerow([i, test_code, ans, cur_ans.replace("```java", "").replace("```", "")])

        xmatch = round(np.mean(acc) * 100, 4)
        print(f"{category} xMatch = {xmatch}  ({sum(acc)} / {len(acc)})")

        with open('re_results.csv', 'a', newline='', encoding="utf-8") as result_file:
            result_writer = csv.writer(result_file)
            result_writer.writerow([category, xmatch, sum(acc), len(acc)])


def test_repair_random(test_bug, test_fix, test_labels, test_clean, training_bug, training_fix, training_clean,
                       category, example_num):
    acc = []
    print('random')
    with open('ans.txt', 'a') as fp, open('mismatched_answers.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['id', 'Test Code', 'Expected Fix', 'Actual Fix', 'sim_ids'])

        for i in range(210, 211):
            test_clean_code = test_clean[i]
            test_code = test_bug[i]
            ans = test_fix[i]
            if test_labels[i] != category:
                continue
            random_ids = []
            val = len(training_bug)  # len for the retrieval set
            # Generate random values
            for k in range(example_num):
                random_ids.append(random.randint(0, val - 1))
            sim_id = ','.join(map(str, random_ids))  # 使用 map 将整数转换为字符串
            prompt_lists = {
                "DIFFERENT_METHOD_SAME_ARGS": prompt_wrong_function_name,
                "OVERLOAD_METHOD_MORE_ARGS": prompt_same_function_more_args,
                "CHANGE_UNARY_OPERATOR": prompt_change_unary_operator,
                "OVERLOAD_METHOD_DELETED_ARGS": prompt_same_function_less_args,
                "CHANGE_NUMERAL": prompt_change_numeric_literal,
                "CHANGE_IDENTIFIER": prompt_change_identifier_used,
                "SWAP_BOOLEAN_LITERAL": prompt_change_boolean_literal,
                "CHANGE_CALLER_IN_FUNCTION_CALL": prompt_same_function_change_caller,
                "CHANGE_MODIFIER": prompt_change_modifier,
                "CHANGE_OPERATOR": prompt_change_binary_operator,
                "CHANGE_OPERAND": prompt_change_operand,
                "LESS_SPECIFIC_IF": prompt_less_specific_if,
                "MORE_SPECIFIC_IF": prompt_more_specific_if,
                "SWAP_ARGUMENTS": prompt_same_function_swap_args,
                "DELETE_THROWS_EXCEPTION": prompt_missing_throws_exception,
                "ADD_THROWS_EXCEPTION": prompt_delete_throws_exception,
                "CLEAN": prompt_clean
            }
            new_prompt = prompt_lists.get(category, prompt_clean)
            for cnt in range(example_num):
                new_prompt += ("\n#Example{}:\n".format(cnt))
                # new_prompt += ("\n#The code before introducing bug:\n" + training_clean[int(random_ids[cnt])])
                new_prompt += ("\n#The code after introducing bug:\n" + training_bug[int(random_ids[cnt])])
                new_prompt += ("\n# The fixed code is:\n" + training_fix[int(random_ids[cnt])])

            new_prompt += "\n#Code to Be Fixed:\n"
            # new_prompt += "#Code Before Introducing the Bug\n" + test_clean_code + "\n"
            new_prompt += "#Code After Introducing the Bug\n" + test_code + "\n"
            new_prompt += "#Note: This code has been confirmed to contain a bug, so you must make modifications " \
                          "instead of returning it unchanged. Additionally, only the most likely erroneous line " \
                          "should be modified. Just return the modified code. Do not add any extra comments or " \
                          "explanations.\n"
            new_prompt += "#The fixed code is:\n"

            response = chat_completion_with_backoff(
                model="ep-20250208180008-r4mcz",
                messages=[
                    {"role": "system",
                     "content": "You are an intelligent assistant specialized in Java code repair. Your primary role "
                                "is to analyze and fix bugs in Java code snippets provided by users. You are equipped "
                                "with extensive knowledge of Java programming, common coding errors, "
                                "and best practices for debugging and code optimization."},
                    {"role": "user", "content": f"{new_prompt}"},
                ]
            )
            cur_ans = response.choices[0].message.content
            t_ans = cur_ans.replace("```java", "").replace("```", "")
            t_ans = normalize_code_regex(t_ans)
            tt_ans = normalize_code_regex(ans)

            acc.append(t_ans == tt_ans)
            fp.write(cur_ans + '\n')

            if tt_ans != t_ans:
                csvwriter.writerow([i, test_code, ans, cur_ans, sim_id])

        xmatch = round(np.mean(acc) * 100, 4)
        print(f"xMatch = {xmatch}  ({sum(acc)} / {len(acc)})")


def read_data(file_address):
    try:
        with open(file_address, "r", encoding='utf-8') as file:
            data = json.load(file)
            ids = [sample.get("id") for sample in data]
            bugtypes = [sample.get("bugType") for sample in data]
            fix_codes = [sample.get("FixCode") for sample in data]
            bug_codes = [sample.get("BugCode") for sample in data]
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
    return ids, fix_codes, bug_codes, bugtypes, clean_codes


def get_data():
    training_bug, training_fix, training_labels, training_clean, test_bug, test_fix, test_labels, test_clean = [], [], [], [], [], [], [], []
    train_file_address = 'train_data.json'
    cur_ids, cur_fixs, cur_bugs, cur_labels, cur_cleans = read_data(train_file_address)
    training_bug += cur_bugs
    training_fix += cur_fixs
    training_labels += cur_labels
    training_clean += cur_cleans
    test_file_address = 'test_data.json'
    cur_ids, cur_fixs, cur_bugs, cur_labels, cur_cleans = read_data(test_file_address)
    test_bug += cur_bugs
    test_fix += cur_fixs
    test_labels += cur_labels
    test_clean += cur_cleans
    return training_bug, training_fix, training_labels, training_clean, test_bug, test_fix, test_labels, test_clean


def pre_filtering():
    test_count(test_bug, test_labels, 'DELETE_THROWS_EXCEPTION', training_labels, 'sim_semantic_combined.txt', 1, 0, 1)
    test_count(test_bug, test_labels, 'ADD_THROWS_EXCEPTION', training_labels, 'sim_semantic_sfr.txt', 1, 0, 1)
    test_count(test_bug, test_labels, 'SWAP_ARGUMENTS', training_labels, 'sim_semantic_sfr.txt', 1, 0, 1)
    test_count(test_bug, test_labels, 'CLEAN', training_labels, 'sim_semantic_cross.txt', 1, 0, 1)

def test_classify_random(test_bug, test_labels, test_clean, training_bug, training_labels, training_clean, example_num,
                         old_category_stats, y_true, y_pred):
    # 初始化每个类别的 TP、FP、TN、FN
    cat_l = ['DIFFERENT_METHOD_SAME_ARGS', 'OVERLOAD_METHOD_MORE_ARGS', 'CHANGE_UNARY_OPERATOR',
             'OVERLOAD_METHOD_DELETED_ARGS', 'CHANGE_NUMERAL', 'CHANGE_IDENTIFIER', 'SWAP_BOOLEAN_LITERAL',
             'CHANGE_CALLER_IN_FUNCTION_CALL', 'CHANGE_MODIFIER', 'CHANGE_OPERATOR', 'CHANGE_OPERAND',
             'LESS_SPECIFIC_IF', 'MORE_SPECIFIC_IF', 'SWAP_ARGUMENTS', 'DELETE_THROWS_EXCEPTION',
             'ADD_THROWS_EXCEPTION', 'CLEAN']

    with open('category_stats_log.txt', 'w', encoding='utf-8') as log_file, open('prompts.txt', 'w',
                                                                                 encoding='utf-8') as fp1:
        num = len(test_bug)
        for i in range(418, 452):
            if i in skip:
                continue

            test_clean_code = test_clean[i]
            test_code = test_bug[i]
            val = len(training_bug)
            random_ids = []
            for k in range(example_num):
                random_ids.append(random.randint(0, val - 1))

            new_prompt = prompt_binary

            for cnt in range(example_num):
                index = int(random_ids[cnt])
                example_prompt = "\n#Example{}:\n".format(cnt)
                example_prompt += "\n#The code before modification\n" + training_clean[index]
                example_prompt += "\n#The code after modification\n" + training_bug[index]
                example_prompt += "\n#The judgment is:\n" + training_labels[index]

                if len(new_prompt) + len(example_prompt) > MAX_LENGTH:
                    break
                new_prompt += example_prompt

            new_prompt += "\n**The code to be analyzed.**\n"
            new_prompt += "#The code before modification\n" + test_clean_code + "\n"
            new_prompt += "#The code after modification\n" + test_code + "\n"

            new_prompt += requirement_binary
            
            fp1.write(f"Prompt for test sample {i}:\n{new_prompt}\n\n" + ",".join(random_ids))

            # 使用 genai 生成内容
            model = genai.GenerativeModel('gemini-2.0-flash')
            # 设置 system 和 messages
            response = model.generate_content(f"{new_prompt}")
            # 去除空格和换行符
            cur_ans = response.text.replace(" ", "").replace("\n", "")

            if cur_ans not in cat_l:
                print(i)
                print(cur_ans)

            true_label = test_labels[i]
            y_true.append(true_label)
            y_pred.append(cur_ans)
            for category in cat_l:
                if cur_ans == category and true_label == category:
                    old_category_stats[category]['tp'] += 1
                elif cur_ans == category and true_label != category:
                    old_category_stats[category]['fp'] += 1
                elif cur_ans != category and true_label == category:
                    old_category_stats[category]['fn'] += 1

            log_file.write(f"After test sample {i}:\n")
            log_file.write(json.dumps(old_category_stats, indent=4) + '\n\n')

    with open('category_stats_log.txt', 'a', encoding='utf-8') as log_file:
        # 将标签转换为二分类任务的形式（CLEAN -> 0, 其他 -> 1）
        b_y_preds = [0 if i == 'CLEAN' else 1 for i in y_pred]
        b_y_trues = [0 if i == 'CLEAN' else 1 for i in y_true]

        # 计算二分类任务的评估指标
        if len(np.unique(b_y_trues)) == 2:  # 确保有两个类别
            b_recall = recall_score(b_y_trues, b_y_preds)
            b_precision = precision_score(b_y_trues, b_y_preds)
            b_f1 = f1_score(b_y_trues, b_y_preds)
        else:
            b_recall = b_precision = b_f1 = None  # 如果只有一个类别，设置为 None

        # 计算每个类别的 precision 和 recall
        precision_recall = {}
        for category, stats in old_category_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision_recall[category] = {'precision': precision, 'recall': recall}

        # 将结果写入日志文件
        log_file.write("Final precision and recall:\n")
        log_file.write(json.dumps(precision_recall, indent=4) + '\n')

        # 写入二分类任务的评估结果
        log_file.write("\nBinary classification metrics:\n")
        log_file.write(f"Precision: {b_precision}\n")
        log_file.write(f"Recall: {b_recall}\n")
        log_file.write(f"F1 Score: {b_f1}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--model", choices=['Classification', 'Repair', 'c_and_r'], required=True,
                        help="Mode of retrieve.")
    parser.add_argument("--retrieve", choices=['random', 'semantic'], default=None,
                        help="Retrieve mode: random or semantic.")
    parser.add_argument("--example_num", type=int, default=5, help="Number of example codes.")
    args = parser.parse_args()

    training_bug, training_fix, training_labels, training_clean, test_bug, test_fix, test_labels, test_clean = get_data()

    categories = ['DIFFERENT_METHOD_SAME_ARGS', 'OVERLOAD_METHOD_MORE_ARGS', 'CHANGE_UNARY_OPERATOR',
                  'OVERLOAD_METHOD_DELETED_ARGS', 'CHANGE_NUMERAL', 'CHANGE_IDENTIFIER', 'SWAP_BOOLEAN_LITERAL',
                  'CHANGE_CALLER_IN_FUNCTION_CALL', 'CHANGE_MODIFIER', 'CHANGE_OPERATOR', 'CHANGE_OPERAND',
                  'LESS_SPECIFIC_IF', 'MORE_SPECIFIC_IF', 'S]WAP_ARGUMENTS', 'DELETE_THROWS_EXCEPTION',
                  'ADD_THROWS_EXCEPTION', 'CLEAN']
    categories = ['LESS_SPECIFIC_IF']
    example_num = args.example_num
    pre_filtering()

    if args.model == 'Classification':
        history_data = load_data('history_data.json')
        old_category_stats = history_data.get('category_stats', {})
        skip = history_data.get('clean_indices', [])
        y_true = history_data.get('y_true', [])
        y_pred = history_data.get('y_pred', [])
        if args.retrieve == 'semantic':
            test_classify(test_bug, test_labels, test_clean, training_bug, training_labels, training_clean,
                          example_num, old_category_stats, y_true, y_pred)
        elif args.retrieve == 'random':
            test_classify_random(test_bug, test_labels, test_clean, training_bug, training_labels, training_clean,
                          example_num, old_category_stats, y_true, y_pred)
    if args.model == 'Repair':
        if args.retrieve == 'random':
            for category in categories:
                test_repair_random(test_bug, test_fix, test_labels, test_clean, training_bug, training_fix,
                                   training_clean,
                                   category, example_num)
        elif args.retrieve == 'semantic':
            for category in categories:
                test_sample_retrieve(test_bug, test_fix, test_labels, test_clean, training_bug, training_fix,
                                     training_clean,
                                     category, example_num)
    if args.model == 'c_and_r':
        for category in categories:
            test_classify_repair(test_bug, test_fix, test_labels, test_clean, training_bug, training_fix,
                                 training_labels,
                                 training_clean,
                                 category, example_num)
