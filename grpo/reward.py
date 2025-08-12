from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
import json
import os
import re
from trl import GRPOConfig, GRPOTrainer, SFTTrainer, SFTConfig

def extract_all_boxed_content(text):
    results = []
    start = 0

    while True:
        # Find the next occurrence of \boxed{
        start = text.find(r"\boxed{", start)
        if start == -1:
            break  # No more \boxed{ found

        brace_count = 0
        result = []
        i = start

        while i < len(text):
            char = text[i]
            result.append(char)

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

            # Stop when the braces are balanced
            if brace_count == 0 and result[-1] == '}':
                break

            i += 1

        # Append the matched content
        results.append(''.join(result))
        start = i + 1  # Move past the current match to find the next

    return results

def juggle_verify(pr_answer, gt_answer):         
    if pr_answer:
        prediction = pr_answer[-1]
        if prediction and '\\boxed' in prediction:
            prediction = prediction.replace('\\boxed{', '').rstrip('}')

    accept = False           
    gold = parse(f"${gt_answer}$", extraction_config=[LatexExtractionConfig()])
    answer = parse(f"${prediction}$", extraction_config=[LatexExtractionConfig()])
    if verify(gold, answer):
        accept = True
    else:
        num_pred = re.findall(r"[-+]?\d*\.\d+|\d+", prediction)
        num_gt = re.findall(r"[-+]?\d*\.\d+|\d+", gt_answer)
        if num_pred and num_gt and float(num_pred[0]) == float(num_gt[0]):
            accept = True

    return accept

def is_format(completion):
    pattern1 = r"^<think>\n"
    pattern2 = r"^<answer>\n"
    if re.match(pattern1, completion, re.DOTALL):
        return "think"
    elif re.match(pattern2, completion):
        return "answer"
    else:
        return "none"
    
def reword_compute(completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    #responses = [completion["response"] for completion in completions]
    #answers = [completion["answer"] for completion in completions]
    reword = []
    for response,answer1 in zip(responses, answer):
        pr_answer = extract_all_boxed_content(response)
        correct = juggle_verify(pr_answer, answer1)
        if correct:
            if is_format(response) == 'think':
                reword.append(1.5)
            elif is_format(response) == 'answer':
                reword.append(2)
            else:
                reword.append(0.5)
        else: 
            if is_format(response) == 'think':
                reword.append(0)
            elif is_format(response) == 'answer':
                reword.append(-1)
            else:
                reword.append(0)
            
    return reword

def format_reword(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    #responses = [completion["response"] for completion in completions]
    reword = []
    pattern1 = r"^<think>\n"
    pattern2 = r"^<answer>\n"
    for response in responses:
        if re.match(pattern1, response, re.DOTALL) or re.match(pattern2, response):
            reword.append(0.5)
        else:
            reword.append(0.0)
    
    return reword
