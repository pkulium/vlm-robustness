import sys
import os
import re
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from base_prompt import get_prompt_shot
# pre_prompt: ""
# post_prompt: "\nProvide the answer by selecting the option's letter from the given choices first, then give a brief reasoning for your selection."


def sqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    context, question, choices = doc["hint"], doc["question"], doc["choices"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    if lmms_eval_specific_kwargs["format"] == "default":
        if context:
            context = f"Context: {context}\n"

        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        shot_prompt = get_prompt_shot(lmms_eval_specific_kwargs.get('shot_number', 0))
        return shot_prompt + f"####{pre_prompt}{context}{question}\n{choices_str}{post_prompt}"
    elif lmms_eval_specific_kwargs["format"] == "qwen_vl":
        prompt = "####Context: {}\nQuestion: {}\nOptions: {}\nAnswer:"
        context = context if context else "N/A"
        prompt = prompt.format(context, question, choices_str)
        shot_prompt = get_prompt_shot(lmms_eval_specific_kwargs.get('shot_number', 0))
        return shot_prompt + prompt    
    else:
        raise ValueError(f"Unknown prompt format: {lmms_eval_specific_kwargs}")


def sqa_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def sqa_doc_to_target(doc):
    len_choices = len(doc["choices"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    return options[doc["answer"]]


# def sqa_process_results(doc, results):
#     # I know this is weird, but it's how llava parse it.
#     target = sqa_doc_to_target(doc)
#     pred = results[0]
#     # print(f"pred:{pred}")
#     if pred == target or pred[0] == target:
#         return {"exact_match": 1.0}
#     # pattern: ^[A-Z]\. .*
#     if len(pred) >= 2 and pred[0].isupper() and pred[1] == ".":
#         result = 1.0 if pred[0] == target else 0.0
#         return {"exact_match": result}
#     return {"exact_match": 0.0}

def sqa_process_results(doc, results):
    # I know this is weird, but it's how llava parse it.
    target = sqa_doc_to_target(doc).strip().lower()
    
    pred = results[0].strip().lower()
    if pred == target or (len(pred) != 0 and pred[0] == target):
        return {"exact_match": 1.0}
    # pattern: ^[A-Z]\. .*
    if len(pred) >= 2 and pred[0].isupper() and pred[1] == ".":
        result = 1.0 if pred[0] == target else 0.0
        return {"exact_match": result}
    return {"exact_match": 0.0}



def sqa_process_results_few_shot(doc, results):
    # I know this is weird, but it's how llava parse it.
    target = sqa_doc_to_target(doc)
    pred = results[0]
    pattern = re.compile(r'Answer: The answer is ([A-Z]).')
    res = pattern.findall(pred)
    if len(res) == 1:
        pred = res[0]  # 'A', 'B', ...

    if pred == target:
        return {"exact_match": 1.0}
    # pattern: ^[A-Z]\. .*
    if len(pred) >= 2 and pred[0].isupper() and pred[1] == ".":
        result = 1.0 if pred[0] == target else 0.0
        return {"exact_match": result}
    return {"exact_match": 0.0}
