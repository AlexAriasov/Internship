import random
import sys
sys.path.append("")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from dotenv import load_dotenv
from huggingface_hub import login
from minicons import scorer

load_dotenv()  # Load variables from .env
token = os.getenv("HF_TOKEN")  # Read token from environment
print(f"Token is: {token}")
login(token, add_to_git_credential=True)

# Load model and tokenizer
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")
scorer = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device='cuda')

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def get_model_response(prompt):
    output = generator(prompt, max_new_tokens=20, do_sample=True)[0]['generated_text']
    return output[len(prompt):].strip()

def randomized_choice_options(num_choices):
    choice_options = list(map(chr, range(65, 91)))
    return np.random.choice(choice_options, num_choices, replace=False)


def compute_logprobs(prefixes, queries, model, tokenizer):
    logprobs = []
    for prefix, query in zip(prefixes, queries):
        full_input = prefix + query
        inputs = tokenizer(full_input, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to("cuda")
        query_ids = input_ids[0][len(prefix_ids[0]):]

        log_probs = F.log_softmax(logits[0][:-1], dim=-1)
        start = len(prefix_ids[0]) - 1  # offset for logits predicting next token

        selected = []
        for i, token_id in enumerate(query_ids):
            logprob = log_probs[start + i, token_id]
            selected.append(logprob)
        selected = torch.tensor(selected)
        print(selected)
        logprobs.append(selected.mean().item())
    return logprobs

###Experiment1

# general task instructions
task_instructions = (
    "In this experiment, you will see images of different situations and answer questions about them.\n"
    "On each trial you will see a display of three creatures (monsters or robots) with different accessories (hats or scarves).\n"
    "At the top of the display you will see a message that was sent by a previous participant who saw the same display as you and was instructed to get you to pick out one of the creatures.\n"
    "Your task is to choose the creature you think the previous participant intended you to pick by pressing a key corresponding to it.\n\n"
)

# every trial instruction
instructions_trial = (
    "The previous participant said:	{message}\n"
    "Images of creatures: \n"
    "{letter_1}: {image_1}\n"
    "{letter_2}: {image_2}\n"
    "{letter_3}: {image_3}\n\n"
    "Choose the creature you think the previous participant intended you to pick by pressing the corresponding key.\n"
    "Remember that the participant could only say one of these things:\n"
    "\"purple monster\"\n"
    "\"green monster\"\n"
    "\"red hat\"\n"
    "\"blue hat\"\n"
    "Your answer:\n"
)

# possible messages and two feature dimensions
messages = ["purple monster", "green monster", "red hat", "blue hat"]
set_1 = {"purple monster", "green monster", "robot"}
set_2 = {"red hat", "blue hat", "scarf"}

def generate_simple(message):

    # depending on the message dimension
    if message in set_1:
        target = message + " with a scarf"
        competitor_obj = random.choice(list(set_2.difference({"scarf"})))
        distractor_obj = set_2.difference({"scarf", competitor_obj}).pop()
        competitor = message + " with a " + competitor_obj
        distractor = random.choice(list(set_1.difference({message}))) + " with a " + distractor_obj
    else:
        target = "robot with a " + message
        competitor_obj = random.choice(list(set_1.difference({"robot"})))
        distractor_obj = set_1.difference({"robot", competitor_obj}).pop()
        competitor = competitor_obj + " with a " + message
        distractor = distractor_obj + " with a " + random.choice(list(set_2.difference({message})))

    return target, competitor, distractor

def generate_complex(message):

    # depending on the message dimension as well
    if message in set_1:
        target_obj = random.choice(list(set_2.difference({"scarf"})))
        target = message + " with a " + target_obj
        competitor_obj = set_2.difference({"scarf", target_obj}).pop()
        competitor = message + " with a " + competitor_obj
        distractor = "robot with a " + target_obj
    else:
        target_obj = random.choice(list(set_1.difference({"robot"})))
        target = target_obj + " with a " + message
        competitor_obj = set_1.difference({"robot", target_obj}).pop()
        competitor = competitor_obj + " with a " + message
        distractor = target_obj + " with a scarf"
    return target, competitor, distractor


def generate_ambiguous(message):
    # depending on the message dimension as well
    if message in set_1:
        target_obj = random.choice(list(set_2.difference({"scarf"})))
        target = message + " with a " + target_obj
        competitor = target
        distractor_obj = set_1.difference({"robot", message}).pop()
        distractor = distractor_obj + " with a " + random.choice(list(set_2.difference({target_obj})))
    else:
        target_obj = random.choice(list(set_1.difference({"robot"})))
        target = target_obj + " with a " + message
        competitor = target
        distractor_obj = set_2.difference({"scarf", message}).pop()
        distractor = random.choice(list(set_1.difference({target_obj}))) + " with a " + distractor_obj
    return target, competitor, distractor

def generate_unambiguous(message):
    # depending on the message dimension as well
    if message in set_1:
        target_obj = random.choice(list(set_2))
        target = message + " with a " + target_obj
        competitor = random.choice(list(set_1.difference({message}))) + " with a " + random.choice(list(set_2))
        distractor = random.choice(list(set_1.difference({message}))) + " with a " + random.choice(list(set_2.difference({target_obj})))
    else:
        target_obj = random.choice(list(set_1))
        target = target_obj + " with a " + message
        competitor = random.choice(list(set_1)) + " with a " + random.choice(list(set_2.difference({message})))
        distractor = random.choice(list(set_1.difference({target_obj}))) + " with a " + random.choice(list(set_2.difference({message})))
    return target, competitor, distractor

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trials = ["unambiguous"] * 50
    target_count = 0
    competitor_count = 0
    distractor_count = 0
    suma = 0
    for trial in trials:
        message = random.choice(messages)
        if trial == "simple":
            target, competitor, distractor = generate_simple(message)
        elif trial == "complex":
            target, competitor, distractor = generate_complex(message)
        elif trial == "ambiguous":
            target, competitor, distractor = generate_ambiguous(message)
        else:
            target, competitor, distractor = generate_unambiguous(message)

        # generate by-participant randomization of keys that correspond to the forced-choice options
        (
            obj_1,
            obj_2,
            obj_3
        ) = randomized_choice_options(3)
        obj_list = [obj_1, obj_2, obj_3]

        # create a list of images and randomly shuffle it
        pictures = [target, distractor, competitor]
        random.shuffle(pictures)

        # fill the parameters to the trial outputs
        trial_instruction = instructions_trial.format(
            message=message,
            letter_1=obj_1,
            letter_2=obj_2,
            letter_3=obj_3,
            image_1=pictures[0],
            image_2=pictures[1],
            image_3=pictures[2],
        )

        prefixes = [trial_instruction] * 6
        queries = [obj_1, obj_2, obj_3, pictures[0], pictures[1], pictures[2]]

        logs_probs = compute_logprobs(prefixes, queries, model, tokenizer)
        print(trial_instruction)
        new_logs = [logs_probs[0] + logs_probs[3], logs_probs[1] + logs_probs[4], logs_probs[2] + logs_probs[5]]
        print(logs_probs)
        print(new_logs)

        response = pictures[new_logs.index(max(new_logs))]

        if response == target:
            target_count+=1
        elif response == competitor:
            competitor_count+=1
        elif response == distractor:
            distractor_count+=1
        suma+=1
    print("RESULTS:")
    print(f"Target: {target_count}")
    print(f"Competitor: {competitor_count}")
    print(f"Distractor: {distractor_count}")

    print(f"Total trials: {suma}")

