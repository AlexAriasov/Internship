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
        selected = log_probs[range(len(query_ids)), query_ids]
        logprobs.append(selected.sum().item())
    return logprobs

def few_shots():
    prompt=task_instructions + "Firstly, you will see 4 example trials:\n\n"
    trials = ["complex", "simple", "ambiguous", "unambiguous"]
    random.shuffle(trials)
    for trial in trials:
        message = random.choice(messages)
        if trial == "simple":
            competitor, distractors, target_image, competitor_image, distractor_image = generate_simple(message)
        elif trial == "complex":
            competitor, distractors, target_image, competitor_image, distractor_image = generate_complex(message)
        elif trial == "ambiguous":
            competitor, distractors, target_image, competitor_image, distractor_image = generate_ambiguous(message)
        else:
            competitor, distractors, target_image, competitor_image, distractor_image = generate_unambiguous(message)

        # create a list of creatures and shuffle it
        creatures = [target_image, competitor_image, distractor_image]
        random.shuffle(creatures)

        # generate by-participant randomization of keys that correspond to the forced-choice options
        (
            obj_1,
            obj_2,
            obj_3,
            obj_4
        ) = randomized_choice_options(4)
        obj_list = [obj_1, obj_2, obj_3, obj_4]

        # shuffle the messages
        random.shuffle(messages)

        # fill the parameters to the trial outputs
        trial_instruction = instructions_trial.format(
            creature_1=creatures[0],
            creature_2=creatures[1],
            creature_3=creatures[2],
            creature=target_image,
            letter_1=obj_1,
            letter_2=obj_2,
            letter_3=obj_3,
            letter_4=obj_4,
            message_1=messages[0],
            message_2=messages[1],
            message_3=messages[2],
            message_4=messages[3]
        )
        prompt+=trial_instruction + obj_list[messages.index(message)] +"\n\n"
    return prompt


def randomized_choice_options(num_choices):
    choice_options = list(map(chr, range(65, 91)))
    return np.random.choice(choice_options, num_choices, replace=False)


###Experiment2

# general task instructions
task_instructions = (
    "In this experiment, you will see images of different situations and answer questions about them.\n"
    "On each trial you will see a display of three creatures (monsters or robots) with different accessories (hats or scarves).\n"
    "One of them will be highlighted.\n"
    "Your task is to get another worker to pick out the highlighted creature by sending one of four possible messages.\n\n"
)

#every trial instruction
instructions_trial = (
    "You see the following creatures:\n"
    "{creature_1}\n"
    "{creature_2}\n"
    "{creature_3}\n"
    "Highlighted creature: {creature}\n\n"
    "Your task is to get another worker to pick out the highlighted creature.\n"
    "It's not highlighted on their display.\n"
    "Choose one of the following four messages to send it to the other worker and get them to pick out the right creature."
    "The other worker knows you can only send these messages:\n"
    "{letter_1}: {message_1}\n"
    "{letter_2}: {message_2}\n"
    "{letter_3}: {message_3}\n"
    "{letter_4}: {message_4}\n"
    "Your answer: "
)

# possible messages and two feature dimensions
messages = ["purple monster", "green monster", "red hat", "blue hat"]
set_1 = {"purple monster", "green monster", "robot"}
set_2 = {"red hat", "blue hat", "scarf"}


def generate_simple(target):

    # depending on the message dimension
    if target in set_1:
        competitor = random.choice(list(set_2.difference({"scarf"})))
        target_image = target + " with a " + competitor
        competitor_image = "robot with a " + competitor
        distractors = [set_1.difference({"robot", target}).pop(), set_2.difference({"scarf", competitor}).pop()]
        distractor_image = distractors[0] + " with a " + distractors[1]
    else:
        competitor = random.choice(list(set_1.difference({"robot"})))
        target_image = competitor + " with a " + target
        competitor_image = competitor + " with a scarf"
        distractors = [set_1.difference({"robot", competitor}).pop(), set_2.difference({"scarf", target}).pop()]
        distractor_image = distractors[0] + " with a " + distractors[1]

    return competitor, distractors, target_image, competitor_image, distractor_image

def generate_complex(target):
    # depending on the message dimension as well
    if target in set_1:
        competitor = random.choice(list(set_2.difference({"scarf"})))
        target_image = target + " with a " + competitor
        competitor_image = "robot with a " + competitor
        distractors = [set_1.difference({"robot", target}).pop(), set_2.difference({"scarf", competitor}).pop()]
        distractor_image = target + " with a " + distractors[1]
    else:
        competitor = random.choice(list(set_1.difference({"robot"})))
        target_image = competitor + " with a " + target
        competitor_image = competitor + " with a scarf"
        distractors = [set_1.difference({"robot", competitor}).pop(), set_2.difference({"scarf", target}).pop()]
        distractor_image = distractors[0] + " with a " + target

    return competitor, distractors, target_image, competitor_image, distractor_image


def generate_ambiguous(target):
    # depending on the message dimension as well
    if target in set_1:
        competitor = random.choice(list(set_2.difference({"scarf"})))
        target_image = target + " with a " + competitor
        competitor_image = random.choice(list(set_1.difference({target}))) + " with a " + random.choice(list(set_2.difference({competitor})))
        distractors = [set_1.difference({"robot", target}).pop(), set_2.difference({"scarf", competitor}).pop()]
        distractor_image = random.choice(list(set_1.difference({target}))) + " with a " + random.choice(list(set_2.difference({competitor})))
    else:
        competitor = random.choice(list(set_1.difference({"robot"})))
        target_image = competitor + " with a " + target
        competitor_image = random.choice(list(set_1.difference({competitor}))) + " with a " + random.choice(list(set_2.difference({target})))
        distractors = [set_1.difference({"robot", competitor}).pop(), set_2.difference({"scarf", target}).pop()]
        distractor_image = random.choice(list(set_1.difference({competitor}))) + " with a " + random.choice(list(set_2.difference({target})))

    return competitor, distractors, target_image, competitor_image, distractor_image


def generate_unambiguous(target):
    # depending on the message dimension as well
    if target in set_1:
        competitor = random.choice(list(set_2.difference({"scarf"})))
        target_image = target + " with a scarf"
        competitor_image = random.choice(list(set_1)) + " with a " + competitor
        distractors = [set_1.difference({"robot", target}).pop(), set_2.difference({"scarf", competitor}).pop()]
        distractor_image = random.choice(list(set_1)) + " with a " + random.choice(
            list(set_2.difference({"scarf"})))
    else:
        competitor = random.choice(list(set_1.difference({"robot"})))
        target_image = "robot with a " + target
        competitor_image = competitor + " with a " + random.choice(list(set_2))
        distractors = [set_1.difference({"robot", competitor}).pop(), set_2.difference({"scarf", target}).pop()]
        distractor_image = random.choice(list(set_1.difference({"robot"}))) + " with a " + random.choice(list(set_2))

    return competitor, distractors, target_image, competitor_image, distractor_image

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trials = ["unambiguous"]*50
    target_count = 0
    competitor_count = 0
    distractor_count = 0
    suma = 0
    for trial in trials:
        prompt = few_shots()
        message = random.choice(messages)
        if trial == "simple":
            competitor, distractors, target_image, competitor_image, distractor_image = generate_simple(message)
        elif trial == "complex":
            competitor, distractors, target_image, competitor_image, distractor_image = generate_complex(message)
        elif trial == "ambiguous":
            competitor, distractors, target_image, competitor_image, distractor_image = generate_ambiguous(message)
        else:
            competitor, distractors, target_image, competitor_image, distractor_image = generate_unambiguous(message)

        # create a list of creatures and shuffle it
        creatures = [target_image, competitor_image, distractor_image]
        random.shuffle(creatures)

        # generate by-participant randomization of keys that correspond to the forced-choice options
        (
            obj_1,
            obj_2,
            obj_3,
            obj_4
        ) = randomized_choice_options(4)
        obj_list = [obj_1, obj_2, obj_3, obj_4]

        # shuffle the messages
        random.shuffle(messages)

        # fill the parameters to the trial outputs
        trial_instruction = instructions_trial.format(
            creature_1=creatures[0],
            creature_2=creatures[1],
            creature_3=creatures[2],
            creature=target_image,
            letter_1=obj_1,
            letter_2=obj_2,
            letter_3=obj_3,
            letter_4=obj_4,
            message_1=messages[0],
            message_2=messages[1],
            message_3=messages[2],
            message_4=messages[3]
        )
        prefixes = [prompt + trial_instruction] * 8
        queries = [obj_1, obj_2, obj_3, obj_4, messages[0], messages[1], messages[2], messages[3]]

        logs_probs = scorer.conditional_score(prefixes, queries)
        print(prompt + trial_instruction)
        new_logs = [logs_probs[0] + logs_probs[4], logs_probs[1] + logs_probs[5], logs_probs[2] + logs_probs[6], logs_probs[3] + logs_probs[7]]
        print(logs_probs)
        print(new_logs)

        response = messages[new_logs.index(max(new_logs))]
        print(response)
        print(message)

        if response == message:
            target_count+=1
        elif response == competitor:
            competitor_count+=1
        elif response in distractors:
            distractor_count+=1
        suma+=1

    print("RESULTS:")
    print(f"Target: {target_count}")
    print(f"Competitor: {competitor_count}")
    print(f"Distractor: {distractor_count}")
    print(f"Total trials: {suma}")