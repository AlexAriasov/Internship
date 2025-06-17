import random
import sys
sys.path.append("")
json_out = []
CHARACTER_LIMIT = 32000
import numpy as np

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
    "Your answer: \n"
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
    trials = ["simple"]*12 + ["complex"]*12 + ["ambiguous"]*15 + ["unambiguous"]*27
    target_count = 0
    competitor_count = 0
    distractor_count = 0
    suma = 0
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

        response_key = ""
        if response_key == obj_1:
            response = messages[0]
        elif response_key == obj_2:
            response = messages[1]
        elif response_key == obj_3:
            response = messages[2]
        elif response_key == obj_4:
            response = messages[3]
        else:
            response = ""

        if response == message:
            target_count+=1
        elif response == competitor:
            competitor_count+=1
        elif response in distractors:
            distractor_count+=1
        suma+=1