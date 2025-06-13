# tasks.py

# A dictionary containing info about each task:
TASKS = {
    "boolean_expressions": {
        "generation_length": 1,
        "task_prompt": "Evaluate the result of a random Boolean expression.",
        "answer_format": "Answer with only 'True' or 'False'.",
        "choices": 2,
    },
    "causal_judgement": {
        "generation_length": 1,
        "task_prompt": "Answer questions about causal attribution.",
        "answer_format": "Answer with only 'Yes' or 'No'.",
        "choices": 2,
    },
    "date_understanding": {
        "generation_length": 3,
        "task_prompt": "Infer the date from context.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 6,
    },
    "disambiguation_qa": {
        "generation_length": 3,
        "task_prompt": "Clarify the meaning of sentences with ambiguous pronouns.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 3,
    },
    "dyck_languages": {
        "generation_length": 3,
        "task_prompt": "Correctly close a Dyck-n word.",
        "answer_format": "Answer with only a sequence of space-separated parentheses.",
        "choices": None,
    },
    "formal_fallacies": {
        "generation_length": 1,
        "task_prompt": "Distinguish deductively valid arguments from formal fallacies.",
        "answer_format": "Answer with only 'valid' or 'invalid'.",
        "choices": 2,
    },
    "geometric_shapes": {
        "generation_length": 3,
        "task_prompt": "Name geometric shapes from their SVG paths.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 11,
    },
    "hyperbaton": {
        "generation_length": 3,
        "task_prompt": "Order adjectives correctly in English sentences.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 2,
    },
    "logical_deduction_five_objects": {
        "generation_length": 3,
        "task_prompt": "A logical deduction task which requires deducing the order of a sequence of objects.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 5,
    },
    "logical_deduction_seven_objects": {
        "generation_length": 3,
        "task_prompt": "A logical deduction task which requires deducing the order of a sequence of objects.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 7,
    },
    "logical_deduction_three_objects": {
        "generation_length": 3,
        "task_prompt": "A logical deduction task which requires deducing the order of a sequence of objects.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 3,
    },
    "movie_recommendation": {
        "generation_length": 3,
        "task_prompt": "Recommend movies similar to the given list of movies.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 4,
    },
    "multistep_arithmetic_two": {
        "generation_length": 3,
        "task_prompt": "Solve multi-step arithmetic problems.",
        "answer_format": "Answer with only an integer.",
        "choices": None,
    },
    "navigate": {
        "generation_length": 1,
        "task_prompt": "Given a series of navigation instructions, determine whether one would end up back at the starting point.",
        "answer_format": "Answer with only 'Yes' or 'No'.",
        "choices": 2,
    },
    "object_counting": {
        "generation_length": 2,
        "task_prompt": "Questions that involve enumerating objects and asking the model to count them.",
        "answer_format": "Answer with only an integer.",
        "choices": None,
    },
    "penguins_in_a_table": {
        "generation_length": 3,
        "task_prompt": "Answer questions about a table of penguins and their attributes.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 5,
    },
    "reasoning_about_colored_objects": {
        "generation_length": 3,
        "task_prompt": "Answer extremely simple questions about the colors of objects on a surface.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 18,
    },
    "ruin_names": {
        "generation_length": 3,
        "task_prompt": "Select the humorous edit that 'ruins' the input movie or musical artist name.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 4,
    },
    "salient_translation_error_detection": {
        "generation_length": 3,
        "task_prompt": "Detect the type of error in an English translation of a German source sentence.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 6,
    },
    "snarks": {
        "generation_length": 3,
        "task_prompt": "Determine which of two sentences is sarcastic.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 2,
    },
    "sports_understanding": {
        "generation_length": 1,
        "task_prompt": "Determine whether an artificially constructed sentence relating to sports is plausible or not.",
        "answer_format": "Answer with only 'yes' or 'no'.",
        "choices": 2,
    },
    "temporal_sequences": {
        "generation_length": 3,
        "task_prompt": "Answer questions about which times certain events could have occurred.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 4,
    },
    "tracking_shuffled_objects_five_objects": {
        "generation_length": 3,
        "task_prompt": "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 5,
    },
    "tracking_shuffled_objects_seven_objects": {
        "generation_length": 3,
        "task_prompt": "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 7,
    },
    "tracking_shuffled_objects_three_objects": {
        "generation_length": 3,
        "task_prompt": "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
        "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
        "choices": 3,
    },
    "web_of_lies": {
        "generation_length": 1,
        "task_prompt": "Evaluate a random boolean function expressed as a word problem.",
        "answer_format": "Answer with only 'Yes' or 'No'.",
        "choices": 2,
    },
    "word_sorting": {
        "generation_length": 50,
        "task_prompt": "Sort a list of words.",
        "answer_format": "Answer with only a sequence of words.",
        "choices": None,
    },
}
