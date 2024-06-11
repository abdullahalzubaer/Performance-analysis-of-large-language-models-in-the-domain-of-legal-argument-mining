
# Note on PROMPT_TEMPLATE_4_SHOT_COT - the 4 shot is chosen from PROMPT_TEMPLATE_4SHOT, it will allow us to compare the peformance between 
# PROMPT_TEMPLATE_4SHOT and PROMPT_TEMPLATE_4_SHOT_COT.


PROMPT_TEMPLATE_ZEROSHOT_WITH_INSTRUCTION=f"""In this task, you will be given a text and your goal is to classify the text as either "conclusion" or "non-conclusion" based on the definitions below. The texts are from the Decisions and Judgements categories of the European Court of Human Rights (ECHR).

"conclusion": In the context of argumentation in case law, a "conclusion" is the final decision or judgment made by the Commission or Court. It is often supported by one or more non-conclusions. The conclusion is the result of the argumentative process and is the central point that the argument is trying to establish.

"non-conclusion": In the context of argumentation in case law, a "non-conclusion" refers to the statements, facts, or assertions that provide the basis/reason for a conclusion. They are the reasons given to support the final decision of the Commission or Court. They form the building blocks of the argumentative structure leading to the conclusion.

Text to classify:[TEST_TEXT]
Classification:"""


# This prompt can be used for any 2shot example that are 
# similar_sentence, disimilar_sentence, random_sentence
PROMPT_TEMPLATE_2SHOT_WITH_INSTRUCTION_SEMANTICALLY_CHOSEN="""In this task, you will be given a text and your goal is to classify the text as either "conclusion" or "non-conclusion" based on the definitions below. The texts are from the Decisions and Judgements categories of the European Court of Human Rights (ECHR).

"conclusion": In the context of argumentation in case law, a "conclusion" is the final decision or judgment made by the Commission or Court. It is often supported by one or more non-conclusions. The conclusion is the result of the argumentative process and is the central point that the argument is trying to establish.

"non-conclusion": In the context of argumentation in case law, a "non-conclusion" refers to the statements, facts, or assertions that provide the basis/reason for a conclusion. They are the reasons given to support the final decision of the Commission or Court. They form the building blocks of the argumentative structure leading to the conclusion.

Below are examples of texts that are correctly classified as "conclusion"/"non-conclusion".

Example:[EXAMPLE_1_TEXT]
Classification:[EXAMPLE_1_LABEL]

Example:[EXAMPLE_2_TEXT]
Classification:[EXAMPLE_2_LABEL]

Text to classify:[TEST_TEXT]
Classification:

"""



# This prompt can be used for any 4shot example that are 
# similar_sentence, disimilar_sentence, random_sentence
PROMPT_TEMPLATE_4SHOT_WITH_INSTRUCTION_SEMANTICALLY_CHOSEN="""In this task, you will be given a text and your goal is to classify the text as either "conclusion" or "non-conclusion" based on the definitions below. The texts are from the Decisions and Judgements categories of the European Court of Human Rights (ECHR).

"conclusion": In the context of argumentation in case law, a "conclusion" is the final decision or judgment made by the Commission or Court. It is often supported by one or more non-conclusions. The conclusion is the result of the argumentative process and is the central point that the argument is trying to establish.

"non-conclusion": In the context of argumentation in case law, a "non-conclusion" refers to the statements, facts, or assertions that provide the basis/reason for a conclusion. They are the reasons given to support the final decision of the Commission or Court. They form the building blocks of the argumentative structure leading to the conclusion.

Below are examples of texts that are correctly classified as "conclusion"/"non-conclusion".

Example:[EXAMPLE_1_TEXT]
Classification:[EXAMPLE_1_LABEL]

Example:[EXAMPLE_2_TEXT]
Classification:[EXAMPLE_2_LABEL]

Example:[EXAMPLE_3_TEXT]
Classification:[EXAMPLE_3_LABEL]

Example:[EXAMPLE_4_TEXT]
Classification:[EXAMPLE_4_LABEL]

Text to classify:[TEST_TEXT]
Classification:

"""

'''
Below I have removed this sentence for answering the reviwer 1 question
" The texts are from the Decisions and Judgements categories of the European Court of Human Rights (ECHR)."

Originally it was like this 

"In this task, you will be given a text and your goal is to classify the text as either "conclusion" or "non-conclusion" based on the definitions below. The texts are from the Decisions and Judgements categories of the European Court of Human Rights (ECHR)."

14.Oct.2023
'''

# This prompt can be used for any 8shot example that are 
# similar_sentence, disimilar_sentence, random_sentence
PROMPT_TEMPLATE_8SHOT_WITH_INSTRUCTION_SEMANTICALLY_CHOSEN="""In this task, you will be given a text and your goal is to classify the text as either "conclusion" or "non-conclusion" based on the definitions below.

"conclusion": In the context of argumentation in case law, a "conclusion" is the final decision or judgment made by the Commission or Court. It is often supported by one or more non-conclusions. The conclusion is the result of the argumentative process and is the central point that the argument is trying to establish.

"non-conclusion": In the context of argumentation in case law, a "non-conclusion" refers to the statements, facts, or assertions that provide the basis/reason for a conclusion. They are the reasons given to support the final decision of the Commission or Court. They form the building blocks of the argumentative structure leading to the conclusion.

Below are examples of texts that are correctly classified as "conclusion"/"non-conclusion".

Example:[EXAMPLE_1_TEXT]
Classification:[EXAMPLE_1_LABEL]

Example:[EXAMPLE_2_TEXT]
Classification:[EXAMPLE_2_LABEL]

Example:[EXAMPLE_3_TEXT]
Classification:[EXAMPLE_3_LABEL]

Example:[EXAMPLE_4_TEXT]
Classification:[EXAMPLE_4_LABEL]

Example:[EXAMPLE_5_TEXT]
Classification:[EXAMPLE_5_LABEL]

Example:[EXAMPLE_6_TEXT]
Classification:[EXAMPLE_6_LABEL]

Example:[EXAMPLE_7_TEXT]
Classification:[EXAMPLE_7_LABEL]

Example:[EXAMPLE_8_TEXT]
Classification:[EXAMPLE_8_LABEL]

Text to classify:[TEST_TEXT]
Classification:

"""



def create_prompt_conclusion(test_text,
                  *args, 
                  zeroshot_with_instruction=False,
                  twoshot_with_instruction=False,
                  fourshot_with_instruction=False,
                  eightshot_with_instruction=False
                 ):
    
    list_length = len(args)
    if zeroshot_with_instruction:
        return PROMPT_TEMPLATE_ZEROSHOT_WITH_INSTRUCTION.replace("[TEST_TEXT]", test_text)
    
    
    elif list_length == 2 and twoshot_with_instruction:
        prompt = PROMPT_TEMPLATE_2SHOT_WITH_INSTRUCTION_SEMANTICALLY_CHOSEN
        for i in range(1, list_length + 1):
            text, label, _ = args[i - 1]
            prompt = prompt.replace(f'[EXAMPLE_{i}_TEXT]', text).replace(f'[EXAMPLE_{i}_LABEL]', label)
        prompt = prompt.replace('[TEST_TEXT]', test_text)
        return prompt
    
    elif list_length == 4 and fourshot_with_instruction:
        prompt = PROMPT_TEMPLATE_4SHOT_WITH_INSTRUCTION_SEMANTICALLY_CHOSEN
        for i in range(1, list_length + 1):
            text, label, _ = args[i - 1]
            prompt = prompt.replace(f'[EXAMPLE_{i}_TEXT]', text).replace(f'[EXAMPLE_{i}_LABEL]', label)
        prompt = prompt.replace('[TEST_TEXT]', test_text)
        return prompt

    elif list_length == 8 and eightshot_with_instruction:
        prompt = PROMPT_TEMPLATE_8SHOT_WITH_INSTRUCTION_SEMANTICALLY_CHOSEN
        for i in range(1, list_length + 1):
            text, label, _ = args[i - 1]
            prompt = prompt.replace(f'[EXAMPLE_{i}_TEXT]', text).replace(f'[EXAMPLE_{i}_LABEL]', label)
        prompt = prompt.replace('[TEST_TEXT]', test_text)
        return prompt

    

    else:
        raise ValueError("""Please provide a valid flag:
        'zeroshot_with_instruction',
        'twoshot_with_instruction'
        'fourshot_with_instruction',
        'eightshot_with_instruction'
        """)
