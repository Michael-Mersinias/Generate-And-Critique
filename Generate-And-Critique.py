#!/usr/bin/env python
# coding: utf-8

# Imported Libraries.

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import string
import nltk
import re
import ast
import gc
import os
import time
import random
import openai
import seaborn as sns
from itertools import islice
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from scipy.stats import spearmanr

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)

KEY = ''

gc.collect()

# Splits each Scarecrow example into sentences and return those sentences as well as their starting and ending indices.
def find_sentence_positions(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    positions = []
    start_positions_list = []
    end_positions_list = []
    start = 0
    for sentence in sentences:
        end = start + len(sentence)
        positions.append((start, end))
        start_positions_list.append(start)
        end_positions_list.append(end)
        start = end
    return {"sentences": sentences, "positions": positions, "start_positions": start_positions_list, "end_positions": end_positions_list}

# Reads the Scarecrow dataset and conducts data preprocessing.
def get_and_preprocess_dataset(dataset_name):
    
    df_scarecrow = pd.read_csv(dataset_name)
    
    df_scarecrow["better_responses"] = [ast.literal_eval(df_scarecrow.responses[i]) for i in range(len(df_scarecrow))]
    
    sents_pos = [find_sentence_positions(i) for i in df_scarecrow.generation]
    df_scarecrow["sentences"] = [i["sentences"] for i in sents_pos]
    df_scarecrow["positions"] = [i["positions"] for i in sents_pos]
    df_scarecrow["start_positions"] = [i["start_positions"] for i in sents_pos]
    df_scarecrow["end_positions"] = [i["end_positions"] for i in sents_pos]
    
    df_scarecrow = df_scarecrow[df_scarecrow['model'] == 'gpt3'].copy()
    df_scarecrow = df_scarecrow.reset_index(drop = True)
    
    list_of_indices = [i for i in range(len(df_scarecrow))]
    
    return df_scarecrow, list_of_indices

df_scarecrow, list_of_indices = get_and_preprocess_dataset("Scarecrow_Initial_Dataset.csv")

# Creates Scarecrow examples for the Critic and returns a string to be used as part of the Critic prompt.
def assign_critic_examples(df_scarecrow):
    
    error_labels = ['Grammar and Usage', 'Redundant', 'Incoherent']

    examples_list = []
    
    # Two "Grammar and Usage" training examples (Generated Text and Human Feedback).
    
    gu_gen_text = ['A PhD student from the University of Kent in the UK claims to have discovered a clever way to explain the positive emoticons in cats.',
                   'A couple is facing criticism for their extravagant birthday party as the bewitching pair had first stripped down to fishnets and backward.']
    
    gu_error_feedback = ['Grammar and Usage Error Detection: Error Text: emoticons. Reason: The word should probably be "emotions". Error: Grammar and Usage.',
                         'Grammar and Usage Error Detection: Error Text: and backward. Reason: This phrase can simply be deleted. Error: Grammar and Usage.']    
    
    # Two "Redundant" training examples (Generated Text and Human Feedback).
    
    rd_gen_text = ['Many merchants worry about the possibility of poor service or service for certain categories of customers.', 
                   'They then made decisions based on Kondos instructions, to the extent that they created decluttered spaces and got rid of clutter and clutter-filled spaces.']
    
    rd_error_feedback = ['Redundant Error Detection: Error Text: or service. Reason: Repetition of the exact same word. Error: Redundant.',
                         'Redundant Error Detection: Error Text: and got rid of clutter and clutter-filled spaces. Reason: Repetition of the same idea using different wording. Error: Redundant.']
    
    # Four "Incoherent" training examples (Generated Text and Human Feedback) - Two are related to Self-Contradiction and two are related to Incoherence.
    
    in_gen_text = ['McDonalds is considering a design which will replace the cardboard packaging and as Mr GoreCotter said: "We recognise the concern around waste and we are now looking at a new design that minimises the plastic bag."',
                   'Mall of America plans to lay off and furlough hundreds of its employees, it has no plans to restrict the number of hours workers can work.', 
                   'Melody Mitsugi, 28, had never given her kids cheese toast before her husband drew a map of it on her toast.', 
                   'Cats naturally show anxiety and fear by at times breaking apart different parts of the brain in an attempt to keep the others from escaping.']
    
    in_error_feedback = ['Incoherent Error Detection: Error Text: plastic bag. Reason: The idea of minimizing the plastic bag contradicts the stated goal of replacing cardboard packaging. Error: Self-contradiction.',
                         'Incoherent Error Detection: Error Text: it has no plans to restrict the number of hours workers can work. Reason: Furloughed workers are explicitly restricted from working. Error: Self-contradiction.', 
                         'Incoherent Error Detection: Error Text: drew a map of it on her toast. Reason: One cannot exactly draw a map of Cheese Toast, and one probably would not draw it on toast itself. Error: Incoherent.',
                         'Incoherent Error Detection: Error Text: breaking apart different parts of the brain in an attempt to keep the others from escaping. Reason: It is difficult to even imagine what is happening in this passage. Error: Incoherent.']    
    
    
    error_gen_text_list = [gu_gen_text, rd_gen_text, in_gen_text]
    error_feedback_list = [gu_error_feedback, rd_error_feedback, in_error_feedback]
    
    # Creates a string with all the examples listed above, so it can be used to prompt the Critic.
    
    for i in range(len(error_labels)):
        
        error_example = 'Critic ChatGPT Training Examples:'
        
        error_gen_text = error_gen_text_list[i]
        error_feedback = error_feedback_list[i]
        
        for j in range(len(error_feedback)):
            
            new_example = '- Training Example ' + str(j + 1) + ': ' + '\n\n' + 'Generated Text: ' + error_gen_text[j] + '\n\n' + error_feedback[j]
            
            error_example = error_example + '\n\n' + new_example

        examples_list.append(error_example)
        
    return examples_list
    
critic_examples_list = assign_critic_examples(df_scarecrow)

# Creates Scarecrow examples for the Generator and returns a string to be used as part of the Generator prompt.
def assign_generator_examples(df_scarecrow):
    
    error_labels = ['Grammar and Usage', 'Redundant', 'Incoherent']

    examples_list = []
    
    # Two "Grammar and Usage" training examples (Generated Text, Human Feedback, Correct Text).
    
    gu_gen_text = ['A PhD student from the University of Kent in the UK claims to have discovered a clever way to explain the positive emoticons in cats.',
                   'A couple is facing criticism for their extravagant birthday party as the bewitching pair had first stripped down to fishnets and backward.']
    
    gu_error_feedback = ['Grammar and Usage Error Detection: Error Text: emoticons. Reason: The word should probably be "emotions". Error: Grammar and Usage.',
                         'Grammar and Usage Error Detection: Error Text: and backward. Reason: This phrase can simply be deleted. Error: Grammar and Usage.']    
    
    gu_error_corrected = ['A PhD student from the University of Kent in the UK claims to have discovered a clever way to explain the positive emotions in cats.',
                          'A couple is facing criticism for their extravagant birthday party as the bewitching pair had first stripped down to fishnets.']

    # Two "Redundant" training examples (Generated Text, Human Feedback, Correct Text).
    
    rd_gen_text = ['Many merchants worry about the possibility of poor service or service for certain categories of customers.', 
                   'They then made decisions based on Kondos instructions, to the extent that they created decluttered spaces and got rid of clutter and clutter-filled spaces.']
    
    rd_error_feedback = ['Redundant Error Detection: Error Text: or service. Reason: Repetition of the exact same word. Error: Redundant.',
                         'Redundant Error Detection: Error Text: and got rid of clutter and clutter-filled spaces. Reason: Repetition of the same idea using different wording. Error: Redundant.']
    
    rd_error_corrected = ['Many merchants worry about the possibility of poor service for certain categories of customers.',
                          'They then made decisions based on Kondos instructions, to the extent that they created decluttered spaces.']
    
    # Four "Incoherent" training examples (Generated Text, Human Feedback, Correct Text) - Two are related to Self-Contradiction and two are related to Incoherence.
    
    in_gen_text = ['McDonalds is considering a design which will replace the cardboard packaging and as Mr GoreCotter said: "We recognise the concern around waste and we are now looking at a new design that minimises the plastic bag."',
                   'Mall of America plans to lay off and furlough hundreds of its employees, it has no plans to restrict the number of hours workers can work.', 
                   'Melody Mitsugi, 28, had never given her kids cheese toast before her husband drew a map of it on her toast.', 
                   'Cats naturally show anxiety and fear by at times breaking apart different parts of the brain in an attempt to keep the others from escaping.']
    
    in_error_feedback = ['Incoherent Error Detection: Error Text: plastic bag. Reason: The idea of minimizing the plastic bag contradicts the stated goal of replacing cardboard packaging. Error: Self-contradiction.',
                         'Incoherent Error Detection: Error Text: it has no plans to restrict the number of hours workers can work. Reason: Furloughed workers are explicitly restricted from working. Error: Self-contradiction.', 
                         'Incoherent Error Detection: Error Text: drew a map of it on her toast. Reason: One cannot exactly draw a map of Cheese Toast, and one probably would not draw it on toast itself. Error: Incoherent.',
                         'Incoherent Error Detection: Error Text: breaking apart different parts of the brain in an attempt to keep the others from escaping. Reason: It is difficult to even imagine what is happening in this passage. Error: Incoherent.']    

    in_error_corrected = ['McDonalds is considering a design which will replace the cardboard packaging and as Mr GoreCotter said: "We recognise the concern around waste and we are now looking at a new design."',
                          'Mall of America plans to lay off and furlough hundreds of its employees.', 
                          'Melody Mitsugi, 28, had never given her kids cheese toast before.',
                          'Cats naturally show anxiety and fear by at times.']
    
    error_gen_text_list = [gu_error_corrected, rd_error_corrected, in_error_corrected]
    error_feedback_list = [gu_error_feedback, rd_error_feedback, in_error_feedback]
    error_corrected_list = [gu_error_corrected, rd_error_corrected, in_error_corrected]
    
    # Creates a string with all the examples listed above, so it can be used to prompt the Generator.
    
    for i in range(len(error_labels)):
        
        error_example = 'Generator ChatGPT Training Examples:'
        
        error_gen_text = error_gen_text_list[i]
        error_feedback = error_feedback_list[i]
        error_corrected = error_corrected_list[i]
        
        for j in range(len(error_feedback)):
            
            new_example = '- ' + error_labels[i] + ' Training Example ' + str(j + 1) + ': ' + '\n\n' + 'Original Sentence: ' + error_gen_text[j] + '\n\n' + 'Critique: ' + error_feedback[j]+ '\n\n' + 'Corrected Sentence: ' + error_corrected[j]
            
            error_example = error_example + '\n\n' + new_example

        examples_list.append(error_example)
        
    example_string = str(examples_list)[2:len(str(examples_list))-2]
        
    return example_string
    
generator_examples_list = assign_generator_examples(df_scarecrow)

# The main Critic function.
def critique_and_find_errors(curr_sentence):
    
    # Defines three Scarecrow errors.
    
    error_labels = ['Grammar and Usage', 'Redundant', 'Incoherent']
    error_definitions = ["Grammar and Usage: Errors related to missing words, extra words, and incorrect or out of order words.", 
                         "Redundant: Errors related to redundant repetition of the exact phrase or redundant repetition of the same idea using different words.", 
                         "Incoherent: Errors related to either contradiction or confusing language within the text."]
    
    # For each one of the Scarecrow errors...
    
    all_critic_response = ''
    error_found = ''
    
    for j in range(len(error_labels)):
        
        # Critic Prompt (1): 
        # System Prompt: Gives the label and the definition of a Scarecrow error to the Critic, and the purpose of detecting it in text.
        # User Prompt: Gives the Scarecrow examples to the Critic for training, and the current sentence to be evaluted by the Critic.
        
        text_to_critique = 'Generated Text: ' + curr_sentence

        critic_system_prompt = "The definition for the Scarecrow error type named " + error_labels[j] + " is the following: " + "\n\n" + error_definitions[j] + "\n\n" + "You are now Critic ChatGPT, a helpful assistant who detects " + error_labels[j] + " errors in text." 
        critic_user_prompt = critic_examples_list[j] + "\n\n" "- Test Example: " + "\n\n" + text_to_critique + "\n\n" + error_labels[j] + " Error Detection: " #  + "\n\n" + "Think about it step-by-step."
        
        critic_response = ''
        
        while len(critic_response) == 0:
            
            try:
                
                response = openai.ChatCompletion.create(
                    api_key=KEY,
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": critic_system_prompt},
                        {"role": "user", "content": critic_user_prompt}
                    ]
                )

                critic_response = response['choices'][0]['message']['content'].strip()
                                
            except Exception as e:
                
                print('API Call Error (1): ', e)
                time.sleep(1)
                critic_response  = ''
        
        
        # Critic Prompt (2): 
        # System Prompt: You are a helpful assistant (Default).
        # User Prompt: We make the LLM decide whether a specific scarecrow error is detected in the Critic's response.
        
        yes_no_response2 = ''
                
        while len(yes_no_response2) == 0:
            
            yes_no_response2 = ''
            
            try:
                
                response = openai.ChatCompletion.create(
                    api_key=KEY,
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Critic Response: " + critic_response + "\n\n" + "Does this Critic Response state the existence of any " + error_labels[j] + " error? Answer with 'Yes' or 'No'."}
                    ]
                )

                yes_no_response2 = response['choices'][0]['message']['content'].strip()
                
            except Exception as e:
                
                print('API Call Error (2): ', e)
                time.sleep(1)
                yes_no_response2  = ''
        
        # Form the full Critic response, as an aggregate of all Critic responses, for each error type that has been detected in text by the Critic.
        
        if ("yes" in yes_no_response2.lower()):
            error_found = error_found + ' ' + error_labels[j]
            all_critic_response = all_critic_response + '\n\n' + error_labels[j] +' Error Detection: ' + yes_no_response2 + ' Critic Feedback: ' + critic_response
        else:
            error_found = ''
            
            
    # Calculate the score (0-3) that corresponds to the number of detected errors, and in case at least 1 error exists, return the Critic feedback too.
        
    critic_errors = 0
    
    full_critic_feedback = 'No error detected.'

    if "Grammar and Usage" in error_found:
        critic_errors = critic_errors + 1
        full_critic_feedback = all_critic_response
        
    if "Redundant" in error_found:
        critic_errors = critic_errors + 1
        full_critic_feedback = all_critic_response
        
    if "Incoherent" in error_found:
        critic_errors = critic_errors + 1
        full_critic_feedback = all_critic_response
            
    critic_score = 0 if (full_critic_feedback == 'No error detected.') else critic_errors
    
    return critic_score, full_critic_feedback

# The main Generate and Critique function.
def error_detection_with_critique(df_scarecrow, list_of_indices):
    
    # Defines three Scarecrow errors.
    
    error_labels = ['Grammar and Usage', 'Redundant', 'Incoherent']
    error_definitions = ["Grammar and Usage: Errors related to missing words, extra words, and incorrect or out of order words.", 
                         "Redundant: Errors related to redundant repetition of the exact phrase or redundant repetition of the same idea using different words.", 
                         "Incoherent: Errors related to either contradiction or confusing language within the text."]
    
    # Read existing CSV or if it does not exist, then define a new one with the correct columns.
    
    try:
        the_df = pd.read_csv('gen_and_crit_results4.csv')
        done_indices = list(the_df['ID'])
    except:
        col_names =  ['ID', 'Scarecrow ID', 'Original Prompt', 'Original Generated Text', 'Revised Generated Text', 'Original Errors per Sentence', 'Revisions per Sentence']
        the_df = pd.DataFrame(columns = col_names)
        done_indices = []
        
    # If we already have a CSV, then we initialize at the index of the last entry.
        
    if len(done_indices) > 0:
        i = int(done_indices[-1]) + 1
    else:
        i = 0
    
    # For each Scarecrow example...
    
    while i <= max(list_of_indices):
        
        # If we already have a CSV, then we take the index of the last entry.
        
        if len(done_indices) > 0:
            i = int(done_indices[-1]) + 1
        else:
            i = 0
        
        try:
            
            if i not in done_indices and str(i) not in done_indices:

                print('-----', i, '-----')
                scarecrow_id = df_scarecrow.iloc[i]['id']
                original_prompt_text = df_scarecrow.iloc[i]['prompt']
                original_gen_text = df_scarecrow.iloc[i]['generation']
                                
                example_sentences = df_scarecrow.iloc[i]["sentences"]
                
                # Sanity check to see to see that the sentence split was done correctly.
                
                if (len(example_sentences)) != len(nltk.sent_tokenize(original_gen_text)):
                    print('Error of Sanity Check!')
                    print('---')
                    print(example_sentences)
                    print('---')
                    print(nltk.sent_tokenize(gen_text))
                    print('---')
                
                # Initializing variables: Original Errors per Sentence (List), Current Errors per Sentence (List), Number of Revisions per Sentence (List), Revised Text (String).
                
                original_errors_sent_list = [-1] * len(example_sentences)
                current_errors_sent_list = [0] * len(example_sentences)
                revisions_sent_list = [0] * len(example_sentences)
                revised_gen_text = ''
                
                # For each sentence of the Scarecrow example...

                for j in range(len(example_sentences)):
                    
                    # While at least one detected error remains in the sentence (and the sentence is valid - longer than two characters)...
                    
                    critic_score = -1    
                    while critic_score != 0 and len(example_sentences[j]) >= 3:
                        
                        # 

                        curr_sentence = example_sentences[j]
                        critic_score, full_critic_feedback = critique_and_find_errors(curr_sentence)
                        
                        # Keep track of the detected errors for the original sentences
                        if original_errors_sent_list[j] == -1:
                            original_errors_sent_list[j] = critic_score
                        
                        if critic_score == 0:
                            # Keep the original sentence if no error was detected
                            current_errors_sent_list[j] = current_errors_sent_list[j] + 0
                            revised_gen_text = revised_gen_text + ' ' + curr_sentence
                        else:
                            # Keep track of the total detected errors per sentence and stop revising it after 7 consecutive failed attempts to get rid of scarecrow errors.
                            current_errors_sent_list[j] = current_errors_sent_list[j] + 1
                            if current_errors_sent_list[j] >= 7:
                                revised_sent_response = ''
                                revised_gen_text = revised_gen_text + ' ' + curr_sentence
                                break
                            
                        # Generator Prompt: 
                        # System Prompt: States the purpose of rewriting a sentence based on the critique to get rid of Scarecrow errors.
                        # User Prompt: Gives the Scarecrow examples to the Generator for training, and the current sentence to be revised and rewritten by the Generator.
                        # Note: It states "Impossible" if there is no way to rewrite the sentence (and in this case, it returns the original sentence), or it returns the revised and rewritten sentence.
                        revised_sent_response = ''
                        while len(revised_sent_response) == 0 and critic_score > 0:

                            revised_sent_response = ''

                            try:

                                response = openai.ChatCompletion.create(
                                    api_key=KEY,
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are Generator ChatGPT, a helpful assistant who rewrites a sentence based on the provided critique, in order to fix detected errors."}, 
                                        {"role": "user", "content": generator_examples_list + "\n\n" + "Rewrite the following sentence in order to correct the detected errors of the critique. State 'Impossible' if there is no way to rewrite it, otherwise rewrite it based on the critique to the best of your ability." + "\n\n" + "- Original Sentence: " + curr_sentence + "\n\n" + "- Critique: " + full_critic_feedback + "- Corrected Sentence: "}
                                    ]
                                )

                                revised_sent_response = response['choices'][0]['message']['content'].strip()
                                
                                if 'impossible' in revised_sent_response.lower():
                                    print('Impossible: ', revised_sent_response)
                                    revised_sent_response = curr_sentence
                                
                                print('-------------------------------------')
                                print('Critic Score: ', critic_score)
                                print('-------------------------------------')
                                print('Critic Feedback: ', full_critic_feedback)
                                print('-------------------------------------')
                                print('Original Sent: ', curr_sentence)
                                print('-------------------------------------')
                                print('Revised Sent: ', revised_sent_response)
                                print('-------------------------------------')

                                revisions_sent_list[j] = revisions_sent_list[j] + 1
                                example_sentences[j] = revised_sent_response

                            except Exception as e:

                                print('API Call Error (4): ', e)
                                time.sleep(1)
                                revised_sent_response  = ''
                                
                    revised_gen_text = revised_gen_text + ' ' + revised_sent_response
                    
                revised_gen_text = revised_gen_text.strip()
                
                print('-------------------------------------')
                print('Prompt: ', original_prompt_text)
                print('-------------------------------------')
                print('Original Text: ', original_gen_text)
                print('-------------------------------------')
                print('Revised Text: ', revised_gen_text)
                print('-------------------------------------')
                
            # Create a row for the final dataframe based on the final values returned by the Generate and Critique method, and append it to the final dataframe.

            dict_keys =  ['ID', 'Scarecrow ID', 'Original Prompt', 'Original Generated Text', 'Revised Generated Text', 'Original Errors per Sentence', 'Revisions per Sentence']
            dict_values = [str(i), scarecrow_id, original_prompt_text, original_gen_text, revised_gen_text, original_errors_sent_list, revisions_sent_list]
            new_entry_dict = dict(zip(dict_keys, dict_values))
            new_row = pd.Series(new_entry_dict)

            print(new_row)

            the_df = pd.concat([the_df, new_row.to_frame().T], ignore_index=True)

            time.sleep(1)
            
            # Save a CSV snapshot every time we write a row in the final dataframe.

            the_df.to_csv('gen_and_crit_results4.csv', index = False)

            done_indices = list(the_df['ID'])
        
        except Exception as e:
            
            print('Loop Error: ', e)
            time.sleep(1)
    
    time.sleep(5)
    
    # Save a CSV of the completed final dataframe.
    
    the_df.to_csv('gen_and_crit_results4_final.csv', index = False)
    
    return the_df

error_detection_with_critique(df_scarecrow, list_of_indices)

df_final = pd.read_csv('gen_and_crit_results4_final.csv')
df_final['Original Generated Text'] = df_final['Original Generated Text'].apply(lambda x: ' '.join(x.split()))
df_final['Revised Generated Text'] = df_final['Revised Generated Text'].apply(lambda x: ' '.join(x.split()))
df_final['Revised'] = df_final.apply(lambda x : 1 if x['Original Generated Text'] != x['Revised Generated Text'] else 0, axis=1)
print(df_final['Revised'].sum())
df_final
