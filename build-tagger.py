# python3.5 build-tagger.py <train_file_absolute_path> <model_file_absolute_path>
import os
import math
import sys
import datetime
import numpy as np
import json


ADD_ONE = "ADD_ONE"
KNESER_NEY = "KNESER_NEY"
NO_SMOOTHING = "NO_SMOOTHING"

def is_invalid_word(word):
    if word[0].isalpha() == False:
        return True

    return False

def update_nested_dict(first_key, second_key, nested_dict, count_key, unigram_prob):

    # First occurence of a first key
    if (first_key in nested_dict) == False:
        nested_dict[first_key] = {}
        nested_dict[first_key][count_key] = 0

    # First occurence of a second key
    if (second_key in nested_dict[first_key]) == False:
        nested_dict[first_key][second_key] = 1
        nested_dict[first_key][count_key] += 1
        # Update unigram type count
        if (second_key in unigram_prob) == False:
            unigram_prob[second_key] = 1
        else:
            unigram_prob[second_key] += 1
        return

    # Increase count of first_key, second_key occurence
    nested_dict[first_key][second_key] += 1
    nested_dict[first_key][count_key] += 1


    return

# Using Add-one
def add_one_smoothing(nested_dict, count_key, unknown_key, type_count, num_smooth):
    # Add one to the probability of all unknowns
    for first_key in nested_dict.keys():
        first_key_count = nested_dict[first_key][count_key]
        smoothed_count_denominator = first_key_count + (type_count * num_smooth)
        for second_key in nested_dict[first_key].keys():
            second_key_count = nested_dict[first_key][second_key]
            smoothed_count_numerator = second_key_count + num_smooth
            nested_dict[first_key][second_key] = smoothed_count_numerator / float(smoothed_count_denominator) 
        nested_dict[first_key][count_key] = first_key_count
        nested_dict[first_key][unknown_key] = num_smooth / smoothed_count_denominator

    
def kneser_smoothing(nested_dict, count_key, kneser_amount):
    for first_key in nested_dict.keys():
        first_key_count = nested_dict[first_key][count_key] # C(W0)
        for second_key in nested_dict[first_key].keys():
            second_key_count = nested_dict[first_key][second_key] - kneser_amount  # number of C(W0W1) - kneser amount
            nested_dict[first_key][second_key] = second_key_count / float(first_key_count)
        nested_dict[first_key][count_key] = first_key_count

def transform_count_to_prob(nested_dict, count_key):
    for first_key in nested_dict.keys():
        first_key_count = nested_dict[first_key][count_key]
        for second_key in nested_dict[first_key].keys():
            second_key_count = nested_dict[first_key][second_key]
            nested_dict[first_key][second_key] = second_key_count / float(first_key_count)
        nested_dict[first_key][count_key] = first_key_count

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.

    # 1. Read text file and split by space.
    train_text_file = open(train_file, "r")
    train_text = train_text_file.read()
    train_text_list = train_text.split()


    # 2. Initialize two dictionaries.
    # tag_word: Keep count of tag:word pair
    # tag_tag: Keep count of tag:tag pair
    prev_tag = "<s>"
    start_sentence = "<s>"
    end_sentence = "</s>"
    full_stop = "."
    tag_word = { }
    tag_tag = { }
    tag_unigram_prob = { }
    word_unigram_prob = { }
    vocab_count = len(train_text_list)
    count_key = "count"
    unknown_key = "<UNK>"

    # 3. Go through each "word/tag" element
    for train_text in train_text_list:
        train_text_parsed = train_text.split("/")
        parsed_length = len(train_text_parsed)

        tag = train_text_parsed[parsed_length - 1]
        word = train_text_parsed[0]

        # Seen a full stop. Update tag to be end of sentence.
        if word == full_stop:
            tag = end_sentence
            update_nested_dict(prev_tag, tag, tag_tag, count_key, tag_unigram_prob)
            # Set prev_tag back to start of sentence.
            prev_tag = start_sentence
            continue

        # Update dictionary.
        update_nested_dict(prev_tag, tag, tag_tag, count_key, tag_unigram_prob)
        update_nested_dict(tag, word, tag_word, count_key, word_unigram_prob)
        
        # Before going to the next loop set prev_tag
        prev_tag = tag

    # 4a. Create P(tag1|tag2) probabilities with counts

    # smoothing = NO_SMOOTHING
    smoothing = ADD_ONE
    # smoothing = KNESER_NEY
    kneser_amount = 0.1 # Must be [0-1]

    if (smoothing == ADD_ONE):
        # Count unique word_types
        word_type_count = 0
        for first_key in tag_word.keys():
            word_type_count += len(list(tag_word[first_key].keys())) - 1
        # Count unique tag_types
        tag_type_count = len(list(tag_tag.keys())) - 1
        # Perform smoothing
        smoothing_count = 1
        add_one_smoothing(tag_word, count_key, unknown_key, word_type_count, smoothing_count)
        add_one_smoothing(tag_tag, count_key, unknown_key, tag_type_count, smoothing_count)
    elif(smoothing == KNESER_NEY):
        # Implement a back-off to use unigram prob
        kneser_smoothing(tag_tag, count_key, kneser_amount)
        kneser_smoothing(tag_word, count_key, kneser_amount)
    else:
        transform_count_to_prob(tag_tag, count_key)
        transform_count_to_prob(tag_word, count_key)

    # 4b. Calculate unigram probability for Kneser
    total_num_unique_tag_word_pair = 0
    total_num_unique_tag_tag_pair = 0
    tag_word_lamda = {}
    tag_tag_lamda = {}

    for tag in tag_word.keys():
        num_unique_tag_word_pair = len(list(tag_word[tag].keys()))
        num_word = tag_word[tag][count_key]
        lamda = (kneser_amount * num_unique_tag_word_pair) / float(num_word)
        tag_word_lamda[tag] = lamda
        total_num_unique_tag_word_pair += num_unique_tag_word_pair
    
    for tag in tag_tag.keys():
        num_unique_tag_tag_pair = len(list(tag_tag[tag].keys()))
        num_tag = tag_tag[tag][count_key]
        lamda = (kneser_amount * num_unique_tag_tag_pair) / float(num_word)
        tag_tag_lamda[tag] = lamda
        total_num_unique_tag_tag_pair += num_unique_tag_tag_pair

    for index in tag_unigram_prob.keys():
        count = tag_unigram_prob[index]
        probability = count / float(total_num_unique_tag_tag_pair)
        tag_unigram_prob[index] = probability
    
    for index in word_unigram_prob.keys():
        count = word_unigram_prob[index] 
        probability = count / float(total_num_unique_tag_word_pair)
        word_unigram_prob[index] = probability

    # 5. Concat tag_word, tag_tag into new dictionary
    tables = {}
    tables["tag_tag"] = tag_tag
    tables["tag_word"] = tag_word
    tables["tag_unigram_prob"] = tag_unigram_prob
    tables["tag_tag_lamda"] = tag_tag_lamda
    tables["word_unigram_prob"] = word_unigram_prob
    tables["tag_word_lamda"] = tag_word_lamda

    # 6. Write dictionary to file
    with open(model_file, "w") as data:
        json.dump(tables, data)

    print("\nFinished...")

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
