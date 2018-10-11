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

def update_nested_dict(first_key, second_key, nested_dict, count_key):

    # First occurence of a first key
    if (first_key in nested_dict) == False:
        nested_dict[first_key] = {}
        nested_dict[first_key][count_key] = 0

    # First occurence of a second key
    if (second_key in nested_dict[first_key]) == False:
        nested_dict[first_key][second_key] = 1
        nested_dict[first_key][count_key] += 1
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
        for second_key in nested_dict[first_key].keys():
            second_key_count = nested_dict[first_key][second_key]
            smoothed_count_numerator = second_key_count + num_smooth
            smoothed_count_denominator = first_key_count + (type_count * num_smooth)
            nested_dict[first_key][second_key] = smoothed_count_numerator / float(smoothed_count_denominator) 
        nested_dict[first_key][count_key] = first_key_count
        nested_dict[first_key][unknown_key] = num_smooth / smoothed_count_denominator

    

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
    vocab_count = 0
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
            update_nested_dict(prev_tag, tag, tag_tag, count_key)
            # Set prev_tag back to start of sentence.
            prev_tag = start_sentence
            continue

        # Update dictionary.
        update_nested_dict(prev_tag, tag, tag_tag, count_key)
        update_nested_dict(tag, word, tag_word, count_key)
        vocab_count += 1
        
        # Before going to the next loop set prev_tag
        prev_tag = tag

    # 4. Create P(tag1|tag2) probabilities with counts

    # Count unique word_types
    word_type_count = 0
    for first_key in tag_word.keys():
        word_type_count += len(list(tag_word[first_key].keys()))
    # Count unique tag_types
    tag_type_count = len(list(tag_tag.keys()))
    print("Tag type count: " + str(tag_type_count))
    print("Word type count: " + str(word_type_count))

    # smoothing = NO_SMOOTHING
    smoothing = ADD_ONE
    # smoothing = KNESER_NEY

    if (smoothing == ADD_ONE):
        smoothing_count = 1
        add_one_smoothing(tag_word, count_key, unknown_key, word_type_count, smoothing_count)
        add_one_smoothing(tag_tag, count_key, unknown_key, tag_type_count, smoothing_count)
    elif(smoothing == KNESER_NEY):
        # Implement a back-off to use unigram prob
        unigram_key = "<UNI>"
        kneser_ney_tag_tag(tag_tag, count_key, unknown_key, tag_type_count, unigram_key)
        kneser_ney_tag_word(tag_word, count_key, unknown_key, word_type_count, unigram_key)
        print("Hello")
    else:
        transform_count_to_prob(tag_tag, count_key)
        transform_count_to_prob(tag_word, count_key)

    # 5. Concat tag_word, tag_tag into new dictionary
    tables = {}
    tables["tag_tag"] = tag_tag
    tables["tag_word"] = tag_word

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
