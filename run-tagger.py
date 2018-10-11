# python3.5 run-tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np
import json


def get_tag_tag_prob(dictionary, first_tag, second_tag, count_key, tag_unigram_prob):
    unknown_key = "<UNK>"
    if (second_tag in dictionary[first_tag]) == True:
        return dictionary[first_tag][second_tag]
    
    return dictionary[first_tag][unknown_key]
    # if second_tag in tag_unigram_prob:
    #     return tag_unigram_prob[second_tag]
    # return 0

def get_tag_word_prob(dictionary, tag, word, count_key):
    unknown_key = "<UNK>"
    if (word in dictionary[tag]) == True:
        return dictionary[tag][word]

    return dictionary[tag][unknown_key]


def viberti(word_list, tag_tag, tag_word, tag_unigram_prob):
    count_key = "count"
    start_sentence = "<s>"
    end_sentence = "</s>"
    tag_list = list(tag_tag)
    tag_list.remove(start_sentence)
    num_tags = len(tag_list)
    num_words = len(word_list)
    viterbi = np.zeros((num_tags, num_words))
    back_pointer = np.zeros((num_tags, num_words))

    # Initialisation
    for i in range (num_tags):
        curr_tag = tag_list[i]
        curr_word = word_list[0]
        tag_tag_prob = get_tag_tag_prob(tag_tag, start_sentence, curr_tag, count_key, tag_unigram_prob) 
        tag_word_prob = get_tag_word_prob(tag_word, curr_tag, curr_word, count_key)
        node_value = tag_tag_prob * tag_word_prob
        viterbi[i, 0] = node_value
        back_pointer[i, 0] = 0

    for word_index in range (1, num_words):
        curr_word = word_list[word_index]
        # viterbi[34, word_index-1] = -math.inf
        for tag_index in range(num_tags):
            curr_tag = tag_list[tag_index]
            # Calculate max to put in viterbi
            max_prob_value = -math.inf
            for node_index in range(num_tags):
                # Get tag-tag prob
                prev_tag = tag_list[node_index]
                tag_tag_prob = get_tag_tag_prob(tag_tag, prev_tag, curr_tag, count_key, tag_unigram_prob)
                # Get node value of column before (which is a log value)
                prev_node_value = viterbi[node_index, word_index-1]
                new_prob_value = prev_node_value * tag_tag_prob
                if (new_prob_value > max_prob_value):
                    max_prob_value = new_prob_value
                    back_pointer[tag_index, word_index] = node_index

            tag_word_prob = get_tag_word_prob(tag_word, curr_tag, curr_word, count_key)
            node_value = tag_word_prob * max_prob_value
            # print("Current word: " + curr_word + " | current tag: " + curr_tag + " | Tag tag prob: " + str(tag_tag_prob) + " | prev node val: " + str(prev_node_value))
            viterbi[tag_index, word_index] = node_value
            # print("Back pointer here: " + str(back_pointer[tag_index][word_index]))
            # print("Tag here: " + curr_tag + "  | New node_value" + str(node_value))
            
    # Final iteration node -> (End of sentence) </s>
    back_node_index = 0
    for node_index in range(num_tags):
        max_prob_value = -math.inf
        prev_tag = tag_list[node_index]
        prev_node_value = viterbi[node_index, num_words-1]
        tag_tag_prob = get_tag_tag_prob(tag_tag, prev_tag, end_sentence, count_key, tag_unigram_prob)
        new_prob_value = prev_node_value * tag_tag_prob

        if (new_prob_value > max_prob_value):
            back_node_index = node_index
            max_prob_value = new_prob_value

    # Trace back
    tagged_sentence = [ ]
    for i in range(1, num_words):
        back_node_index = int(back_node_index)
        index = num_words - i
        curr_word = word_list[index]
        curr_tag = tag_list[back_node_index]
        tagged_word = curr_word + "/" + curr_tag
        tagged_sentence.append(tagged_word)
        back_node_index = back_pointer[back_node_index, index]

    # For the first word and tag
    first_tag = tag_list[int(back_node_index)]
    first_word = word_list[0]
    tagged_word = first_word + "/" + first_tag
    tagged_sentence.append(tagged_word)

    tagged_sentence.reverse()
    return tagged_sentence


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.

    # 1. Read model file into a json
    dictionary = open(model_file, "r")
    dictionary_text = dictionary.read()
    json_acceptable_string = dictionary_text.replace("'", "\"")
    tables = json.loads(str(dictionary_text))
    tag_tag = tables["tag_tag"]
    tag_word = tables["tag_word"]


    # Get unigram probabilities
    tag_unigram_prob = {}
    word_unigram_prob = {}

    total_vocab_count = 0
    for primary_tag in tag_tag.keys():
        total_vocab_count += tag_tag[primary_tag]["count"]

    for primary_tag in tag_tag.keys():
        tag_count = tag_tag[primary_tag]["count"]
        unigram_prob = tag_count / total_vocab_count
        tag_unigram_prob[primary_tag] = unigram_prob

    # # To check for total prob.
    # for first_key in tag_word.keys():
    #     prob_count = 0
    #     for second_key in tag_word[first_key].keys():
    #         prob_count += tag_word[first_key][second_key]
    #     prob_count -= tag_word[first_key]["count"]
    #     print("Tag: " + first_key + " | total prob: " + str(prob_count))
    #     break
        
    # 2. Read test file line by line
    with open(test_file, "r") as ins:
        sentences = []
        for line in ins:
            sentences.append(line)

    # 3. Go through each sentence and tag them with the viberti algorithm. Write to out_file
    with open(out_file, "w") as data:
        for sentence in sentences:
            word_list = sentence.split()
            tagged_list = viberti(word_list, tag_tag, tag_word, tag_unigram_prob)
            # Write to file after tagging a sentence
            for tagged_word in tagged_list:
                data.write(tagged_word + " ")
            data.write("\n")
            # break
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
