def viberti(word_list, tag_tag, tag_word):
    start_sentence = "<s>"
    end_sentence = "</s>"
    tag_list = list(tag_tag)
    tag_list.remove(start_sentence)
    num_tag = len(tag_list)
    num_words = len(word_list)
    viberti = np.zeros((num_tag, num_words))
    viberti_tags = np.zeros([num_tag, num_words-1])

    # print"Sanity check: Shape of viberti: " + str(viberti.shape)
    # print"Sanity check: Shape of viberti_tags: " + str(viberti.shape)
    # print"Num tags: " + str(num_tag)
    # for tag in tag_tag[start_sentence].keys():
        # print("Sanity check in tag tag : " + str(tag_tag[start_sentence][tag]))
    for word_index in range(len(word_list)):
        word = word_list[word_index]
        for tag_index in range(num_tag):
            tag = tag_list[tag_index]
            word_tag_prob = get_word_tag_prob(tag_word, tag, word)
            # First iteration is treated differently. Tag is start of sentence.
            if word_index == 0:
                tag_tag_prob = get_tag_tag_prob(tag_tag, start_sentence, tag)
                prob_value = 0
                if word_tag_prob != 0:
                    prob_value = math.log(word_tag_prob) + math.log(tag_tag_prob)
                # Update viberti with prob value
                viberti[tag_index][word_index] = prob_value

            # Second iteration onwards.
            if word_index != 0:
                # Set maximum prob value in current node to be negative infinity
                maximum_prob_value = float('-inf')

                # Go through each previous nodes to find new maximum prob value
                for prev_tag_index in range(num_tag):
                    prev_tag = tag_list[prev_tag_index]
                    prev_value = viberti[prev_tag_index][word_index-1]
                    tag_tag_prob = get_tag_tag_prob(tag_tag, prev_tag, tag)

                    # Calculate new probability value
                    prob_value = 0
                    if tag_tag_prob != 0 and prev_value != 0:
                        prob_value = prev_value + math.log(tag_tag_prob)

                    # Update maximum_prob_value
                    if prob_value != 0 and prob_value > maximum_prob_value:
                        viberti[tag_index][word_index] = prob_value
                        viberti_tags[tag_index][word_index-1] = prev_tag_index

                # Update viberti
                if word_tag_prob != 0 and viberti[tag_index][word_index] != 0:
                    viberti[tag_index][word_index] = viberti[tag_index][word_index] + math.log(word_tag_prob)
        break

    # Last iteration.
    maximum_prob_value = float('-inf')
    highest_prob_index = 0
    for tag_index in range(num_tag):
        tag = tag_list[tag_index]
        tag_tag_prob = get_tag_tag_prob(tag_tag, tag, end_sentence)
        prev_value = viberti[tag_index][num_words -1]

        # Calculate new prob from each node to end node.
        prob_value = 0
        if tag_tag_prob != 0 and prev_value != 0:
            prob_value = prev_value + math.log(tag_tag_prob)
        
        if prob_value != 0 and prob_value > maximum_prob_value:
            highest_prob_index = int(tag_index)
            maximum_prob_value = prob_value

    # Trace back optimal path
    index = num_words - 1
    tagged_sentence = []
    while index >= 0:
        # Get tag and update tagged_sentence
        highest_prob_tag = tag_list[highest_prob_index]
        current_word = word_list[index]
        tag_word = current_word + "/" + highest_prob_tag
        # print("Sanity check| Word: " + current_word + " Highest prob tag: " + highest_prob_tag + "Highest prob index: "+ str(highest_prob_index) + "Tag word: [" + tag_word + "]")
        tagged_sentence.append(tag_word)

        # Get next highest_prob_tag
        if index != 0:
            highest_prob_index = int(viberti_tags[highest_prob_index][index-1])
        index -= 1

    tagged_sentence.reverse()
    # print ' '.join(word for word in tagged_sentence)
    

    # print(np.matrix(viberti_tags))
    # print(np.matrix(viberti))
    return []