import json, sys
from collections import Counter
import numpy as np
import random
from math import ceil

PUNC = '¬`!"£$%^&*()-_=+[]{}:;@#~,.<>/?|\\'
PUNC += "'"
EOS_PUNC = "!?."
PRONOUNS = ["we", "I", "you", "they", "he", "she",
            "us", "me", "them", "him", "her"]


def save(filepath, labelled_corpus):
    sentences = 0
    tokens = 0
    labelled = 0
    phrases = 0
    with open(filepath, "w") as o:
        for doc in labelled_corpus:
            for sentence in doc:
                sentences += 1
                for i, token in enumerate(sentence):
                    tokens += 1
                    label = token[1]
                    if label != "O":
                        labelled += 1
                    if label.startswith("B"):
                        phrases += 1
                    output = ["_"]*10
                    output[0] = str(i+1)
                    output[1] = token[0]
                    output[3] = token[1]
                    output[-1] = str(token[-1])
                    o.write("\t".join(output))
                    o.write("\n")
                o.write("\n")
    print(filepath)
    print("\t #docs: ", len(labelled_corpus))
    print("\t #sentences: ", sentences)
    print("\t #tokens: ", tokens)
    print("\t #labelled tokens: ", labelled)
    print("\t #entities: ", phrases)

def read_data(filepath):
    with open(filepath, "r") as f:
        corpus = json.load(f)
    return corpus

def convert(doc):
    sentences = []
    tokens = []
    token = ""
    text = doc["text"] + " "
    offset_start = 0
    for i, char in enumerate(doc["text"]):
        if char == " ":
            if token != "":
                tokens.append((token, (offset_start, i)))
            token = ""
            offset_start = i+1
        elif char in PUNC:
            if char == "'" and text[i+1] != " ":
                token+=char
                continue
            if token != "":
                tokens.append((token, (offset_start, i)))
            tokens.append((char, (i, i+1)))
            token = ""
            offset_start = i+1
            if char in EOS_PUNC and text[i+1] == " ":
                sentences.append(tokens)
                tokens = []
                token = ""
                offset_start = i
        else:
            token+=char




    for sentence in sentences:
        for token in sentence:
            if token[0] != text[token[1][0]:token[1][1]]:
                print("<" + token[0] + ">",
                      "<" + text[token[1][0]:token[1][1]] + ">")
                exit("misaligned tokenization!")

    offsets2label = {}    
    for entry in doc["entities"]:
        so = entry["start_offset"]
        eo = entry["end_offset"]
        for n in range(so, eo):
            offsets2label[n] = entry["label"]

    labelled_sentences = []
    labelled_tokens = []
    last_label = ""
    for sentence in sentences:
        entity_seen = False
        if len(sentence) == 0:
            continue
        begin = False
        for token in sentence:
            if len(token) == 0:
                continue
            labels = []
            for n in range(token[1][0], token[1][1]):
                l = offsets2label.get(n, "O")
                labels.append(l)
            x = Counter(labels)
            label = x.most_common(1)[0][0]
            if token[0].strip().lower() in PRONOUNS:
                if label != "O":
                    label = "O"
            if label != "O":
                if begin:
                    if label == last_label.split("-")[-1]:
                        label = "I-" + label
                    else:
                        label = "B-" + label
                        begin = True
                else:
                    label = "B-" + label
                    begin = True
            else:
                begin = False
            if label != "O":
                entity_seen = True
            last_label = label
            
            labelled_tokens.append((token[0], label, token[1]))
        if entity_seen and len(labelled_tokens) > 0:
            labelled_sentences.append(labelled_tokens)
        labelled_tokens = []
    return labelled_sentences
    
corpus = read_data(sys.argv[1])

labelled_corpus = [convert(doc) for doc in corpus if convert(doc) is not None]
print(len(labelled_corpus))
random.shuffle(labelled_corpus)
train_ind = ceil(len(labelled_corpus) * 0.8)
test_ind = ceil(len(labelled_corpus) * 0.1) + train_ind
print(train_ind, test_ind)

train = labelled_corpus[:train_ind]
test = labelled_corpus[train_ind:test_ind]
dev = labelled_corpus[test_ind:]


save("casie-cer-train.conllu", train)
save("casie-cer-dev.conllu", dev)
save("casie-cer-test.conllu", test)


