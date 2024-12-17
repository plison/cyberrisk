import json, sys, random
from nltk.tokenize import word_tokenize
from math import ceil




def read_data(filepath):
    docs = []
    doc = ""
    with open(filepath, "r") as f:
        for line in f:
            if line == "\n":
                docs.append(doc)
                doc = ""
            else:
                doc+=line
    return docs


def convert(document):
    tokens = word_tokenize(document)
    cve2exploit_status = {}
    cve_in_sentence = False
    tree = []
    for i, token in enumerate(tokens):
        flag = "not_cve"
        label = "not_cve"
        if token.startswith("CVE-"):
            flag = "cve"
            label = "unclear"
            cve_in_sentence = True
        columns = [str(i+1), token.strip(), "_",
                           label, flag, "_", "_",
                           "_", "_", "_"]
        tree.append(columns)
    if cve_in_sentence:
        return tree
    else:
        return None


def save(filepath, labelled_corpus):
    sentences = 0
    tokens = 0
    cves = 0
    with open(filepath, "w") as o:
        for tree in labelled_corpus:
            sentences += 1
            for i, entry in enumerate(tree):
                tokens += 1
                if entry[3] != "not_cve":
                    cves += 1
                o.write("\t".join(entry))
                o.write("\n")
            o.write("\n")
    print(filepath)
    print("\t #sentences: ", sentences)
    print("\t #tokens: ", tokens)
    print("\t #CVEs: ", cves)



def split_on_label(corpus, label):
    wit, without = [], []
    seen_unclear = False
    for tree in corpus:
        for token in tree:
            if token[3] == label:
                seen_unclear = True
                break
        if seen_unclear:
            wit.append(tree)
        else:
            without.append(tree)
        seen_unclear = False
    return without, wit

corpus = read_data(sys.argv[1])
labelled_corpus = [convert(doc) for doc in corpus if convert(doc) is not None]
save(sys.argv[2], labelled_corpus)
