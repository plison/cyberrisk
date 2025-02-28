import json, sys, random
from nltk.tokenize import word_tokenize
from math import ceil


status2severity = {"probably_exploited": 4,
                   "could_be_exploited": 3,
                   "unlikely_to_be_exploited": 2,
                   "probably_not_exploited": 1,
                   "unclear": 0,
                   "unk": -1} ## remove
#                   "exploited": 5}  ## remove 

def read_data(filepath):
    with open(filepath, "r") as f:
        corpus = json.load(f)
    return corpus


def convert(document):
    sentence = document["text"]
    tokens = word_tokenize(sentence)
    cve2exploit_status = {}
    cve_in_sentence = False
    for entity in document["entities"]:
        if entity["span_text"].startswith("CVE-"):

            status = entity.get("exploit_status","unk")
            cve = entity["span_text"].strip()
            if cve not in cve2exploit_status.keys():
                cve2exploit_status[cve] = status
            else:
                status_severity = status2severity[status]
                current_status = cve2exploit_status[cve]
                current_status_severity = status2severity[current_status]
                if status_severity > current_status_severity:
                    cve2exploit_status[cve] = status
                
                             
    tree = []
    for i, token in enumerate(tokens):
        label = cve2exploit_status.get(token, "not_cve")
        flag = "not_cve"
        if label != "not_cve":
            flag = "cve"
            if label != "unk":
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
    labelled = 0
    cves = 0
    with open(filepath, "w") as o:
        #for doc in labelled_corpus:
        for tree in labelled_corpus:
            sentences += 1
            for i, entry in enumerate(tree):
                tokens += 1
                if entry[3] != "not_cve":
                    cves += 1
                    if entry[3] != "unk":
                        labelled += 1
                o.write("\t".join(entry))
                o.write("\n")
            o.write("\n")
    print(filepath)
    print("\t #sentences: ", sentences)
    print("\t #tokens: ", tokens)
    print("\t #CVEs: ", cves)
    print("\t #labelled CVEs: ", labelled)



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
print(len(labelled_corpus))
random.shuffle(labelled_corpus)
train_ind = ceil(len(labelled_corpus) * 0.8) - 45
test_ind = ceil(len(labelled_corpus) * 0.1) + train_ind - 9
print(train_ind, test_ind)
labelled_corpus, unclear_instances = split_on_label(labelled_corpus, "unclear")
print("unclear", len(unclear_instances))
print("without", len(labelled_corpus))
test = [unclear_instances[0]]
train = [unclear_instances[1]]
labelled_corpus, prob_exp_instances = split_on_label(labelled_corpus, "probably_exploited")
print("probably_exploited", len(prob_exp_instances))
print("without", len(labelled_corpus))
test += prob_exp_instances[0:2]
train += prob_exp_instances[2:11]
dev = prob_exp_instances[11:]

labelled_corpus, prob_not_exp_instances = split_on_label(labelled_corpus, "probably_not_exploited")
print("probably_not_exploited", len(prob_not_exp_instances))
print("without", len(labelled_corpus))
test += prob_not_exp_instances[0:4]
train += prob_not_exp_instances[2:12]
dev += prob_not_exp_instances[12:]

labelled_corpus, could_instances = split_on_label(labelled_corpus, "could_be_exploited")
print("could_be_exploited", len(could_instances))
print("without", len(labelled_corpus))

test += could_instances[0:4]
train += could_instances[4:27]
dev += could_instances[27:]

print(len(train), len(test), len(dev))

train += labelled_corpus[:train_ind]
test += labelled_corpus[train_ind:test_ind]
dev += labelled_corpus[test_ind:]
print(len(train), len(test), len(dev))



save("casie-eela-train.conllu", train)
save("casie-eela-dev.conllu", dev)
save("casie-eela-test.conllu", test)
