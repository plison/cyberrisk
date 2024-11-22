from nervaluate import Evaluator
import sys
from pprint import pprint
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

def read_treebank(filepath):
    tokens = []
    with open(filepath, "r") as f:
        for line in f:
            if line == "\n":
                continue
            items = line.strip().split("\t")
            tokens.append(items[3])
    return tokens


def read_treebank_bin(filepath):
    tokens = []
    with open(filepath, "r") as f:
        for line in f:
            if line == "\n":
                continue
            items = line.strip().split("\t")
            if items[3].startswith("I") or items[3].startswith("B"):
                tokens.append(":)")
            else:
                tokens.append(items[3])
    return tokens

true = read_treebank(sys.argv[1])
pred = read_treebank(sys.argv[2])
labels = list(set([l for l in true]))
labels.sort()
print(labels)
#evaluator = Evaluator(true, pred, tags=["Asset", "Victim", "Method", "Attacker", "Motivation"], loader="list")

#results, results_by_tag, result_indices, result_indices_by_tag = evaluator.evaluate()
#pprint(results)
#()
f1_scores = f1_score(true, pred, average=None, labels=labels)
f1_scores_with_labels = {label:score for label,score in zip(labels, f1_scores)}
print(f1_scores_with_labels)
#print(f1_score(true, pred, average="binary", pos_label=":)"))
#
print("f1-weighted, precision, recall",precision_recall_fscore_support(true, pred, average="weighted"))
true_bin = read_treebank_bin(sys.argv[1])
pred_bin = read_treebank_bin(sys.argv[2])
print("f1-binary, precision, recall", precision_recall_fscore_support(true_bin, pred_bin, average="binary", pos_label=":)"))
