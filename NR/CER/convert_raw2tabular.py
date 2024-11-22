import json, sys
from argparse import ArgumentParser

PUNC = '¬`!"£$%^&*()-_=+[]{}:;@#~,.<>/?|\\'
PUNC += "'"
EOS_PUNC = "!?."


def save(filepath, tokenised_sentences):
    with open(filepath, "w") as o:
        for sentence in tokenised_sentences:
            for i, token in enumerate(sentence):
                output = ["_"]*10
                output[0] = str(i+1)
                output[1] = token
                o.write("\t".join(output))
                o.write("\n")
            o.write("\n")
    print(filepath)
    print("\t #sentences: ", len(tokenised_sentences))

def read_data(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    return text

def convert(text):
    sentences = []
    tokens = []
    token = ""
    text_check = text + " "
    offset_start = 0
    for i, char in enumerate(text):
        if char == " ":
            if token != "":
                tokens.append(token)
            token = ""
            offset_start = i+1
        elif char in PUNC:
            if char == "'" and text_check[i+1] != " ":
                token+=char
                continue
            if token != "":
                tokens.append(token)
            tokens.append(char)
            token = ""
            offset_start = i+1
            if char in EOS_PUNC and text_check[i+1] == " ":
                sentences.append(tokens)
                tokens = []
                token = ""
                offset_start = i
        else:
            token+=char

    return sentences

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--infile", required=True)
    argparser.add_argument("--outfile", required=True)
    args = argparser.parse_args()
    text = read_data(args.infile)
    trees = convert(text)
    save(args.outfile, trees)


