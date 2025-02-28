import json, sys

def read_data(filepath):
    with open(filepath, "r") as f:
        corpus = json.load(f)
    return corpus

def read_docs(filepath):
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

def add_text(predictions, docs):
    for pred, doc in zip(predictions, docs):
        pred["text"] = doc
    return predictions

def save(output, outpath):
    with open(outpath, "w") as o:
        json.dump(output, o, ensure_ascii=False, indent=3)


predictions = read_data(sys.argv[1])
docs = read_docs(sys.argv[2])
predictions = add_text(predictions, docs)
save(predictions, sys.argv[3])

