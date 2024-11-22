from argparse import ArgumentParser

def read_tabular_data(filepath):
    tree, trees = [], []
    with open(filepath, "r") as f:
        for line in f:
            if line == "\n":
                trees.append(tree)
                tree = []
                continue
            items = line.split("\t")
            tree.append(items)
    return trees

def extract_entities(trees):
    entities = []
    entity = []
    for tree in trees:
        for token in tree:
            if token[3].startswith("O"):
                if len(token) > 0:
                    entities.append(" ".join(entity))
                    entity = []
            elif token[3].startswith("B"):
                entities.append(" ".join(entity))
                entity.append(token[1])
            elif token[3].startswith("I"):
                entity.append(token[1])
    return set(entities)

def save(entities, filepath):
    with open(filepath, "w") as o:
        o.write(str(entities))
        
if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--labelled_data", required=True)
    argparser.add_argument("--outfile", required=True)
    args = argparser.parse_args()
    trees = read_tabular_data(args.labelled_data)
    entities = extract_entities(trees)
    save(entities, args.outfile)
