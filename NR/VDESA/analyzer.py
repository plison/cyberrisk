from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer
from skorch import NeuralNetClassifier 
from torch import nn
from skorch.callbacks import EpochScoring
from src.corpus import Corpus, read_data
from src.model import Model
from src.mlp import MLP

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--corpus_filepath", required=True)
    argparser.add_argument("--model_filepath", required=True)
    argparser.add_argument("--output_filepath", default="predicted.txt")
    argparser.add_argument("--mode", required=True,
                           choices=["train", "predict", "eval"])
    args = argparser.parse_args()
    if args.mode == "train":
        corpus = read_data(args.corpus_filepath)
        mlp = MLP(512, len(corpus.statuses))
        f1 = EpochScoring(scoring="f1_weighted", lower_is_better=False)
        model = Model(encoder=SentenceTransformer("distiluse-base-multilingual-cased-v2"),
                      classifier=NeuralNetClassifier(mlp, max_epochs=1000, lr=0.25,
                                                     criterion=nn.CrossEntropyLoss,
                                                     callbacks=[f1],
                                                     optimizer__weight_decay=0.001,
                                                     iterator_train__shuffle=True))
        model.train(corpus)
        model.save(args.model_filepath)

    if args.mode == "predict":
        corpus = read_data(args.corpus_filepath, predict=True)
        model = Model()
        model.load(args.model_filepath)
        predicted = model.predict(corpus)
        predicted.save(args.output_filepath)

    if args.mode == "eval":
        corpus = read_data(args.corpus_filepath)
        model = Model()
        model.load(args.model_filepath)
        model.eval(corpus)
