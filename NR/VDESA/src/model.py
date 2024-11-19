from sklearn.metrics import f1_score
import pickle
import numpy as np
from .corpus import Corpus

class Model(object):
    def __init__(self, encoder=None, classifier=None):
        self.encoder = encoder
        self.classifier = classifier

    def train(self, corpus: Corpus) -> None:
        embedded_descriptions = np.array([self.encoder.encode(d) for d in corpus.descriptions])
        self.status2id = self.get_status2id_dict(corpus)
        self.id2status = self.get_id2status_dict(corpus)
        target = np.array(self.numericalise_target(corpus.status_labels))
        self.classifier.fit(embedded_descriptions, target)
        

    def predict(self, corpus: Corpus) -> Corpus:
        embedded_descriptions = np.array([self.encoder.encode(d) for d in corpus.descriptions])
        predictions = [self.id2status[p] for p in self.classifier.predict(embedded_descriptions)]
        corpus.add_predictions(predictions)
        return corpus
        
    def eval(self, corpus: Corpus) -> None:
        embedded_descriptions = np.array([self.encoder.encode(d) for d in corpus.descriptions])
        target = np.array(self.numericalise_target(corpus.status_labels))
        predictions = self.classifier.predict(embedded_descriptions)
        print(f1_score(target, predictions, average="weighted"))
        print(f1_score(target, predictions, average=None))
        print(self.id2status)
        print(corpus.class_distribution)

    def get_status2id_dict(self, corpus: Corpus) -> dict:
        return {status: i for i, status in enumerate(corpus.statuses)}

    def get_id2status_dict(self, corpus: Corpus) -> dict:
        return {i: status for i, status in enumerate(corpus.statuses)}

    def numericalise_target(self, target: list) -> np.array:
        return np.array([self.status2id[t] for t in target])
    
    def save(self,  filepath: str) -> None:
        with open(filepath, 'wb') as o:
            o.write(pickle.dumps(self.__dict__))
            
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            self.__dict__ = pickle.loads(f.read())


