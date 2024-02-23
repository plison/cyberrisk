from abc import ABC, abstractmethod
from src.utils import dynamic_import
import pickle

class EPSS(ABC):
    def __init__(self, config):
        self.config = config
        need_LM = False
        self.lm = None
        lm_dependent_modules = ["ClassifierCWE", "ClassifierCVSS", "DescriptionEmbedder"]
        selected_modules = set([k for k,v in config.modules.items() if v])
        if len(selected_modules.intersection(set(lm_dependent_modules))) > 0:
            from sentence_transformers import SentenceTransformer
            self.lm = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        self.modules = []
        for m in selected_modules:
            Module = dynamic_import("src.modules", m)
            self.modules.append(Module(self.lm))
        self.nvd_features = config.nvd_features

    @abstractmethod
    def train(self, data):
        ## actually train model
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def evaluate(self, data):
        pass

    @abstractmethod
    def cast_target(self, data):
        ## needs to be different for each model type
        pass

    def save(self,  filepath):
        with open(filepath, 'wb') as o:
            o.write(pickle.dumps(self.__dict__))
            
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.__dict__ = pickle.loads(f.read())

