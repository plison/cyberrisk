# -*- coding: utf-8 -*-

from src import Tagger, Model
from src.utils import Corpus
from src.utils import reintroduce
from src.utils.data import TextDataset, batchify
import time
import torch
import tempfile
import json

class Predict(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--batch-size', default=5000, type=int,
                               help='batch size')
        subparser.add_argument('--fdata', default='data/ptb/test.conllx',
                               help='path to dataset')
        subparser.add_argument('--fpred', default='pred.conllx',
                               help='path to predicted result')

        return subparser

    def apply_tags(self, corpus, tags):
        #CVEs = []
        CVEs2tags = []
        for i, sentence in enumerate(corpus.sentences):
            cves = {}
            for j, element in enumerate(sentence.FLAG):
                if element == "cve":
                    cves[sentence.FORM[j]] = []
            CVEs2tags.append(cves)
        for i, (C, T) in  enumerate(zip(CVEs2tags, tags)):
            for cve, tag in zip(C.keys(), T):
                CVEs2tags[i][cve].append(tag)
        return self.select_tags(CVEs2tags)
        
    def select_tags(self, CVEs2tags):
        status2severity = {"probably_exploited": 4,
                           "could_be_exploited": 3,
                           "unlikely_to_be_exploited": 2,
                           "probably_not_exploited": 1}
        CVE2tag = []
        for CVEs in CVEs2tags:
            cves = {}
            for cve, tags in CVEs.items():
                severity = -1
                if len(tags) > 1:
                    if "unclear" in tags:
                        tags.remove("unclear")
                    if len(tags) == 0:
                        selected_tag = "unclear"
                        continue
                for tag in tags:
                    tag_severity = status2severity[tag]
                    if tag_severity > severity:
                        severity = tag_severity
                        selected_tag = tag
                cves[cve] = selected_tag
            CVE2tag.append(cves)
        return CVE2tag

    def extract_text(self, filepath):
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


    def save(self, corpus_filepath, CVE2tag, outpath):
        docs = self.extract_text(corpus_filepath)
        output = []
        for doc, cve2tag in zip(docs, CVE2tag):
            entry = {#"text": doc,
                     "tagged_CVEs": cve2tag}
            output.append(entry)
        with open(outpath, "w") as o:
            json.dump(output, o, ensure_ascii=False, indent=3)
        
    def __call__(self, config):
        time_t0 = time.time()
        print("Load the model")
        vocab = torch.load(config.vocab)
        parser = Tagger.load(config.model)
        model = Model(vocab, parser)
        print("Load the dataset")
        start_time = time.time()
        corpus = Corpus.load(config.fdata)
        print(":(", len(corpus))
        dataset = TextDataset(vocab.numericalize(corpus, training=False))
        # set the data loader
        
        loader = batchify(dataset, config.batch_size)
        print("Make predictions on the dataset")
        start_parse_time = time.time()
        tags = model.predict(loader)
        CVE2tag = self.apply_tags(corpus, tags)
        self.save(config.fdata, CVE2tag, config.fpred)
        exit()
        parse_time = time.time() - start_parse_time        
        print(f"Save the predicted result to {config.fpred}")
        tmp = tempfile.NamedTemporaryFile(delete=False)
        corpus.save(tmp.name)
        ### for reintroducing 1-2 and 1.5 entries for evaluation
        reintroduce(config.fdata, tmp.name, config.fpred)
        time_total = time.time() - time_t0
        
        print("TIMES", str(time_total) + "," + str(parse_time))
        #speed = float(len(corpus.heads)) /timeTotal
        #print("Speed: " + str(speed) + " sent/s")
