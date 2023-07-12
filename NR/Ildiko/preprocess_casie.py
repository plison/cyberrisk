import os, codecs
import random
import json
import spacy
from spacy.tokens import DocBin
from collections import Counter
import argparse

def extract_text(fname):
    """ Extract texts from CASIE XML-like format with only texts 
    under the 'source/' directory.
    """
    with codecs.open(fname, 'r', 'utf8') as f:
        content = f.read()
        text_start = content.index('<text>')+6
        return content[text_start:-7] # drop closing </text> tag

def load_annotations(fname):
    " Load CASIE JSON data from the 'annotations/' directory. "
    with open(fname, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_casie(ann_dir, texts_dir):
    """ Collect entity information (span start, end and label) from the 
    CASIE dataset (https://github.com/Ebiquity/CASIE). Both event nuggets
    and arguments are considered entities and their 'type' attribute is used
    as entity label. 
    Returns a list with tuples of text and the list of entity information tuples.
    @ ann_dir: str; path to the CASIE JSON data with annotations;
    @ texts_dir: str; path to the CASIE text data in 'source/'. 
    """
    data = [] 
    for fname in os.listdir(ann_dir):
        if fname != '10002.json':
            text_info = {}
            ann = load_annotations(os.path.join(ann_dir,fname))
            text = ann['content']
            text_txt = extract_text(os.path.join(texts_dir, fname.replace('json', 'txt')))

            # Checking length mismatch between .txt and .json texts
            # if len(text) != len(text_txt):
            #     if len(text_txt)-len(text) not in [3,4]:
            #         print(fname, len(text), len(text_txt), len(text_txt)-len(text))

            entities = [] 
            for event_cluster in ann['cyberevent']['hopper']: # list of dict with related events ('same')  
                cluster_relation = event_cluster.get('relation', '')
                for event in event_cluster['events']:
                    event_type = 'event_' + event['type'] + event['subtype']
                    if event_type.endswith('Vulnerability'): 
                        event_type = 'event_' + event_type.split('-')[-1].replace('related', '')
                    event_realis = event['realis']
                    event_text = event['nugget']['text']
                    start = event['nugget']['startOffset']
                    end = event['nugget']['endOffset']
                    entities.append((start,end, event_type))
                    for arg in event.get('argument', []):
                        entities.append((arg['startOffset'], arg['endOffset'], arg['type']))
                    event_subtype = event['subtype']
            text_info['fname'] = fname
            text_info['text'] = text
            text_info['entities'] = entities
            data.append(text_info)
    print()
    return data

def create_docbin(data, subset):
    # Based on https://spacy.io/usage/training
    """ Creates Spacy docs with a data split and sets CASIE annotated argument and event nugget 
    labels to .ents attrib (input data to train a Spacy named entity recognizer). 
    Saves data to disk in a .spacy file. 
    @ data: list; output of the process_casie function
    @ subset: str; 'train', 'dev' or 'test'.
    """
    print('Creating {} subset...'.format(subset))
    db = DocBin()
    boundary_mismatch_cnt = 0
    ent_cnt = 0
    skipped_files = []
    all_labels = []
    for text_info in data:
        text = text_info['text']
        doc = nlp(text)
        ents = []
        labels = []
        for start, end, label in text_info['entities']:
            span = doc.char_span(start, end, label=label)
            if not span:
                boundary_mismatch_cnt += 1
            else:
                labels.append(label)
            ents.append(span)
        ent_cnt += len(ents)
        # Some ents are None (when character boundaries don't align with token boundaries).
        # We remove these to avoid Spacy throwing a TypeError.
        filtered_ents = list(filter(lambda item: item is not None, ents)) 
        if len(ents) != len(filtered_ents):
            print('\t{}: {} boundary mismatches'.format(text_info['fname'], len(ents) - len(filtered_ents)))
        # Filter doc.s with many boundary mismatches
        if abs(len(ents) - len(filtered_ents)) > 5: 
            skipped_files.append(text_info['fname'])
        try:
            if text_info['fname'] not in skipped_files:
                doc.ents = filtered_ents
                doc.user_data = {'doc_id':text_info['fname'].replace('.json', '')}
                db.add(doc)
                all_labels.extend(labels)
        except ValueError:
            print('\t{}: ValueError with token included in more than one span in entities'.format(text_info['fname']))
            skipped_files.append(text_info['fname'])
        
    #if boundary_mismatch_cnt:
    #    print('{} ent with boundary mismatch out of a total of {} ents.'.format(boundary_mismatch_cnt, ent_cnt))
    db.to_disk("./{}.spacy".format(subset))
    print('Saved {} files (skipped {})'.format(len(data)-len(skipped_files), len(skipped_files)))
    print()
    for lbl, cnt in sorted(Counter(all_labels).items()):
        print(cnt, lbl)
    return db

def create_docbin_spancat(data, subset, span_key):
    """ Creates Spacy docs with a data split and sets CASIE annotated argument and event nugget 
    labels to .spans attrib (input data usable to train a Spacy span categorizer). 
    Saves data to disk in a .spacy file. 
    @ data: list; output of the process_casie function
    @ subset: str; 'train', 'dev' or 'test';
    @ span_key: str; specified in the config file (chosen by user, usually 'sc') 
    """
    print('Creating {} subset...'.format(subset))
    db = DocBin()
    all_labels = []
    for text_info in data:
        text = text_info['text']
        doc = nlp(text)
        ents = []
        labels = []
        # Get annotated entities from the corpus and set them to the .ents attrib
        for start, end, label in text_info['entities']:
            # fix character offset outside doc (in 464.json)
            if end > len(text):
                print(text_info['fname'], len(text), end)
                end = len(text)
            span = doc.char_span(start, end, label=label)
            ents.append(span)
            labels.append(label)
        all_labels.extend(labels)
        filtered_ents = list(filter(lambda item: item is not None, ents))
        doc.spans[span_key] = filtered_ents
        db.add(doc)
    db.to_disk("./{}.spacy".format(subset))
    for lbl, cnt in sorted(Counter(all_labels).items()):
        print(cnt, lbl)
    return db

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-data_dir", type=str, help="Path to CASIE 'data/' directory", required=True)
    argparser.add_argument("-eval_size", type=int, help="The size of the development and the test set. (Same size is used.)", default=100)
    argparser.add_argument("-attr_type", type=str, help="The Spacy Doc attribute to which to add annotations 'ents' (to use data with an entity recognizer) or 'spans' (to use data with a span categorizer)", default='ents')
    argparser.add_argument("-span_key", type=str, help="The span key specified in the config file of the span categorizer to be used.", default='sc')
    args = argparser.parse_args()

    texts_dir = os.path.join(args.data_dir, 'source')
    ann_dir = os.path.join(args.data_dir, 'annotation')

    nlp = spacy.blank("en")
    random.seed(1234)
    data = preprocess_casie(ann_dir, texts_dir)
    random.shuffle(data)
    
    if args.attr_type == 'spans':
        train = create_docbin_spancat(data[args.eval_size*2:], 'train', args.span_key)
        dev = create_docbin_spancat(data[args.eval_size:args.eval_size*2], 'dev', args.span_key)
        test = create_docbin_spancat(data[:args.eval_size], 'test', args.span_key)
    else:
        train = create_docbin(data[args.eval_size*2:], 'train')
        dev = create_docbin(data[args.eval_size:args.eval_size*2], 'dev')
        test = create_docbin(data[:args.eval_size], 'test')

    print('Creating {} train, {} dev and {} test instances.'.format(len(train), len(dev), len(test)))
    

