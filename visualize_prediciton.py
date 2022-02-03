from wsd_models.util import *
import sys
import pandas as pd
from nltk.corpus import wordnet as wn

if __name__ == '__main__':
    semeval07 = 'WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007'
    eval_data = load_data(semeval07, 'semeval2007')

    predictions_path = 'WSD_Evaluation_Framework/Evaluation_Datasets/inference/semeval2007_predictions.txt'
    preds = {}
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id, sense = line.split()
            preds[id] = sense
    
    with open('wsd-biencoder/semeval2007_pred.tsv', 'w', encoding='utf-8') as f:
        for sent in eval_data:
            for token in sent:
                word, lemma, pos, id, label = token
                if id in preds:
                    gloss = wn.lemma_from_key(preds[id]).synset().definition()
                else:
                    gloss = ''
                f.write('\t'.join([word, pos, gloss])  + '\n')
            f.write('\n')