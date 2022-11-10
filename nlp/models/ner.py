import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings

load_dotenv()

REDUCED_NER_DATASET = os.getenv('REDUCED_NER_DATASET')
NER_EPOCHS = int(os.getenv('NER_EPOCHS'))

COLUMNS = {0: 'text', 1: 'ner'}

def init_tagger():
    if(REDUCED_NER_DATASET == 'True'):
        CORPUS: Corpus = ColumnCorpus('datasets', COLUMNS, 
        train_file='tiny_ner_dataset_train.txt', 
        test_file='tiny_ner_dataset_test.txt', 
        dev_file='tiny_ner_dataset_dev.txt')
    else:
        CORPUS: Corpus = ColumnCorpus('datasets', COLUMNS, 
        train_file='ner_dataset_train.txt', 
        test_file='ner_dataset_test.txt', 
        dev_file='ner_dataset_dev.txt')

    TAG_TYPE = 'ner'
    TAG_DICTIONARY = CORPUS.make_label_dictionary(label_type=TAG_TYPE)
    EMBEDDING_TYPES = [
        # GloVe embeddings
        WordEmbeddings('glove'),
        # contextual string embeddings, forward
        FlairEmbeddings('news-forward'),
        # contextual string embeddings, backward
        FlairEmbeddings('news-backward')
        ]
    EMBEDDINGS = StackedEmbeddings(embeddings=EMBEDDING_TYPES)
    TAGGER = SequenceTagger(hidden_size=256,             
                embeddings=EMBEDDINGS, 
                tag_dictionary=TAG_DICTIONARY,
                tag_type=TAG_TYPE,
                use_crf=True,
                allow_unk_predictions=True)
    return CORPUS, TAGGER

def plot_training(loss_path):
    dataframe = pd.read_table(loss_path)

    plt.figure(figsize=(10,10))
    
    plt.plot(dataframe['EPOCH'], dataframe['TRAIN_LOSS'], label='Training loss')
    plt.plot(dataframe['EPOCH'], dataframe['DEV_LOSS'], label='Test loss')
    
    plt.legend(loc='upper right')
    plt.title('Training and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.show()

def train():
    corpus, tagger = init_tagger()
    model_trainer = ModelTrainer(tagger, corpus)
    model_trainer.train('resources/taggers/example-ner',
                       learning_rate=0.1,
                       mini_batch_size=32,
                       max_epochs=NER_EPOCHS)

if __name__ == '__main__':
    train()