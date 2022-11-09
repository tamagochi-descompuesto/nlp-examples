from models.ner import train, plot_training
from models.sentiment_analysis import predict
from models.api_translation import global_translate


def load_dataset(path):
    file = open(path, 'r', encoding='utf-8')
    return file.readlines()

if __name__ == '__main__':
    movie_dataset = load_dataset('datasets/tiny_movie_reviews_dataset.txt')
    predict(movie_dataset)

    train()
    plot_training('resources/taggers/example-ner/loss.tsv')

    translation_dataset_en = load_dataset('datasets/europarl-v7.es-en.en')
    translation_dataset_es = load_dataset('datasets/europarl-v7.es-en.es')
    libre_score, deep_score = global_translate(translation_dataset_en, translation_dataset_es)
    print(f'Libre Translate score: {libre_score}\nDeep Translate score: {deep_score}')