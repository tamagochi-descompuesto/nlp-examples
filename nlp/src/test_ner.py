import ner
import flair
import unittest

class Testing(unittest.TestCase):
    def test_init_tagger(self):
        corpus, tagger = ner.init_tagger()

        self.assertEqual(type(corpus),
            flair.datasets.sequence_labeling.ColumnCorpus,
            f'Expected flair.datasets.sequence_labeling.ColumnCorpus, instead got {type(corpus)} check the initialization of your tagger.'
        )

        self.assertEqual(type(tagger),
            flair.models.sequence_tagger_model.SequenceTagger,
            f'Expected flair.models.sequence_tagger_model.SequenceTagger, instead got {type(tagger)} check the initialization of your tagger.'
        )

    def test_ner_training(self):
        ner_dict = ner.train()

        self.assertEqual(type(ner_dict),
            dict,
            f'Expected type dict, instead got {type(ner_dict)}.'
        )

if __name__ == '__main__':
    unittest.main()