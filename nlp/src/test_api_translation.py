import unittest
import api_translation

class Testing(unittest.TestCase):
    def test_calc_bleu(self):
        bleu_scores = api_translation.calc_bleu(['Hola amigo'], ['Hola amigo'])

        self.assertAlmostEqual(int(bleu_scores[0]),
            0,
            f'Bad translation, expected [0, 0] and instead got {bleu_scores}'
        )
    
    def test_libre_translation(self): 
        en_es_translation = api_translation.libre_translate(['Hello friend'])

        self.assertEqual(
            en_es_translation,
            ['Hola amigo'],
            f'Bad translation, expected ["Hola amigo"], instead got {en_es_translation}'
        )

    def test_deep_translation(self): 
        en_es_translation = api_translation.deep_translate(['Hello friend'])

        self.assertEqual(
            en_es_translation,
            ['Hola amigo'],
            f'Bad translation, expected ["Hola amigo"], instead got {en_es_translation}'
        )

    def test_global_translate(self):
        libre_score, deep_score = api_translation.global_translate(['Hello friend'], ['Hola amigo'])

        self.assertAlmostEqual(int(libre_score),
            0,
            f'Translation is not working as expected, got {libre_score} instead of a number near zero'
        )

        self.assertAlmostEqual(int(deep_score),
            0,
            f'Translation is not working as expected, got {deep_score} instead of a number near zero'
        )

if __name__ == '__main__':
    unittest.main()