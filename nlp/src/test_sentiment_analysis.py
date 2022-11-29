import unittest
import transformers
import sentiment_analysis

class Testing(unittest.TestCase):
    def test_init_pipeline(self):
        sentiment_pipeline = sentiment_analysis.init_pipeline()

        self.assertEqual(type(sentiment_pipeline), 
            transformers.pipelines.text_classification.TextClassificationPipeline, 
            f'Wrong pipeline, expected transformers.pipelines.text_classification.TextClassificationPipeline instead got {type(sentiment_pipeline)}.'
        )

    def test_sentiment_analysis_predict(self):
        output_list = sentiment_analysis.predict(['I like you', 'I love you', 'I hate you'])
        
        self.assertEqual(output_list, 
            ['POSITIVE', 'POSITIVE', 'NEGATIVE'],
            f'Incorrect prediction, expected [POSITIVE, POSITIVE, NEGATIVE] instead got {output_list}, check if the pipeline is correct.'
        )

if __name__ == '__main__':
    unittest.main()