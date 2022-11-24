import string
from transformers import pipeline

def init_pipeline():
	SENTIMENT_PIPELINE = pipeline('sentiment-analysis')
	return SENTIMENT_PIPELINE

def predict(dataset):
	sentiment_pipeline = init_pipeline()
	# Removing punctuation since the pipeline crashes if the tensor contains text with punctuation, 
	# may search for a specific pipeline that allows punctuation 
	preprocessed_dataset = [review.translate(str.maketrans('', '', string.punctuation)) for review in dataset]

	outputs = sentiment_pipeline(preprocessed_dataset)

	for output in outputs:
		print(output['label'])
		
if __name__ == "__main__":
	predict(['I love you', 'I like you', 'I hate you'])