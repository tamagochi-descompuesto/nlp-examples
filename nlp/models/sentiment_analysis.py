import string
from transformers import pipeline

def init_pipeline():
	SENTIMENT_PIPELINE = pipeline('sentiment-analysis')
	return SENTIMENT_PIPELINE

def predict(dataset):
	sentiment_pipeline = init_pipeline()
	# Can remove punct like this: https://datagy.io/python-remove-punctuation-from-string/#:~:text=One%20of%20the%20easiest%20ways,maketrans()%20method.
	# but may not want to remove punctuation for Sent analysis! sometimes punctuation helps us, e.g.  ":("  
	preprocessed_dataset = [review.replace('!', '')
	.replace('"', '')
	.replace('#', '')
	.replace('$', '')
	.replace('%', '')
	.replace('&', '')
	.replace("'", '')
	.replace('(', '')
	.replace(')', '')
	.replace('*', '')
	.replace('+', '')
	.replace(',', '')
	.replace('-', '')
	.replace('.', '')
	.replace('/', '')
	.replace(':', '')
	.replace(';', '')
	.replace('<', '')
	.replace('=', '')
	.replace('>', '')
	.replace('?', '')
	.replace('@', '')
	.replace('[', '')
	.replace('\\', '')
	.replace(']', '')
	.replace('^', '')
	.replace('_', '')
	.replace('`', '')
	.replace('{', '')
	.replace('|', '')
	.replace('}', '')
	.replace('~', '') for review in dataset]

	outputs = sentiment_pipeline(preprocessed_dataset)

	for output in outputs:
		print(output['label'])
		
if __name__ == "__main__":
	predict(['I love you', 'I like you', 'I hate you'])
