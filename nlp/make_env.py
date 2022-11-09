with open('.env', 'w') as file:
    file.write('REDUCED_NER_DATASET=True')
    file.write('NER_EPOCHS=10')
    file.write('X_RAPID_API_KEY=YOUR_API_KEY')
    file.write('X_RAPID_API_HOST=YOUR_API_HOST_GOOGLE')
file.close()