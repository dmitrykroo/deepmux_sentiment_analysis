import transformers

classifier = transformers.pipeline('sentiment-analysis')

def classify(data):
    global classifier
    return str(classifier(data.decode('utf-8')))