# Easy Sentiment Analysis in the Cloud with Hugging Face Transformers and DeepMux

Sentiment analysis is an actual task for NLP, which arises in various fields - from online shopping to social media analysis. In this guiding article, we'll show how to deploy in the cloud and use a state-of-the-art model for sentiment analysis, using just a few lines of code.

*Hugging Face's Transformers* library provides a convenient, easy-to-use API for downloading and running pre-trained NLP models. There are thousands of them for a wide variety of tasks such as text generation, question answering, machine translation, etc. You can find out more at https://github.com/huggingface/transformers

*DeepMux*, in turn, offers a convenient and flexible environment for deploying machine learning systems in the form of GPU-powered Cloud Functions. No knowledge of DevOps is required, and pretty much all the configurations fit in five lines. 

Let's get started and see how easy it actually is!

## Setting up the environment.

To run everything, you need to install deepmux-cli and log in. This is done with the following two commands

```
pip install deepmux-cli
deepmux login 
```

The last command will ask you to enter your unique API token, which you can find on app.deepmux.com.

Let's create a folder for the project:

`mkdir sentiment_analysis && cd sentiment_analysis `

## Writing some code:

Let's create a `main.py` file and write the following code:

```python
import transformers

classifier = transformers.pipeline('sentiment-analysis')

def classify(data):
    global classifier
    return str(classifier(data.decode('utf-8')))
```

That’s it!

## Adding requirements and initializing:

Create a `requirements.txt` file and write the name and version of the library used there:
```
transformers==4.3.2
```
Our project is almost ready! We write the following command:

`deepmux init`

The deepmux.yaml file will appear in the project folder. Fill it in with your data:

```yaml
name: sentiment_analysis # project name
env: <...>
python:
  call: main:classify # file and function to call
  requirements: requirements.txt # path to requirements file
```

To fill in the env line, ru the deepmux env command. The result will be something like this:
```
name: python3.6 language: python
name: python3.7 language: python
name: python3.6-tensorflow2.1-pytorch-1.6-cuda10.1 language: python
name: python3.7-tensorflow2.1-pytorch-1.6-cuda10.1 language: python
name: python3.7-tensorflow2.2-pytorch-1.6-cuda10.1 language: python
```
GPU acceleration is supported by the last three. Choose an environment with the latest versions of pytorch and tensorflow libraries, transformers use them!

The resulting yaml file should look like this:
```yaml
name: sentiment_analysis 
env: python3.7-tensorflow2.2-pytorch-1.6-cuda10.1
python:
  call: main:classify
  requirements: requirements.txt
```
## Loading the model:

Your model is ready to deploy! Now, just call the `deepmux upload` command. This might take a few minutes to process.

## Running the model:

Now the model is uploaded and ready to use. Let's run the model on some line, for example, “Hello, buddy!”

To do this, you need to run the following command:

```
deepmux run --name sentiment_analysis --data “Hello my friend!”
```
Expected output:

`[{'label': 'POSITIVE', 'score': 0.999329686164856}]`

What happens if you write something not so pleasant?

```
deepmux run --name sentiment_analysis --data “I’m not sure I’ll be able to help you”
```

`[{'label': 'NEGATIVE', 'score': 0.9994960427284241}]`









