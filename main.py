# import modules
import json
import string
import random
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout

nltk.download("punkt")  # required for tokenization
nltk.download("wordnet")  # Word database
nltk.download('omw-1.4')

# Creates JSON file
ourData = {"intents": [

    {"tag": "age",
     "patterns": ["how old are you?", "Age?", "You're sexy"],
     "responses": ["I am 2 years old"]
     },
    {"tag": "greeting",
     "patterns": ["Hi", "Hello", "Hey"],
     "responses": ["Hi there", "Hello", "Hi :)"],
     },
    {"tag": "goodbye",
     "patterns": ["bye", "later"],
     "responses": ["Bye", "take care"]
     },
    {"tag": "name",
     "patterns": ["what's your name?", "who are you?"],
     "responses": ["I have no name yet," "You can give me one, and I will appreciate it"]
     }
]}

# Processed data
lm = WordNetLemmatizer()  # Gets words

# lists
ourClasses = []
newWords = []
documentX = []
documentY = []

# Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists
for intent in ourData["intents"]:
    for pattern in intent["patterns"]:
        ourTokens = nltk.word_tokenize(pattern)  # Tokenize patterns
        newWords.extend(ourTokens)
        documentX.append(pattern)
        documentY.append(intent["tag"])

    if intent["tag"] not in ourClasses:
        ourClasses.append(intent["tag"])

newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation]
newWords = sorted(set(newWords))  # sorting words
ourClasses = sorted(set(ourClasses))  # sorting classes

# Designing of the neural network model
trainingData = []
outEmpty = [0] * len(ourClasses)

for idx, doc in enumerate(documentX):
    bagOfwords = []
    text = lm.lemmatize(doc.lower())
    for word in newWords:
        bagOfwords.append(1) if word in text else bagOfwords.append(0)

    outputRow = list(outEmpty)
    outputRow[ourClasses.index(documentY[idx])] = 1
    trainingData.append([bagOfwords, outputRow])

random.shuffle(trainingData)
trainingData = num.array(trainingData, dtype=object)

x = num.array(list(trainingData[:, 0]))  # First training phase
y = num.array(list(trainingData[:, 1]))  # Second training phase

iShape = (len(x[0]),)
oShape = len(y[0])

# Parameter definition
ourNewModel = Sequential()

ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))
ourNewModel.add(Dropout(0.5))

ourNewModel.add(Dense(64, activation="relu"))
ourNewModel.add(Dropout(0.3))
ourNewModel.add(Dense(oShape, activation="softmax"))

md = keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
ourNewModel.compile(loss='categorical_crossentropy',
                    optimizer=md,
                    metrics=['accuracy'])
print(ourNewModel.summary())
ourNewModel.fit(x, y, epochs=200, verbose=1)


# Useful features
def ourText(text):
    newtkns = nltk.word_tokenize(text)
    newtkns = [lm.lemmatize(word) for word in newtkns]
    return newtkns


def wordBag(text, vocab):
    newtkns = ourText(text)
    bagOwords = [0] * len(vocab)
    for w in newtkns:
        for idx, word in enumerate(vocab):
            if word == w:
                bagOwords[idx] = 1
    return num.array(bagOwords)


def Pclass(text, vocab, labels):
    bagOwords = wordBag(text, vocab)
    ourResult = ourNewModel.predict(num.array([bagOwords]))[0]
    newThresh = 0.2
    yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

    yp.sort(key=lambda x: x[1], reverse=True)
    newList = []
    for r in yp:
        newList.append(labels[r[0]])
    return newList


def getRes(firstlist, fJson):
    tag = firstlist[0]
    listOfIntents = fJson["intents"]
    for i in listOfIntents:
        if i["tag"] == tag:
            ourResult = random.choice(i["responses"])
            break
    return ourResult


while True:
    newMessage = input("")
    intents = Pclass(newMessage, newWords, ourClasses)
    ourResult = getRes(intents, ourData)
    print(ourResult)
