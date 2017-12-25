#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import io 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 

#The generator iterates through every single file in a directory, 
#then it reads it in.
#While reading it skips the e-mail's header by looking for the first blank line.

def readFiles(path): 
    for root, dirnames, filenames in os.walk(path): 
        for filename in filenames: 
            path = os.path.join(root, filename) 
 
            inBody = False 
            lines = [] 
            f = io.open(path, 'r', encoding='latin1') 
            for line in f:
                if inBody: 
                    lines.append(line) 
                elif line == '\n': 
                    inBody = True 
            f.close() 
            message = '\n'.join(lines) 
            yield path, message 
 
# The function reads e-mails from  a folder and puts it to a DataFrame.
            
def dataFrameFromDirectory(path, classification): 
    rows = [] 
    index = [] 
    for filename, message in readFiles(path): 
        rows.append({'message': message, 'class': classification}) 
        index.append(filename) 
 
    return pd.DataFrame(rows, index=index) 

# Creating a DataFrame that contains two columns, the text from messages
# and class of e-mails, either spam or normal.
    
data = pd.DataFrame({'message': [], 'class': []}) 

# Throwing into data all the e-mails from spam and normal folders
 
data = data.append(dataFrameFromDirectory('emails/spam/', 'spam')) 
data = data.append(dataFrameFromDirectory('emails/normal/', 'normal'))

vectorizer = CountVectorizer()

# A list of all the words tokenized in each e-mail and the number 
# of times that word occurs. 

counts = vectorizer.fit_transform(data['message'].values)

# Classification data for each e-mail

targets = data['class'].values

# Creating a model using Naive Bayes, which will predict whether new e-mails 
# are spam or not.

classifier = MultinomialNB()
classifier.fit(counts, targets)

# A list of examples

examples = ['Free Money now!!!', "Hi Bob, how about a game of golf tomorrow?"]

# Converting the examples into the same format that I trained my model on,
# using the same vectorizer that I created when creating the model
# to convert each message into a list of words and their frequencies, 
# where the words are represented by positions in an array.

example_counts = vectorizer.transform(examples)

# Using the predict() function on the classifier, on the array of examples.

predictions = classifier.predict(example_counts)
print(predictions)