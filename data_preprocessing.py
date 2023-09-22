#Text Data Preprocessing Lib
import nltk

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle
import numpy as np

words=[]
classes = [] #tags
word_tags_list = [] 
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

# function for appending stem words
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)  
    return stem_words

#loop to extract all the words in patterns under intents and adding all 4 tags in classes list. Then calling get_stem_words()
for intent in intents['intents']:
    
        # Add all words of patterns to list
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)
            #print(pattern_word)            
            words.extend(pattern_word)                      
            word_tags_list.append((pattern_word, intent['tag']))
        # Add all 4 tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignore_words)

#print(stem_words)
print(words,"\n")
#print("Word Tag List: ",word_tags_list) 
print(classes,"\n")   

# Create word corpus for chatbot - stem words and classes list for each pattern; 
# corpus means training data file with data as 0's and 1's. File format .pkl

def create_bot_corpus(stem_words, classes):

    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))
    #print(stem_words)
    #print(classes)
    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes

stem_words, classes = create_bot_corpus(stem_words,classes)  

print("\n Stem word: ",stem_words)
print("\n Classes: ",classes)

training_data = []
number_of_tags = len(classes)
labels = [0]*number_of_tags
#print(labels)

# SA1 - Create bag of words and labels_encoding
for word_tags in word_tags_list:
        
        bag_of_words = []       
        pattern_words = word_tags[0] #only words not tags
        #print(word_tags)
        for word in pattern_words:
            index=pattern_words.index(word)
            word=stemmer.stem(word.lower())
            pattern_words[index]=word
        print("\nPatterns: ",pattern_words)  

        



# SA2 - Create training data
#def preprocess_train_data(training_data):
   
   





