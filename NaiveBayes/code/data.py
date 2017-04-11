import numpy as np

def loadData1():
    #a row equals a document
    dataSet=[["my","dog","has","flea","problems","help","please"],
             ["maybe","not","take","him","to","dog","park","stupid"],
             ["my","dalmation","is","so","cute","I","love","him"],
             ["stop","posting","stupid","worthless","garbage"],
             ["mr","licks","ate","my","steak","how","to","stop","him"],
             ["quit","buying","worthless","dog","food","stupid"]]

    labels=[0,1,0,1,0,1]
    return dataSet,labels

def createWordList(dataset):
    wordList=set([])
    for doc in dataset:
        wordList=wordList | set(doc)
    return list(wordList)

#trans a splited doc to a vector
def docToVector(wordList,doc):
    #the same shape of wordList
    vec=[0]*len(wordList)
    for word in doc:
        if word in wordList:
            vec[wordList.index(word)]=1
        else:
            print("the word",word,"is not in wordList")

    return np.array(vec)

#trans a dataset to feature Matrix
def datasetToMatrix(wordList,dataSet):
    matrix=[]
    for doc in dataSet:
        vec=docToVector(wordList,doc)
        matrix.append(vec)

    return np.array(matrix)


#return probability vectors
def train(trainMatrix,labels):
    num_of_docs=trainMatrix.shape[0]
    num_of_word=trainMatrix.shape[1]
    p_positive=sum(labels)/float(num_of_docs)
    p_negtive=1-p_positive
    '''
    positive_num_words_vector=np.zeros(shape=(num_of_word,))
    negtive_num_words_vector=np.zeros(shape=(num_of_word,))

    positive_total_word=0.0
    negtive_total_word=0.0
    '''
    positive_num_words_vector = np.ones(shape=(num_of_word,))
    negtive_num_words_vector = np.ones(shape=(num_of_word,))

    positive_total_word = 2.0
    negtive_total_word = 2.0

    for i in range(num_of_docs):
        if(labels[i]==1):
            positive_num_words_vector+=trainMatrix[i]
            positive_total_word+=np.sum(trainMatrix[i])
        else:
            negtive_num_words_vector+=trainMatrix[i]
            negtive_total_word+=np.sum(trainMatrix[i])

    '''
    prob_positive_vector=positive_num_words_vector/positive_total_word
    prob_negtive_vector=negtive_num_words_vector/negtive_total_word
    '''
    prob_positive_vector = np.log(positive_num_words_vector / positive_total_word)
    prob_negtive_vector = np.log(negtive_num_words_vector / negtive_total_word)

    return prob_positive_vector,prob_negtive_vector,p_positive,p_negtive


def classify(vec,prob_positive_vec,prob_negtive_vec,p_positive,p_negtive):
    p1=np.sum(vec*prob_positive_vec)+np.log(p_positive)
    p0 = np.sum(vec * prob_negtive_vec) + np.log(p_negtive)
    if p1>p0:
        return 1
    else:
        return 0



#test
dataSet,labels=loadData1()
wordList=createWordList(dataSet)
#print(wordList)
vec=docToVector(wordList,dataSet[0])
#print(vec)

trainMatrix=datasetToMatrix(wordList,dataSet)
#print("trainMatrix:",trainMatrix)

p1v,p0v,p1,p0=train(trainMatrix,labels)
print("p1v\n",p1v)
print("p0v\n",p0v)
print("p1\n",p1)
print("p0\n",p0)

testDoc=["stupid","garbage"]
testVec=docToVector(wordList,testDoc)

testDoc1=["stupid","love","dalmation"]
testVec1=docToVector(wordList,testDoc1)

testDoc2=["my","love","dalmation"]
testVec2=docToVector(wordList,testDoc2)

print(classify(testVec,p1v,p0v,p1,p0))
print(classify(testVec1,p1v,p0v,p1,p0))
print(classify(testVec2,p1v,p0v,p1,p0))

