
def summarize(para):
    import networkx as nx
    import numpy as np
    import pandas as pd
    import nltk
    from nltk.tokenize import sent_tokenize
    # nltk.download('punkt')  # one time execution
    import re

    sentences = []
    sentences.append(sent_tokenize(para))

    # flatten the list
    sentences = [y for x in sentences for y in x]
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    # nltk.download('stopwords')  # one time execution
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')

    # function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    # download pretrained GloVe word embeddings
    # ! wget http://nlp.stanford.edu/data/glove.6B.zip
    # ! unzip glove*.zip

    # Extract word vectors
    word_embeddings = {}
    print("opening glove...")
    f = open('summarizer/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    print("glove closed...")
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    # print("timeStamp 1")
    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    from sklearn.metrics.pairwise import cosine_similarity
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = \
                cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]
    # print("timeStamp 2")
    # import networkx as nx
    # print("timeStamp 2a")
    nx_graph = nx.from_numpy_array(sim_mat)
    # print("timeStamp 2b")
    scores = nx.pagerank(nx_graph)
    # print("timeStamp 2c")
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    # print("timeStamp 3")
    # Specify number of sentences to form the summary
    sn = 5

    # Generate summary
    summary = ""
    for i in range(sn):
        summary += ranked_sentences[i][1]
    # print("timeStamp 4")
    return summary
