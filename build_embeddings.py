import numpy
import codecs
import gensim

# glove reading
def glove_embed(filename):
	embeddings_index = dict()
	gf = codecs.open(filename)
	for line in gf:
		values = line.split()
		word = values[0]
		coefs = numpy.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	gf.close()
	return embeddings_index

# create a weight matrix for words in training docs
def build_embedmatrix(tokenizer, file_name, vocab_size, embed_type):
	if embed_type == "glove":
		embeddings_index = glove_embed(file_name)
		embedding_matrix = numpy.zeros((vocab_size, len(embeddings_index[0])))
		for word, i in tokenizer.word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
	if embed_type == "w2vec":
		w2v_model = gensim.models.KeyedVectors.load_word2vec_format("embed_files/GoogleNews-vectors-negative300.bin.gz",
																binary=True)
		vocab = w2v_model .vocab.keys()
		embedding_matrix = numpy.zeros((vocab_size, len(w2v_model["book"])))
		for word, i in tokenizer.word_index.items():
			if word in vocab:
				embedding_vector = w2v_model [word]
				if embedding_vector is not None:
					embedding_matrix[i] = embedding_vector
	return embedding_matrix

