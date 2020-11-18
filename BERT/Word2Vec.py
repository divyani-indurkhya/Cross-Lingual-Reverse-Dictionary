from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify
from googletrans import Translator


translator = Translator()

import gensim
from gensim.models import KeyedVectors


dict_words = []
f = open("words.txt", "r", encoding="utf-8")


for word in f:
	dict_words.append(word.strip())


f.close()
model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 


def find_words(definition):

	possible_words = definition.split()


	for i in range(len(possible_words)-1,-1,-1):
		if possible_words[i] not in model.vocab:
			del possible_words[i]


	possible_expressions = []
	for w in [possible_words[i:i+3] for i in range(len(possible_words)-3+1)]:
		possible_expressions.append('_'.join(w))


	ex_to_remove = []
	for i in range(len(possible_expressions)):
		if possible_expressions[i] in model.vocab:
			ex_to_remove.append(i)



	words_to_remove = []
	for i in ex_to_remove:
		words_to_remove+=[i, i+1, i+2]
	words_to_remove = sorted(set(words_to_remove))
	words = [possible_expressions[i] for i in ex_to_remove]


	for i in range(len(possible_words)):
		if i not in words_to_remove:
			words.append(possible_words[i])

	similar = [i[0] for i in model.most_similar(positive=words, topn=30)]
	ans = []
	for word in similar:
		if(word in dict_words):
			ans.append(word)

			
	if(len(ans) > 20):
		ans = ans[0:20]
	ans = [translator.translate(word, dest="hi") for word in ans]
	return ans

q = input("Enter your definition:")
ans = find_words(q)
for word in ans:
	print(word.text)
# app = Flask(__name__)
# @app.route('/', methods=['GET', 'POST'])
# def index():
# 	return render_template("index.html")

# @app.route('/get_results', methods=['GET', 'POST'])
# def get_results():
# 	q = request.form['query']
# 	return jsonify(find_words(q))


# if(__name__=='__main__'):
# 	app.run(debug=True)