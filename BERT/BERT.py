from flask import Flask
from flask import render_template
from flask import request
from sentence_transformers import SentenceTransformer, util
import torch
import time


app = Flask(__name__)

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

f = open("dataset.txt", "r", encoding="utf-8")


corpus = f.read()
data = corpus.splitlines()


job = []
desc = []

for i in range(0,len(data)):

    data[i] = data[i].split('-')

    job.append(data[i][0].strip())
    desc.append(data[i][1].strip())

f.close()


corpus_embeddings = embedder.encode(desc, convert_to_tensor=True)


@app.route('/', methods = ['POST', 'GET'])
def index():
	if(request.method == 'POST'):

		top_k = 1
		data = request.form
		query = data['query']
		start = time.time()
		temp = query

		query_embedding = embedder.encode(query, convert_to_tensor=True)
		cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]


		cos_scores = cos_scores.cpu()

		#We use torch.topk to find the highest k scores


		top_results = torch.topk(cos_scores, k=top_k)

		print("\n\n======================\n\n")
		print("Query:", query)

		result = []

		for score, idx in zip(top_results[0], top_results[1]):

			words = job[idx].split()

			for word in words:
				result.append(word.strip())
		end = time.time()
		print(end-start)


		return render_template('index.html', result = result, query = query)


	else:

		return render_template('index.html', result = [], query="")




if(__name__ == '__main__'):
	app.run(debug = 'true')













