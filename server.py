from flask import Flask, render_template, request, url_for
import numpy as np
import csv

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/enterparams", methods=["GET", "POST"])
def enterparams():
	rcover = request.form.get("rcover")
	overlap = request.form.get("overlap")
	lens = request.form.get("Lens")

	eps = request.form.get("eps")
	min_samples = request.form.get("min_samples")
	#strength = int(request.form.get("strength"))
	
    #if not rcover or not overlap or not lens:
     #   return render_template("failure.html")
    
	 
	import mapper as mp

	dataset = request.form.get("Dataset")

	if dataset == "IRIS":	
		
		from sklearn.datasets import fetch_mldata
		data = fetch_mldata('iris').data.tolist()
	
	elif dataset == "MNIST":

		from sklearn.datasets import fetch_mldata
		mnist = fetch_mldata('MNIST original')
		dataf = mnist.data[::70,:]
		data = dataf.astype(np.float32).tolist()

		#from sklearn.manifold import TSNE
		#data_embedded = TSNE(n_components = 10).fit_transform(data)
		#data = data_embedded
	
	elif dataset == "BREAST_CANCER":
		from sklearn.datasets import load_breast_cancer
		dataf = load_breast_cancer()
		data = dataf.data.tolist()

	elif dataset == "BLOBS":
		from sklearn.datasets.samples_generator import make_blobs
		centers = [[1, 1], [-1, -1] , [1,-1]]
		X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.3, random_state=0)
		data = X.tolist()

	
	elif dataset == "OTHER": 
		f = open("datasets/winequality-white.csv", "r")
		dataf = list(csv.reader(f, delimiter = ";"))
		f.close()
		data = [[float(x) for x in a[0:-1]] for a in dataf[1:]]


	else: pass

	

	out = mp.Mapper(lens = lens, clusterer = "DBSCAN", n_rcover = [int(rcover), float(overlap)], clusterer_params  = (float(eps),int(min_samples)))
	out.write_to_json(data)
	url = url_for("static", filename = "data/mapperViz.json")
	return render_template("mapperViz.html", url = url)
	


if __name__ == ("__main__"):
	app.run(debug=True)


