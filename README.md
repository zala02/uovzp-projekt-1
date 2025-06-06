### ZAGON KODE

Kodo se enostavno zažene z zagonom glavne datoteke main.py, npr:
```
python3 main.py
```

Po nekaj minutah se prikaže prva glavna vizualizacija člankov, narejena s t-SNE, po zaprtju le te, se pojavi še druga, narejena s PCA.

  
### UPORABLJENE KNJIŽNICE

V nadaljevanju so predstavljene uporabljene knjižnice. Za osnovni zagon potrebujete le tiste, ki so odebeljene. Ostale so bile potrebne za delovanje celotne kode in so v sami kodi zakomentirane (za nemoten proces vizualizacije). Za hitro namestitev osnovnih nujnih knjižnice uporabite naslednji ukaz:

```
pip install numpy scipy joblib scikit-learn matplotlib
```

Osnovne knjižnice:
- **os**
- **json**
- yaml
- re
- **numpy**
- tqdm

Shranjevanje:
- save_npz (iz spicy.sparse)
- **load_npz (iz spicy.sparse)**
- dump (iz joblib)
- **load (iz joblib)**

Predprocesiranje podatkov:
- classla (opozorilo: precej velika knjižnica)
- re

Pretvorba besedil:
- TfidfVectorizer (iz sklearn.feature_extraction.text)

Zmanjšanje dimenzij podatkov:
- TruncatedSVD (iz sklearn.decomposition)
- SparsePCA (iz sklearn.decomposition)
- **Normalizer (iz sklearn.preprocessing)**
- **TSNE (iz sklearn.manifold)**
- **PCA (iz sklearn.decomposition)**

Gručenje:
- DBSCAN (iz sklearn.cluster)
- KMeans (iz sklearn.cluster)
- AgglomerativeClustering (iz sklearn.cluster)
- StandardScaler (iz sklearn.preprocessing)
- NearestNeighbors (iz sklearn.neighbors)

Ovrednotenje:
- silhouette_score (iz sklearn.metrics)

Vizualizacija:
- **plt (iz matplotlib.pyplot)**
- sns (iz seaborn)
- defaultdict (iz collections)
- random
- **adjust_text (iz adjustText)**
- dendrogram (iz scipy.cluster.hierarchy)
- linkage (iz scipy.cluster.hierarchy)

Lastne pomožne knjižnice (ni treba nalagati):
- **my_colors iz color_pallete.py**
- **stopwords-slovene.txt**


### POUSTVARJENJE VMESNIH REZULTATOV

Ves postopek je zakomentiran v main.py. Če želimo poustvariti vmesne rezultate, lahko cel del odkomentiramo (končnega pa zakomentiramo, da se ne ponavljamo), rezultati se shranijo v mapo 'resources'. Sicer pa tega priporočam, saj traja precej časa. 

--- 

Za predprocesiranje podatkov zaženemo naslednji dve funkciji:
```
articles = load_articles()
preprocess(articles)
```

Pozor - load_articles() predvideva, da so članki v resources/articles.yaml. Sicer pa funkcija preprocess() sama ustvari datoteko resources/cleaned_text.json. Za dostop do nje uporabimo ukaza

```
clean_text_path = os.path.join("resources", "cleaned_text.json")
load_data(clean_text_path)`
```
Dobimo json datoteko, kjer je vsak člen sprocesiran odstavek članka, pretvorjen v male začetnice, počiščen nekaterih znakov, lematiziran, odstranjene so stop besede. Stop besede se nahajajo v pomožni datoteki resources/stopwords_slovene.txt.

---

Za ustvarjenje pretvorjenega besedila v vektorje zaženemo naslednji niz ukazov:
```
clean_text_path = os.path.join("resources", "cleaned_text.json")
tfidf_sparse_path = os.path.join("resources", "tfidf_sparse.npz")
tfidf_feauter_names_path = os.path.join("resources", "tfidf_feauter_names.json")

create_tfidf(clean_text_path, tfidf_sparse_path, tfidf_feauter_names_path)
```

Funkcija create_tfidf prebere prečiščene podatke in jih pretvori v sparse matriko. To shranimo, shranimo pa tudi posamezne besede.

---

Preden se lotimo dejanskega gručenja, si ustvarimo še nekaj pomožnih datotek.

Kosinusna razdalja in podobnost. Ta nam koristi, če želimo videti hierarhično gručenje ali pa DBSCAN. Datoteki sta precej veliki (vsaka 7GB):

```
cosine_sim_matrix_path = os.path.join("resources", "cosine_sim_matrix.npy")
create_cosine_sim_matrix(tfidf_sparse, tfidf_sparse, cosine_sim_matrix_path)
cosine_similarity_matrix = np.load(cosine_sim_matrix_path)

cosine_dist_matrix_path = os.path.join("resources", "cosine_dist_matrix.npy")
create_cosine_dist_matrix(tfidf_sparse, tfidf_sparse, cosine_dist_matrix_path)
cosine_dist_matrix = np.load(cosine_dist_matrix_path)
```

TF-IDF matrika je trenutno v obliki sparse, potrebovali pa bi dense. Direktno pretvorjenje žal ne gre, saj bi potrebovali ogromno prostora, zato moramo zmanjšati dimenzije, kar naredimo s TruncatedSVD:
```
tfidf_truncatedSVD_2000_path = os.path.join("resources", "tfidf_truncatedSVD_2000.npz")
tfidf_truncatedSVD = create_truncatedSVD(tfidf_sparse, tfidf_truncatedSVD_2000_path)

with np.load(tfidf_truncatedSVD_2000_path) as data:
    tfidf_truncatedSVD_dense = data['svd_matrix']
    tfidf_truncatedSVD_variance = data['explained_variance']
    tfidf_truncatedSVD_cumulative = data['cumulative_variance']
    tfidf_truncatedSVD_components = data['components']
```
Na koncu je treba še normalizirati:
```
normalizer = Normalizer(copy=False)
tfidf_truncatedSVD_dense_normalized = normalizer.transform(tfidf_truncatedSVD_dense)
```

Dobimo normalizirano reducirano dense matriko, s katero lahko normalno gručimo.


---

Na koncu sledi še gručenje. V zakomentiranem delu so prikazana še ostala možna gručenja, testirana na množici (to so hierarhično gručenje - ward, complete, single in average linkage, in DBSCAN). Za podrobnosti glejte zakomentirani del main funkcije. Od vseh gručenj se je najbolj izkazal k-means, zato je tukaj opisan postopek zgolj k-means vmesnega rezultata.

Potrebujemo zgolj narediti model, ki se je izkazal za najboljšega, in ga shraniti v mapo resources:
```
kmeans_model_path = os.path.join("resources", "best_kmeans_model.joblib")
create_best_kmeans(tfidf_truncatedSVD_dense_normalized, kmeans_model_path)
```

In nato model še naložiti ter poslati v funkcijo za vizualizacijo:
```
kmeans_model = load(kmeans_model_path)
visualize_best_kmeans(tfidf_truncatedSVD_dense_normalized, tfidf_feauter_names, tfidf_truncatedSVD_components, kmeans_model)
```

Dobimo vizualizacijo modela v t-SNE in nato po zaprtju okenca še v PCA. To si lahko ogledamo tudi v datoteki rezultati in sicer slika z imenom kmeans_k=28_tsne03.pgn oz. kmeans_k=28_pca03.pgn. V tej datoteki najdemo najboljše vizualizacije tudi ostalih implementiranih metod. Vredno ogleda je predvsem hierarhični ward, ki je primerljiv s kmeans metodo, ostali pa so kljub dolgemu iskanju dobrih parametrov precej slabi.

