from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from rapidfuzz import process, fuzz
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib
import faiss
import torch
import json


PROJECT_DIR = Path(__file__).resolve().parent.parent

class EntityLinker:
    def __init__(
        self,
        device,
        HNSW=False,
        M=32,
        batch_size=2048,
        faiss_index_dir=PROJECT_DIR / "graph_src" / "data" / "entity-faiss.index",
        nodes_csv=PROJECT_DIR / "graph_src" / "data" / "nodes.csv",
        embeddings_file=PROJECT_DIR / "graph_src" / "data" / "nodes_embeddings_98-315.csv",
        pca_file=PROJECT_DIR / "graph_src" / "data" / "pca_model-315.joblib",
        llm_name="ncbi/MedCPT-Query-Encoder",
    ):
        
        self.device = "cpu" if device=="cpu" else "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModel.from_pretrained(llm_name)
        self.model.to(self.device)
        self.model.eval()

        self.save_data_folder = PROJECT_DIR / "graph_src" / "data"
        self.df = pd.read_csv(nodes_csv, usecols=['node_index', 'node_type', 'node_name'])
        self.idxs = self.df['node_index'].tolist()
        self.types = self.df['node_type'].tolist()
        self.names = self.df['node_name'].tolist()

        self.perc_components = 0.98

        if not Path(embeddings_file).exists():
            # Crea embeddings
            all_embeddings = self.embedder(batch_size)
            # Carica/Calcola la PCA
            self.pca = self.pca_compute_or_load(pca_file, all_embeddings)

            embeddings_reduced = self.pca.transform(all_embeddings)
            embeddings_str = [json.dumps(vec.tolist()) for vec in embeddings_reduced]
            dimensions = [str(self.perc_components).split(".")[1], str(embeddings_reduced.shape[1])]
            # print("Shape embeddings ridotti:", dimensions[1])

            output_df = pd.DataFrame({
                "node_index": self.idxs,
                "node_type": self.types,
                "node_name": self.names,
                "embedding": embeddings_str
            })
            # Salva gli Embeddings
            embeddings_file = self.save_node_embeddings(output_df, dimensions)
        else:
            # Carica gli embeddings
            all_embeddings = self.embeddings_loader(embeddings_file)
            
            # Carica/Calcola la PCA
            self.pca = self.pca_compute_or_load(pca_file, all_embeddings)
            # Estrai numero tra l'ultimo "_" e "-", e tra "-" e ".csv"
            filename = Path(embeddings_file).name
            name_no_ext = Path(filename).stem
            
            last_part = name_no_ext.split("_")[-1]
            dim0, dim1 = last_part.split("-")
            dimensions = [dim0, dim1]
            
        if not Path(pca_file).exists():
            # Salva la PCA
            self.save_pca(self.pca, dimensions)

        if not Path(faiss_index_dir).exists():
            self.index = self.construct_index(embeddings_file, h_dim=int(dimensions[1]), HNSW=HNSW, M=M)
        else:
            print("Loading FAISS Index")
            self.index = faiss.read_index(str(faiss_index_dir))

    def link(self, query_word, fuzzy_f=1, sim_k=30, white_spaces=False):
        query_word = query_word["text"]
        embedding = self.create_embeddings(query_word)
        embeddings_reduced = self.pca.transform(embedding)

        faiss.normalize_L2(embeddings_reduced)
        hits = self.index.search(embeddings_reduced, k=sim_k)
        top_k = self.df.iloc[hits[1][0]]

        database_strings = top_k['node_name'].tolist()

        # print(database_strings)
        if white_spaces:
            query_word = query_word.replace(' ', '')
            database_strings = [i.replace(' ', '') for i in database_strings] # nel caso volessi usare metodo senza spazi

        scorer = fuzz.ratio if ' ' not in query_word and not white_spaces else fuzz.WRatio

        fuzzy_candidate = process.extract(
            query_word,
            database_strings,
            scorer=scorer,
            limit=fuzzy_f
        )

        # If more fuzzy candidates are retrieved they will be in faiss order
        fuzzy_candidate = [x[0] for x in fuzzy_candidate]
        # print(fuzzy_candidate)
        results_df = top_k[top_k['node_name'].isin(fuzzy_candidate)]

        subset_df = results_df[['node_index', 'node_type', 'node_name']]
        result = subset_df.values.squeeze().tolist()

        return result
    
    def embedder(self, batch_size):
        print("Creating Embeddings")
        all_embeddings = []
        for i in tqdm(range(0, len(self.names), batch_size), desc="Calcolo embedding", ncols=100):
            batch_names = self.names[i:i+batch_size]
            batch_embeddings = self.create_embeddings(batch_names)
            
            all_embeddings.append(batch_embeddings)

        all_embeddings = np.vstack(all_embeddings)
        # print("Shape embeddings originali:", all_embeddings.shape)

        return all_embeddings

    def create_embeddings(self, words):
        inputs = self.tokenizer(words, padding=True, truncation=False, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            # print(words, last_hidden_state.shape)
            batch_embeddings = last_hidden_state.mean(dim=1).cpu().numpy()
            # takes all the embedding values obtained by tokenization of a word/sentence and compute the mean of them

        return batch_embeddings
    
    def construct_index(self, embedding_csv, h_dim, HNSW=False, M=32):
        print("Building FAISS Index")
        if HNSW:
            index = faiss.IndexHNSWFlat(h_dim, M)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            index = faiss.IndexFlatIP(h_dim)

        df = pd.read_csv(embedding_csv)

        embeddings = df['embedding'].apply(lambda x: np.array(eval(x), dtype='float32'))
        embeddings_matrix = np.stack(embeddings.to_numpy())

        faiss.normalize_L2(embeddings_matrix)
        index.add(embeddings_matrix)
        faiss.write_index(index, str(self.save_data_folder / "entity-faiss.index"))

        print(f"Faiss index built with {index.ntotal} vectors")
        
        return index
    
    def save_pca(self, pca, dimensions):
        print("Saving PCA on file")
        output_path = self.save_data_folder / ("pca_model-" + dimensions[1] + ".joblib")
        joblib.dump(pca, output_path)
        print("PCA file saved in: " + str(output_path))

    def pca_compute_or_load(self, pca_file, all_embeddings):
        if not Path(pca_file).exists():
            print("Computing PCA")
            pca = PCA(n_components=self.perc_components)
            pca.fit(all_embeddings)
        else:
            print("Loading PCA")
            pca = joblib.load(pca_file)
        
        return pca

    def save_node_embeddings(self, output_df, dimensions):
        print("Saving Embeddings on file")
        output_csv = "nodes_embeddings_" + dimensions[0] + "-" + dimensions[1] + ".csv"
        output_path = self.save_data_folder / output_csv
        output_df.to_csv(output_path, index=False)
        print(f"Node Embeddings CSV saved in: {output_path}")
        
        return output_path
    
    def embeddings_loader(self, embeddings_file):
        print("Loading Embeddings")
        df = pd.read_csv(embeddings_file)
        embeddings = df['embedding'].apply(lambda x: np.array(json.loads(x), dtype='float32'))
        embeddings = np.stack(embeddings.to_numpy())

        return embeddings
