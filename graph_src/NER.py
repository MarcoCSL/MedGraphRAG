from pathlib import Path
import spacy


PROJECT_DIR = Path(__file__).resolve().parent.parent

class NER:
    def __init__(self, model_names=["en_ner_jnlpba_md", "en_ner_bc5cdr_md", "en_ner_bionlp13cg_md", PROJECT_DIR / "graph_src" / "health_ner"]):
        print("Loading NER models")
        self.models = [spacy.load(name) for name in model_names]
        print("NER models loaded successfully")

        self.label_map = {
            "ANATOMICAL_SYSTEM": "anatomy",
            "CELL": "anatomy",
            "CELLULAR_COMPONENT": "cellular_component",
            "DEVELOPING_ANATOMICAL_STRUCTURE": "anatomy",
            "IMMATERIAL_ANATOMICAL_ENTITY": "anatomy",
            "MULTI-TISSUE_STRUCTURE": "anatomy",
            "ORGAN": "anatomy",
            "ORGANISM_SUBDIVISION": "anatomy",
            "ORGANISM_SUBSTANCE": "anatomy",
            "PATHOLOGICAL_FORMATION": "anatomy",
            "TISSUE": "anatomy",

            "GENE_OR_GENE_PRODUCT": "gene_protein",
            "PROTEIN": "gene_protein",
            "DNA": "gene_protein",
            "RNA": "gene_protein",

            "DISEASE": "disease",
            "CANCER": "disease",

            "CHEMICAL": "drug",
            "SIMPLE_CHEMICAL": "drug",

            # Not caught by ordinary NER (caught by my ruler!)
            "ANATOMY": "anatomy",
            "DRUG": "drug",
            "BIOLOGICAL_PROCESS": "biological_process",
            "MOLECULAR_FUNCTION": "molecular_function",
            "PATHWAY": "pathway",
            "EFFECT_PHENOTYPE": "effect_phenotype",
            "EXPOSURE": "exposure",
        }

        self.target_labels = set(self.label_map.values())

    def find(self, query, subset_deleter=True):
        all_entities = []
        for ner in self.models:
            doc = ner(query)
            for ent in doc.ents:
                mapped_label = self.label_map.get(ent.label_)
                if mapped_label in self.target_labels:
                    all_entities.append((ent.text.lower().strip(), mapped_label, ent.start_char, ent.end_char))

        seen = set()
        unique_entities = []

        for text, label, start, end in all_entities:
            if text not in seen:
                seen.add(text)
                unique_entities.append({
                    "text": text,
                    "label": label,
                    "start": start,
                    "end": end
                })

        if subset_deleter:
            unique_entities.sort(key=lambda e: e["end"] - e["start"], reverse=True)
            filtered = []
            for candidate in unique_entities:
                is_subspan = False
                for kept in filtered:
                    if (candidate["start"] >= kept["start"] and candidate["end"] <= kept["end"]):
                        is_subspan = True
                        break
                if not is_subspan:
                    filtered.append(candidate)
            unique_entities = filtered
        
        return unique_entities


# if __name__ == "__main__":
#     ner = NER()
#     result = ner.find("TP53 is a tumor suppressor gene often mutated in breast cancer.", True)
#     print(result)
