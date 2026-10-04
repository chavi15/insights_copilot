import math
import os
import re
from collections import Counter

from src.config import CHROMA_DIR
from src.schema import load_schema

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from how in is it of on or per that the their there to was were what which who with each all me show list give "
    "total number many much did does do has have had than then them this these those across over under into about between during".split()
)


def _stem(token):
    for suffix in ("ing", "ies", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text):
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class KeywordRetriever:
    name = "keyword"

    def __init__(self, schema=None):
        self.schema = schema or load_schema()
        self.docs = {}
        for name in self.schema.table_names:
            table = self.schema.tables[name]
            weighted = tokenize(name.replace("_", " ")) * 3
            weighted += tokenize(" ".join(table.get("keywords", []))) * 3
            weighted += tokenize(table["description"]) * 2
            for col, meta in table["columns"].items():
                weighted += tokenize(col.replace("_", " "))
                weighted += tokenize(meta["description"])
            self.docs[name] = Counter(weighted)
        document_frequency = Counter()
        for counts in self.docs.values():
            document_frequency.update(counts.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + n / df) for t, df in document_frequency.items()}

    def rank(self, question):
        query = Counter(tokenize(question))
        scores = {}
        for name, counts in self.docs.items():
            length = sum(counts.values())
            score = 0.0
            for token in query:
                tf = counts.get(token, 0)
                if tf:
                    score += self.idf.get(token, 0.0) * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * length / 40))
            scores[name] = score
        return sorted(scores, key=lambda n: (-scores[n], n))

    def top_tables(self, question, k=3):
        ranked = self.rank(question)
        return [n for n in ranked[:k] if self.rank_score_positive(question, n)] or ranked[:k]

    def rank_score_positive(self, question, name):
        query = set(tokenize(question))
        return any(t in self.docs[name] for t in query)


class ChromaRetriever:
    name = "chroma"

    def __init__(self, schema=None, path=CHROMA_DIR):
        import chromadb

        self.schema = schema or load_schema()
        client = chromadb.PersistentClient(path=str(path))
        documents = [self.schema.table_document(n) for n in self.schema.table_names]
        collection = client.get_or_create_collection("schema_tables", metadata={"hnsw:space": "cosine"})
        stored = collection.get()
        if sorted(stored["ids"]) != sorted(self.schema.table_names) or stored["documents"] != [
            documents[self.schema.table_names.index(i)] for i in stored["ids"]
        ]:
            client.delete_collection("schema_tables")
            collection = client.get_or_create_collection("schema_tables", metadata={"hnsw:space": "cosine"})
            collection.add(ids=self.schema.table_names, documents=documents)
        self.collection = collection

    def top_tables(self, question, k=3):
        result = self.collection.query(query_texts=[question], n_results=min(k, len(self.schema.table_names)))
        return list(result["ids"][0])


class TableRetriever:
    def __init__(self, backend, schema=None, k=3):
        self.backend = backend
        self.schema = schema or load_schema()
        self.k = k
        self.name = backend.name

    def tables_for(self, question):
        top = self.backend.top_tables(question, self.k)
        return self.schema.expand_join_paths(top)

    def context_for(self, question):
        return self.schema.render(self.tables_for(question))


def get_retriever(kind=None, k=3):
    kind = (kind or os.getenv("RETRIEVER") or "chroma").lower()
    schema = load_schema()
    if kind == "chroma":
        try:
            return TableRetriever(ChromaRetriever(schema), schema, k)
        except Exception as error:
            print(f"[retrieval] ChromaDB unavailable ({type(error).__name__}); using keyword retriever instead.")
    return TableRetriever(KeywordRetriever(schema), schema, k)
