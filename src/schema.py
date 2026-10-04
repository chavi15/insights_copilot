from collections import deque
from functools import lru_cache

import yaml

from src.config import SCHEMA_DOCS_PATH


class SchemaDocs:
    def __init__(self, data):
        self.notes = data["dialect_notes"]
        self.tables = data["tables"]
        self.joins = [tuple(j) for j in data["joins"]]
        self.graph = {name: {} for name in self.tables}
        for a, b, key in self.joins:
            self.graph[a][b] = key
            self.graph[b][a] = key

    @property
    def table_names(self):
        return list(self.tables)

    def table_document(self, name):
        table = self.tables[name]
        columns = " ".join(f"{col} {meta['description']}." for col, meta in table["columns"].items())
        return f"{name}: {table['description']} Columns: {columns} Keywords: {' '.join(table.get('keywords', []))}"

    def shortest_path(self, source, target):
        if source == target:
            return [source]
        seen = {source: None}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbour in self.graph[node]:
                if neighbour not in seen:
                    seen[neighbour] = node
                    if neighbour == target:
                        path = [target]
                        while seen[path[-1]] is not None:
                            path.append(seen[path[-1]])
                        return path[::-1]
                    queue.append(neighbour)
        return [source, target]

    def expand_join_paths(self, selected):
        selected = [t for t in selected if t in self.tables]
        result = set(selected)
        for i, a in enumerate(selected):
            for b in selected[i + 1 :]:
                result.update(self.shortest_path(a, b))
        return [name for name in self.tables if name in result]

    def render(self, names):
        names = [n for n in self.tables if n in set(names)]
        blocks = []
        for name in names:
            table = self.tables[name]
            lines = [f"TABLE {name} -- {table['description']}"]
            for col, meta in table["columns"].items():
                lines.append(f"  {col} {meta['type']} -- {meta['description']}")
            blocks.append("\n".join(lines))
        edges = [f"{a}.{key} = {b}.{key}" for a, b, key in self.joins if a in names and b in names]
        text = "\n\n".join(blocks)
        if edges:
            text += "\n\nJOIN KEYS:\n" + "\n".join(f"  {e}" for e in edges)
        return text

    def render_notes(self):
        return "\n".join(f"- {n}" for n in self.notes)


@lru_cache(maxsize=4)
def load_schema(path=str(SCHEMA_DOCS_PATH)):
    with open(path, encoding="utf-8") as handle:
        return SchemaDocs(yaml.safe_load(handle))
