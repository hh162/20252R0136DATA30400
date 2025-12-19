import collections

def load_classes(path):
    id2name = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cid, name = line.strip().split("\t")
            id2name[int(cid)] = name
    return id2name

def load_keywords(path):
    name2keywords = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name, kws = line.strip().split(":")
            name2keywords[name] = kws.split(",")
    return name2keywords

def load_hierarchy(path):
    parents = collections.defaultdict(set)
    children = collections.defaultdict(set)
    with open(path, "r") as f:
        for line in f:
            p, c = map(int, line.split())
            parents[c].add(p)
            children[p].add(c)
    return parents, children

def get_all_ancestors(label, parents):
    ancestors = set()
    stack = [label]
    while stack:
        node = stack.pop()
        for p in parents.get(node, []):
            if p not in ancestors:
                ancestors.add(p)
                stack.append(p)
    return ancestors
