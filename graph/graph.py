from py2neo import Graph as NeoGraph, Node, Relationship


class Graph:
    def __init__(self, addr):
        self.graph = NeoGraph(addr)

    def add_node(self, username, gender):
        node = Node("Person", name=username, gender=gender)
        tx = self.graph.begin()
        tx.merge(node, primary_label='Person', primary_key=('name'))
        tx.commit()

    def add_edge(self, username_a, gender_a, username_b, gender_b):
        node_a = Node("Person", name=username_a, gender=gender_a)
        node_b = Node("Person", name=username_b, gender=gender_b)
        rel = Relationship(node_a, "FOLLOW", node_b)
        tx = self.graph.begin()
        tx.merge(rel, primary_label='Person', primary_key=('name'))
        tx.commit()
