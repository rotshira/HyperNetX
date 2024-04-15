from collections import OrderedDict

import pytest
import numpy as np
import pandas as pd
from hypernetx.classes.hypergraph import Hypergraph

from networkx.algorithms import bipartite

from hypernetx.classes.property_store import PropertyStore


def test_hypergraph_from_iterable_of_sets(sbs):
    H = Hypergraph(sbs.edges)
    assert len(H.edges) == 6
    assert len(H.nodes) == 7
    assert H.degree("A") == 3
    assert H.number_of_edges() == 6
    assert H.number_of_nodes() == 7


def test_hypergraph_from_dict(sbs):
    H = Hypergraph(sbs.edgedict)
    assert len(H.edges) == 6
    assert len(H.nodes) == 7
    assert H.degree("A") == 3
    assert H.size("R") == 2
    assert H.order() == 7


def test_hypergraph_custom_attributes(sbs):
    H = Hypergraph(sbs.edges)
    assert isinstance(H.__str__(), str)
    assert isinstance(H.__repr__(), str)
    assert H.__contains__("A")
    assert H.__len__() == 7
    nodes = [key for key in H.__iter__()]
    assert sorted(nodes) == ["A", "C", "E", "K", "T1", "T2", "V"]
    assert sorted(H.__getitem__("C")) == ["A", "E", "K"]


def test_get_linegraph(sbs):
    H = Hypergraph(sbs.edges)
    assert len(H.edges) == 6
    assert len(H.nodes) == 7
    assert len(set(H.get_linegraph(s=1)).difference(set([0, 1, 2, 3, 4, 5]))) == 0


def test_hypergraph_from_incidence_dataframe(lesmis):
    df = lesmis.hypergraph.incidence_dataframe()
    H = Hypergraph.from_incidence_dataframe(df)
    assert H.shape == (40, 8)
    assert H.size(3) == 8
    assert H.degree("JA") == 3


def test_hypergraph_from_numpy_array(sbs):
    H = Hypergraph.from_numpy_array(sbs.arr)
    assert len(H.nodes) == 6
    assert len(H.edges) == 7
    assert H.dim("e5") == 2
    assert set(H.neighbors("v2")) == {"v0", "v5"}


def test_hypergraph_from_bipartite(sbsd_hypergraph):
    H = sbsd_hypergraph
    HB = Hypergraph.from_bipartite(H.bipartite())
    assert len(HB.edges) == 7
    assert len(HB.nodes) == 8


def test_add_edge_inplace(sbs):
    h = Hypergraph(sbs.edgedict)
    assert h.shape == (7, 6)

    # add a new edge in place; i.e. the current hypergraph should be mutated
    new_edge = "X"
    h.add_edge(new_edge)

    # the Hypergraph should not increase its number of edges and incidences because the current behavior of adding
    # an edge does not connect two or more nodes.
    # In other words, adding an edge with no nodes
    assert h.shape == (7, 6)
    assert new_edge not in h.edges.elements

    # the new edge has no user-defined property data, so it should not be listed in the PropertyStore
    assert new_edge not in h.edges.properties

    # However, the new_edge will be listed in the complete list of all user and non-user-define properties for all edges
    assert new_edge in h.edges.to_dataframe.index.tolist()

    assert new_edge in h.edges.to_dataframe.index.tolist()


def test_add_edge_not_inplace(sbs):
    h = Hypergraph(sbs.edgedict)
    assert h.shape == (7, 6)

    # add a new edge not in place; the current hypergraph should be diffrent from the new hypergraph
    # created from add_edge
    new_edge = "X"
    new_hg = h.add_edge(new_edge, inplace=False)

    assert new_hg.shape == (7, 6)
    assert new_edge not in new_hg.edges.elements

    assert new_edge not in new_hg.edges.properties
    assert new_edge in new_hg.edges.to_dataframe.index.tolist()

    # verify that the new edge is not in the old HyperGraph
    assert new_edge not in h.edges.to_dataframe.index.tolist()


def test_remove_edges(sbs):
    H = Hypergraph(sbs.edgedict)
    assert H.shape == (7, 6)
    # remove an edge without removing any nodes
    H = H.remove_edges("P")
    assert H.shape == (7, 5)
    # remove an edge containing a singleton ear
    H = H.remove_edges("O")
    assert H.shape == (6, 4)


@pytest.mark.skip("reason=remove has a new signature")
def test_remove(triloop2):
    H = triloop2.hypergraph
    k = "ACD2"
    assert H.shape == (5, 4)
    newH = H.remove(k, level=0)
    assert newH.shape == (5, 3)
    newH = H.remove("E", level=1)
    assert newH.shape == (4, 4)
    newH = H.remove("ACD", level=0)
    assert newH.shape == (5, 3)
    newH = H.remove(["ACD", "E"])
    assert newH.shape == (4, 3)
    # with pytest.raises(TypeError):
    #     H.remove({"ACD": "edge"})


def test_remove_nodes():
    a, b, c, d = "a", "b", "c", "d"
    hbug = Hypergraph({0: [a, b], 1: [a, c], 2: [a, d]})
    assert a in hbug.nodes
    assert a in hbug.edges[0]
    assert a in hbug.edges[1]
    assert a in hbug.edges[2]
    hbug = hbug.remove_nodes(a)
    assert a not in hbug.nodes
    assert a not in hbug.edges[0]
    assert a not in hbug.edges[1]
    assert a not in hbug.edges[2]


def test_matrix(sbs_hypergraph):
    H = sbs_hypergraph
    assert H.incidence_matrix().todense().shape == (7, 6)
    assert H.adjacency_matrix(s=2).todense().shape == (7, 7)
    assert H.edge_adjacency_matrix().todense().shape == (6, 6)
    aux_matrix = H.auxiliary_matrix(node=False)
    assert aux_matrix.todense().shape == (6, 6)


def test_collapse_edges(sbsd_hypergraph):
    H = sbsd_hypergraph
    assert len(H.edges) == 7
    HC = H.collapse_edges()
    assert len(HC.edges) == 6


def test_collapse_nodes(sbsd_hypergraph):
    H = sbsd_hypergraph
    assert len(H.nodes) == 8
    HC = H.collapse_nodes()
    assert len(HC.nodes) == 7


def test_collapse_nodes_and_edges(sbsd_hypergraph):
    H = sbsd_hypergraph
    HC2 = H.collapse_nodes_and_edges()
    assert len(H.edges) == 7
    assert len(HC2.edges) == 6
    assert len(H.nodes) == 8
    assert len(HC2.nodes) == 7


def test_restrict_to_edges(sbs_hypergraph):
    H = sbs_hypergraph
    HS = H.restrict_to_edges(["P", "O"])
    assert len(H.edges) == 6
    assert len(HS.edges) == 2


def test_restrict_to_nodes(sbs_hypergraph):
    H = sbs_hypergraph
    assert len(H.nodes) == 7
    H1 = H.restrict_to_nodes(["A", "E", "K"])
    assert len(H.nodes) == 7
    assert len(H1.nodes) == 3
    assert len(H1.edges) == 5
    assert "C" in H.edges["P"]
    assert "C" not in H1.edges["P"]


# @pytest.mark.skip("reason=Deprecated method")
def test_remove_from_restriction(triloop):
    h = triloop.hypergraph
    h1 = h.restrict_to_nodes(h.neighbors("A")).remove_nodes(
        "A"
    )  # Hypergraph does not have a remove_node method
    assert "A" not in h1
    assert "A" not in h1.edges["ACD"]


def test_toplexes(sbsd_hypergraph):
    H = sbsd_hypergraph
    T = H.toplexes(return_hyp=True)
    assert len(T.nodes) == 8
    assert len(T.edges) == 5
    T = T.collapse_nodes()
    assert len(T.nodes) == 7


def test_is_connected():
    setsystem = [{1, 2, 3, 4}, {3, 4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    assert h.is_connected() is True
    assert h.is_connected(s=2) is False
    assert h.is_connected(s=2, edges=True) is True
    # test case below will raise nx.NetworkXPointlessConcept
    assert h.is_connected(s=3, edges=True) is False


# @pytest.mark.skip("Deprecated methods")
def test_singletons():
    E = {1: {2, 3, 4, 5}, 6: {2, 5, 7, 8, 9}, 10: {11}, 12: {13}, 14: {7}}
    h = Hypergraph(E)
    assert h.shape == (9, 5)
    singles = h.singletons()
    assert len(singles) == 2
    h = h.remove_edges(singles)
    assert h.shape == (7, 3)


def test_remove_singletons():
    E = {1: {2, 3, 4, 5}, 6: {2, 5, 7, 8, 9}, 10: {11}, 12: {13}, 14: {7}}
    h = Hypergraph(E)
    assert h.shape == (9, 5)
    h1 = h.remove_singletons()
    assert h1.shape == (7, 3)
    assert h.shape == (9, 5)


def test_components():
    setsystem = [{1, 2, 3, 4}, {4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    # h.components() causes an error
    assert [len(g) for g in h.component_subgraphs()] == [8]


def test_connected_components():
    setsystem = [{1, 2, 3, 4}, {4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    assert len(list(h.connected_components())) == 1
    assert list(h.connected_components(edges=True)) == [{0, 1, 2, 3}]
    assert [len(g) for g in h.connected_component_subgraphs()] == [8]


def test_s_components():
    setsystem = [{1, 2, 3, 4}, {4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    assert len(list(h.s_components())) == 1
    assert len(list(h.s_components(s=2))) == 2
    assert len(list(h.s_components(s=3))) == 4
    assert len(list(h.s_components(s=3, edges=False))) == 7
    assert len(list(h.s_components(s=4, edges=False))) == 8


def test_s_connected_components():
    setsystem = [{1, 2, 3, 4}, {4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    assert list(h.s_connected_components()) == [{0, 1, 2, 3}]
    assert list(h.s_connected_components(s=2)) == [{1, 2, 3}]
    assert list(h.s_connected_components(s=2, edges=False)) == [{5, 6}]


def test_s_component_subgraphs():
    setsystem = [{1, 2, 3, 4}, {4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    assert {5, 4}.issubset(
        [len(g) for g in h.s_component_subgraphs(s=2, return_singletons=True)]
    )
    assert {3, 4}.issubset(
        [len(g) for g in h.s_component_subgraphs(s=3, return_singletons=True)]
    )


def test_size(sbs):
    h = Hypergraph(sbs.edgedict)
    assert h.size("S") == 4
    assert h.size("S", {"T2", "V"}) == 2
    assert h.size("S", {"T1", "T2"}) == 1
    assert h.size("S", {"T2"}) == 1
    assert h.size("S", {"T1"}) == 0
    assert h.size("S", {}) == 0


def test_diameter(sbs):
    h = Hypergraph(sbs.edgedict)
    assert h.diameter() == 3
    with pytest.raises(Exception) as excinfo:
        h.diameter(s=2)
    assert "Hypergraph is not s-connected." in str(excinfo.value)


def test_node_diameters(sbs):
    h = Hypergraph(sbs.edgedict)
    assert h.node_diameters()[0] == 3
    assert h.node_diameters()[2] == [{"A", "C", "E", "K", "T1", "T2", "V"}]


def test_edge_diameter(sbs):
    h = Hypergraph(sbs.edgedict)
    assert h.edge_diameter() == 3
    assert h.edge_diameters()[2] == [{"I", "L", "O", "P", "R", "S"}]
    with pytest.raises(Exception) as excinfo:
        h.edge_diameter(s=2)
    assert "Hypergraph is not s-connected." in str(excinfo.value)


def test_bipartite(sbs_hypergraph):
    assert bipartite.is_bipartite(sbs_hypergraph.bipartite())


def test_dual(sbs_hypergraph):
    H = sbs_hypergraph
    HD = H.dual()
    assert isinstance(HD.nodes.property_store, PropertyStore)
    assert isinstance(HD.edges.property_store, PropertyStore)
    assert set(H.nodes) == set(HD.edges)
    assert set(H.edges) == set(HD.nodes)
    assert list(H.dataframe.columns) == list(HD.dataframe.columns)


@pytest.mark.filterwarnings("ignore:No 3-path between ME and FN")
def test_distance(lesmis):
    h = lesmis.hypergraph
    assert h.distance("ME", "FN") == 2
    assert h.distance("ME", "FN", s=2) == 3
    assert h.distance("ME", "FN", s=3) == np.inf


def test_edge_distance(lesmis):
    h = lesmis.hypergraph
    assert h.edge_distance(1, 4) == 2
    h2 = h.remove([5], 0)
    assert h2.edge_distance(1, 4) == 3
    assert h2.edge_distance(1, 4, s=2) == np.inf


def test_dataframe(lesmis):
    h = lesmis.hypergraph
    df = h.incidence_dataframe()
    assert np.allclose(np.array(np.sum(df)), np.array([10, 9, 8, 4, 8, 3, 12, 6]))


def test_construct_empty_hypergraph():
    h = Hypergraph()
    assert h.shape == (0, 0)
    assert h.edges.is_empty()
    assert h.nodes.is_empty()
    assert isinstance(h.dataframe, pd.DataFrame)


def test_construct_hypergraph_from_empty_dict():
    h = Hypergraph({})
    assert h.shape == (0, 0)
    assert h.edges.is_empty()
    assert h.nodes.is_empty()


def test_construct_hypergraph_empty_dict():
    h = Hypergraph(dict())
    assert h.shape == (0, 0)
    assert h.edges.is_empty()
    assert h.nodes.is_empty()


def test_static_hypergraph_s_connected_components(lesmis):
    H = Hypergraph(lesmis.edgedict)
    assert {7, 8} in list(H.s_connected_components(edges=True, s=4))


def test_difference_on_same_hypergraph(lesmis):
    hg = Hypergraph(lesmis.edgedict)
    hg_copy = Hypergraph(lesmis.edgedict)

    hg_diff = hg - hg_copy

    assert len(hg_diff) == 0
    assert len(hg_diff.nodes) == 0
    assert len(hg_diff.edges) == 0
    assert hg_diff.shape == (0, 0)
    assert hg_diff.incidence_dict == {}


def test_difference_on_empty_hypergraph(sbs_hypergraph):
    hg_empty = Hypergraph()

    hg_diff = sbs_hypergraph - hg_empty

    assert len(hg_diff) == 7
    assert len(hg_diff.nodes) == 7
    assert len(hg_diff.edges) == 6
    assert hg_diff.shape == (7, 6)

    edges = ["I", "L", "O", "P", "R", "S"]
    assert all(uid in edges for uid in hg_diff.incidence_dict.keys())


def test_difference_on_similar_hypergraph(sbs_hypergraph):
    a, c, e, k, t1, t2, v = ("A", "C", "E", "K", "T1", "T2", "V")
    l, o, p, r, s = ("L", "O", "P", "R", "S")
    data = OrderedDict(
        [(p, {a, c, k}), (r, {a, e}), (s, {a, k, t2, v}), (l, {c, e}), (o, {t1, t2})]
    )
    hg_similar = Hypergraph(data, edge_col="edges", node_col="nodes")

    hg_diff = sbs_hypergraph - hg_similar

    assert len(hg_diff) == 2
    assert len(hg_diff.nodes) == 2
    assert len(hg_diff.edges) == 1
    assert hg_diff.shape == (2, 1)
    print(hg_diff.incidence_dict.keys())
    edges = ["I"]
    assert all(uid in edges for uid in hg_diff.incidence_dict.keys())
