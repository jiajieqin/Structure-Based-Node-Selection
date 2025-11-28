import networkx as nx
import math
import pandas as pd
import numpy as np

'''
介数中心性
'''


def regular_betweeness_centrality(graph: nx.Graph):
    betweeness_dic = {}
    betweeness_sum = 0
    # 做一下归一化
    print("regular_betweeness_centrality")
    betweeness_list = nx.betweenness_centrality(graph)
    for i in betweeness_list:
        betweeness_sum = betweeness_sum + betweeness_list[i]
        # print(i)
    for i in betweeness_list:
        betweeness_dic[i] = round(betweeness_list[i] / betweeness_sum, 6)
    return betweeness_dic


'''
coreness
'''


def regular_coreness(graph: nx.Graph):
    coreness_dic = {}
    coreness_sum = 0
    # 做一下归一化
    print("regular_coreness")
    coreness_list = nx.core_number(graph)
    for i in list(coreness_list.keys()):
        coreness_sum = coreness_sum + coreness_list[i]
    for i in list(coreness_list.keys()):
        coreness_dic[i] = round(coreness_list[i] / coreness_sum, 6)

    return coreness_dic


'''
特征向量中心性
'''


def regular_eigenector_centrality(graph: nx.Graph):
    eigenector_dic = {}
    eigenector_sum = 0
    # 做一下归一化
    print("regular_eigenector_centrality")
    eigenector_centrality_list = nx.eigenvector_centrality(graph, max_iter=1000)
    for i in list(eigenector_centrality_list.keys()):
        eigenector_sum = eigenector_sum + eigenector_centrality_list[i]
    for i in list(eigenector_centrality_list.keys()):
        eigenector_dic[i] = round(eigenector_centrality_list[i] / eigenector_sum, 12)
    return eigenector_dic


'''
pagerank
'''


def regular_page_rank(graph: nx.Graph):
    return nx.pagerank(graph)

def regular_katz_centrality(graph: nx.Graph):
    A = nx.to_numpy_array(graph)
    lambda_max = max(np.linalg.eigvals(A).real)  # 无向非负时，最大特征值是实数
    return nx.katz_centrality(graph, 0.9/lambda_max, 0.5)

def regular_closeness_centrality(graph: nx.Graph):
    return nx.closeness_centrality(graph)


