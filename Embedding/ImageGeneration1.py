import networkx as nx
import numpy as np
import argparse
import os
import pickle
import glob
from multiprocessing import Pool
from functools import partial
# 导入CodeEmbedder类
from embedding import CodeEmbedder

def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some dot_files',default="./")
    parser.add_argument('-o', '--out', help='The path of output.', required=False,default="./")
    args = parser.parse_args()
    return args

def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph

# 初始化CodeEmbedder的实例
code_embedder = CodeEmbedder()

def image_generation(dot):
    try:
        pdg = graph_extraction(dot)
        labels_dict = nx.get_node_attributes(pdg, 'label')
        labels_code = dict()
        for label, all_code in labels_dict.items():
            code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
            code = code.replace("static void", "void")
            labels_code[label] = code

        degree_cen_dict = nx.degree_centrality(pdg)
        closeness_cen_dict = nx.closeness_centrality(pdg)
        G = nx.DiGraph()
        G.add_nodes_from(pdg.nodes())
        G.add_edges_from(pdg.edges())
        katz_cen_dict = nx.katz_centrality(G)

        degree_channel = []
        closeness_channel = []
        katz_channel = []
        for label, code in labels_code.items():
            # 使用CodeEmbedder实例获取代码嵌入
            line_vec = code_embedder.get_code_embedding(code)

            degree_cen = degree_cen_dict[label]
            degree_channel.append(degree_cen * line_vec)

            closeness_cen = closeness_cen_dict[label]
            closeness_channel.append(closeness_cen * line_vec)

            katz_cen = katz_cen_dict[label]
            katz_channel.append(katz_cen * line_vec)

        return (degree_channel, closeness_channel, katz_channel)
    except Exception as e:
        print(e)
        return None

def write_to_pkl(dot, out, existing_files):
    dot_name = dot.split('/')[-1].split('.dot')[0]
    if dot_name in existing_files:
        return None
    else:
        print(dot_name)
        channels = image_generation(dot)
        print(channels)
        if channels == None:
            return None
        else:
            (degree_channel, closeness_channel, katz_channel) = channels
            out_pkl = out + dot_name + '.pkl'
            data = [degree_channel, closeness_channel, katz_channel]
            with open(out_pkl, 'wb') as f:
                pickle.dump(data, f)

def main():
    args = parse_options()
    dir_name = args.input
    out_path = args.out

    if dir_name[-1] == '/':
        dir_name = dir_name
    else:
        dir_name += "/"
    dotfiles = glob.glob(dir_name + '*.dot')

    if out_path[-1] == '/':
        out_path = out_path
    else:
        out_path += '/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    existing_files = glob.glob(out_path + "/*.pkl")
    existing_files = [f.split('.pkl')[0] for f in existing_files]

    pool = Pool(5)
    pool.map(partial(write_to_pkl, out=out_path, existing_files=existing_files), dotfiles)




if __name__ == '__main__':
    main()
    # path = "./data/real_data"
    # save_path = "./data/outputs"
    # dataset_name = os.listdir(path)
    # for dataset in dataset_name:
    #     pathname = path + "/" + dataset
    #     for type_name in os.listdir(pathname):
    #         full_path = pathname + "/" + type_name
    #         save_dir = save_path + "/" + dataset + "/" + type_name
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         main(full_path, save_dir)

    # pathname ="./pdgs"
    # save_path = "./data/outputs"
    # for type_name in os.listdir(pathname):
    #     full_path = pathname + "/" + type_name
    #     save_dir = save_path + "/sard-2/" + type_name
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     main(full_path, save_dir)

