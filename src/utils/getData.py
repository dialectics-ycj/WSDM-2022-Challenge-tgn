import os

# set cwd
os.chdir(os.getcwd()+"/../data")
# train set
if not os.path.exists("raw_A_train_edges.csv"):
    # A
    # {ml/raw}_{data_name}_{data_type}_edges.csv
    # {ml/raw}_{data_name}_edge_feats.csv
    # {ml/raw}_{data_name}_node_feats.csv
    os.system('wget -O raw_A_train_edges.csv.gz https://data.dgl.ai/dataset/WSDMCup2022/edges_train_A.csv.gz')
    os.system('wget -O raw_A_node_feats.csv.gz https://data.dgl.ai/dataset/WSDMCup2022/node_features.csv.gz')
    os.system(
        'wget -O raw_A_edge_feats.csv.gz https://data.dgl.ai/dataset/WSDMCup2022/edge_type_features.csv.gz')
    # B
    os.system('wget -O raw_B_train_edges.csv.gz https://data.dgl.ai/dataset/WSDMCup2022/edges_train_B.csv.gz')
    os.system('gzip -d *.gz')
# test set
if not os.path.exists("raw_A_test_edges.csv"):
    os.system(
        'wget -O raw_A_test_edges.csv.gz https://data.dgl.ai/dataset/WSDMCup2022/input_A_initial.csv.gz')
    os.system(
        'wget -O raw_B_test_edges.csv.gz https://data.dgl.ai/dataset/WSDMCup2022/input_B_initial.csv.gz')
    os.system('gzip -d *.gz')
# predict set
if not os.path.exists("raw_A_predict_edges.csv"):
    os.system('wget -O raw_A_predict_edges.csv.gz https://data.dgl.ai/dataset/WSDMCup2022/intermediate/input_A.csv.gz')
    os.system('wget -O raw_B_predict_edges.csv.gz https://data.dgl.ai/dataset/WSDMCup2022/intermediate/input_B.csv.gz')
    os.system('gzip -d *.gz')
# final predict set
if not os.path.exists("raw_A_final_predict_edges.csv"):
    os.system('wget -O raw_A_final_predict_edges.csv.gz https://data.dgl.ai/dataset/WSDMCup2022/final/input_A.csv.gz')
    os.system('wget -O raw_B_final_predict_edges.csv.gz https://data.dgl.ai/dataset/WSDMCup2022/final/input_B.csv.gz')
    os.system('gzip -d *.gz')
