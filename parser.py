import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--inter_batch', default=4096, type=int, help='batch size')
    # parser.add_argument('--inter_batch', default=2048, type=int, help='batch size')
    
    parser.add_argument('--note', default=None, type=str, help='note')
    parser.add_argument('--lambda1', default=0.2, type=float, help='weight of cl loss')
    parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
    parser.add_argument('--d', default=64, type=int, help='embedding size')
    # parser.add_argument('--d', default=32, type=int, help='embedding size')
    
    parser.add_argument('--q', default=5, type=int, help='rank')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--dropout', default=0.0, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
    parser.add_argument('--lambda2', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')
    parser.add_argument('--denoise', default="True", type=str, help='False: to use normalized adjacency matrix; True: to use denoised adjacency matrix')
    parser.add_argument('--add_noise_to_emb', default="True", type=str, help='False: random dropout; True: add random noise to embedding')
    parser.add_argument('--cl_crossLayer', default="True", type=str, help='False: without Cross-Layer loss, use ; True: add Cross-Layer loss to the contrastive learning loss')
    parser.add_argument('--beta', default=0.1, type=float, help='threshold for denoising the graph, if beta is larger, more "low-user-item-similarity edges" will be dropped')
    parser.add_argument('--cl_crossLayer_weight', default=0.2, type=float, help='should be a float number between 0 and 1, indicates the weight of cross_layer contrastive loss.')
    parser.add_argument('--eps', default=0.1, type=float, help='should be a float number in adding random noise to embedding')
    
    # parser.add_argument('--beta', default=0.1, type=float, help='beta for denoising the graph, if beta is larger, more "low-user-item-similarity edges" will be dropped')
    
    return parser.parse_args()
args = parse_args()
