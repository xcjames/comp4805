import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F
import numpy as np
import gc
import scipy.sparse as sp

class DLightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_mat, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, beta, denoise, cl_crossLayer,cl_crossLayer_weight,  add_noise_to_emb, eps,device):
        super(DLightGCL,self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))#user embedding, d is embedding size
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))#item embedding

        self.train_csr = train_csr #csr matrix for training
        self.adj_mat = adj_mat  # adjacent matrix in sparse matrix format
        self.adj_norm = adj_norm # normalized adj matrix
        self.l = l                # number of GNN layers
        self.E_u_list = [None] * (l+1) # list of user embeddings for all l GNN layers
        self.E_i_list = [None] * (l+1) # list of item embeddings for all l GNN layers
        self.E_u_list[0] = self.E_u_0  # user embeddings for the first GNN layer
        self.E_i_list[0] = self.E_i_0  # item embeddings for the first GNN layer
        self.Z_u_list = [None] * (l+1) 
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1) # list of user embeddings for all l GNN layers from reconstructed graph
        self.G_i_list = [None] * (l+1) # list of item embeddings for all l GNN layers from reconstructed graph
        self.G_u_list[0] = self.E_u_0  # user embeddings for the first GNN layer for reconstructed graph
        self.G_i_list[0] = self.E_i_0  # item embeddings for the first GNN layer for reconstructed graph
        self.temp = temp               #temperature
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.beta = beta
        
        self.denoise = denoise
        self.add_noise_to_emb = add_noise_to_emb
        self.cl_crossLayer = cl_crossLayer
        self.cl_crossLayer_weight = cl_crossLayer_weight
        self.eps = eps
        self.device = device

        self.first_epoch = True

    #Calculating contrastive loss function
    def cal_cl_loss(self, emb1, emb2, ids):
        gnn = nn.functional.normalize(emb1,p=2,dim=1)
        hyper = nn.functional.normalize(emb2,p=2,dim=1)
        
        pos_score = ((gnn[ids] * hyper[ids]).sum(1)/self.temp).mean()
        neg_score = torch.exp(gnn[ids] @ hyper.T / self.temp).sum(1)
        neg_score = torch.log(neg_score + 1e-8).mean()
        loss_s_emb1_emb2 = -pos_score + neg_score
        
        return loss_s_emb1_emb2

    #Originally, the code use for loop to iterate the adjacency list to normalize it
    #Here we use CUDA, Speed up normalizing adjacency matrix
    def normalize_adj_mat(self,A):
        with torch.no_grad():
            row_node_deg = torch.sum(A,dim=1).cuda(torch.device(self.device))
            col_node_deg = torch.sum(A,dim=0).cuda(torch.device(self.device))
            A_updated = torch.clone(A)
            CHUNK_SIZE = 100000
            for i in range(0,A.indices()[0].shape[0],CHUNK_SIZE):
                batch_indices = A.indices()[:, i:i + CHUNK_SIZE]
                row_batch = torch.index_select(row_node_deg, 0, batch_indices[0, :])
                col_batch = torch.index_select(col_node_deg, 0, batch_indices[1, :])
                A_updated.values()[i:i + CHUNK_SIZE] *= torch.pow(row_batch * col_batch,-0.5)
        return A_updated

    def denoising(self, R, user_emb, item_emb, beta):
        # R is the Interaction matrix        
        user_emb_n, item_emb_n = user_emb.norm(dim=1)[:, None], item_emb.norm(dim=1)[:, None]
        user_emb_norm = user_emb / torch.max(user_emb_n, 1e-8 * torch.ones_like(user_emb_n))
        item_emb_norm = item_emb / torch.max(item_emb_n, 1e-8 * torch.ones_like(item_emb_n))
        user_H = R @ item_emb_norm
        item_H = R.T @ user_emb_norm
        # here we don't need to count gradient of R_updated and CHUNK_size, add no_grad() to save computational cost.
        with torch.no_grad():
            R_updated = torch.clone(R)
            CHUNK_SIZE = 100000
        for i in range(0, R.indices()[0].shape[0], CHUNK_SIZE):
            batch_indices = R.indices()[:, i:i + CHUNK_SIZE]
            # a_batch = torch.index_select(user_emb_norm, 0, batch_indices[0, :])
            # b_batch = torch.index_select(item_emb_norm, 0, batch_indices[1, :])
            a_batch = torch.index_select(user_H, 0, batch_indices[0, :])
            b_batch = torch.index_select(item_H, 0, batch_indices[1, :])
            cos_sim = torch.nn.functional.cosine_similarity(a_batch, b_batch, dim=1)
             
            # here we don't need to count gradient of sim, add no_grad() to save computational cost.
            with torch.no_grad():
                sim = torch.div(torch.add(cos_sim,1) , 2)
               
                sim = torch.where(sim > beta, sim, 0)
                
                R_updated.values()[i:i + CHUNK_SIZE] = sim
        
        return R_updated

    def forward(self, uids, iids, pos, neg, test=False):
        
        if test==True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase  
            for layer in range(1,self.l+1):
                # GNN propagation
                # use the denoising to do edge dropout.
                
                if(self.denoise=="True"): 
                    if(self.add_noise_to_emb=="True"):            
                        denoised_adj = self.normalize_adj_mat(self.denoising(self.adj_mat,self.E_u_list[layer-1], self.E_i_list[layer-1], beta=self.beta))
                        self.Z_u_list[layer] = (torch.spmm(denoised_adj, self.E_i_list[layer-1]))
                        self.Z_i_list[layer] = (torch.spmm(denoised_adj.transpose(0,1), self.E_u_list[layer-1]))
                        
                        #Add random noise to User Item Embeddings directly(from SimGCL)
                        Z_i_random_noise = torch.rand_like(self.Z_i_list[layer]).cuda()
                        Z_u_random_noise = torch.rand_like(self.Z_u_list[layer]).cuda()
                        self.Z_i_list[layer].data += torch.sign(self.Z_i_list[layer]) * (torch.nn.functional.normalize(Z_i_random_noise, dim=1) * self.eps)
                        self.Z_u_list[layer].data += torch.sign(self.Z_u_list[layer]) * (torch.nn.functional.normalize(Z_u_random_noise, dim=1) * self.eps)

                        
                    else:#random dropout
                        denoised_adj = self.normalize_adj_mat(self.denoising(self.adj_mat,self.E_u_list[layer-1], self.E_i_list[layer-1], beta=self.beta))  
                        self.Z_u_list[layer] = (torch.spmm(sparse_dropout(denoised_adj,self.dropout), self.E_i_list[layer-1]))
                        self.Z_i_list[layer] = (torch.spmm(sparse_dropout(denoised_adj,self.dropout).transpose(0,1), self.E_u_list[layer-1]))
                        
                else:
                    if(self.add_noise_to_emb=="True"): 
                        
                        self.Z_u_list[layer] = (torch.spmm(self.adj_norm, self.E_i_list[layer-1]))
                        self.Z_i_list[layer] = (torch.spmm(self.adj_norm.transpose(0,1), self.E_u_list[layer-1]))
                        
                        #Add random noise to User Item Embeddings directly(from SimGCL)
                        Z_i_random_noise = torch.rand_like(self.Z_i_list[layer]).cuda()
                        Z_u_random_noise = torch.rand_like(self.Z_u_list[layer]).cuda()
                        self.Z_i_list[layer].data += torch.sign(self.Z_i_list[layer]) * (torch.nn.functional.normalize(Z_i_random_noise, dim=1) * self.eps)
                        self.Z_u_list[layer].data += torch.sign(self.Z_u_list[layer]) * (torch.nn.functional.normalize(Z_u_random_noise, dim=1) * self.eps)

                    else:#dropout
                        self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
                        self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

                # svd_adj propagation
                vt_ei = self.vt @ self.E_i_list[layer-1]
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu)
                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]                       

            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)
            
            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            E_u_avg = torch.div(self.E_u, self.l)
            E_i_avg = torch.div(self.E_i, self.l)

            # G_u_avg = torch.div(self.G_u, self.l)
            # G_i_avg = torch.div(self.G_i, self.l)
            
            # loss_s = 
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i

            #The E_G Contrastive Loss (Embeddings' contrastive loss from 2 graphs, original graph E and reconstructed graph G)
            # print("G_u_norm[uids] ", G_u_norm[uids].shape,"   E_u_norm", E_u_norm.shape,   "   G_i_norm", G_i_norm.shape ,"   E_i_norm", E_i_norm.shape , "G_u_norm[uids]",G_u_norm[uids].shape)
            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
            loss_s = -pos_score + neg_score
            # print("loss_s",loss_s)


            #total cross-layer loss
            if(self.cl_crossLayer == "True"):
                # cross-layer loss in original graph: first layer embedding and all layer average embedding (XSimGCL)
                neg_orig_1st_last = torch.log(torch.exp(self.E_u_list[1][uids] @ E_u_avg.T / self.temp).sum(1)+1e-8).mean()
                neg_orig_1st_last += torch.log(torch.exp(self.E_i_list[1][iids] @ E_i_avg.T / self.temp).sum(1)+1e-8).mean()
                pos_orig_1st_last = (torch.clamp((self.E_u_list[1][uids] * E_u_avg[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((self.E_i_list[1][iids] * E_i_avg[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
                
                loss_orig_1st_last = -pos_orig_1st_last + neg_orig_1st_last
               
                ####################################
                # # cross-layer loss in reconstructed graph: first layer embedding and all layer average embedding (XSimGCL) 
                # # (Not as useful as original graph information, the reason may be that it only selects the first q column of the U S V, which may lose some information)
                
                # neg_recon_1st_last = torch.log(torch.exp((self.G_u_list[1])[uids] @ G_u_avg.T / self.temp).sum(1)+1e-8).mean()
                # neg_recon_1st_last += torch.log(torch.exp((self.G_i_list[1])[iids] @ G_i_avg.T / self.temp).sum(1)+1e-8).mean()
                # pos_recon_1st_last = (torch.clamp(((self.G_u_list[1])[uids] * G_u_avg[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp(((self.G_i_list[1])[iids] * G_i_avg[iids]).sum(1) / self.temp,-5.0,5.0)).mean()

                # loss_recon_1st_last= -pos_recon_1st_last + neg_recon_1st_last
                ####################################
                
                # #################################################
                # # I also tried the cross-layer loss in original graph: first layer embedding and last layer embedding, performance not very good
                # neg_orig_1st_last = torch.log(torch.exp((self.E_u_list[1])[uids] @ self.E_u_list[self.l].T / self.temp).sum(1)+1e-8).mean()
                # neg_orig_1st_last += torch.log(torch.exp((self.E_i_list[1])[iids] @ self.E_i_list[self.l].T / self.temp).sum(1)+1e-8).mean()
                # pos_orig_1st_last = (torch.clamp(((self.E_u_list[1])[uids] * (self.E_u_list[self.l])[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp(((self.E_i_list[1])[iids] * (self.E_i_list[self.l])[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
                
                # loss_orig_1st_last = -pos_orig_1st_last + neg_orig_1st_last
                
                # # cl loss in reconstructed graph first and last layer (XSimGCL)
                # neg_recon_1st_last = torch.log(torch.exp((self.G_u_list[1])[uids] @ self.G_u_list[self.l].T / self.temp).sum(1)+1e-8).mean()
                # neg_recon_1st_last += torch.log(torch.exp((self.G_i_list[1])[iids] @ self.G_i_list[self.l].T / self.temp).sum(1)+1e-8).mean()
                # pos_recon_1st_last = (torch.clamp(((self.G_u_list[1])[uids] * (self.G_u_list[self.l])[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp(((self.G_i_list[1])[iids] * (self.G_i_list[self.l])[iids]).sum(1) / self.temp,-5.0,5.0)).mean()

                # loss_recon_1st_last= -pos_recon_1st_last + neg_recon_1st_last
                # #################################################

                # #Here I tried to compute contrastive loss through function cal_cl_loss
                ##################################
                # loss_orig_1st_last = loss_orig_1st_last + self.cal_cl_loss(self.Z_u_list[1], self.Z_u_list[self.l], uids)
                # loss_orig_1st_last = loss_orig_1st_last + self.cal_cl_loss(self.Z_i_list[1], self.Z_i_list[self.l], iids)

                # loss_recon_1st_last = loss_recon_1st_last + self.cal_cl_loss(self.G_u_list[1], self.G_u_list[self.l], uids)
                # loss_recon_1st_last = loss_recon_1st_last + self.cal_cl_loss(self.G_i_list[1], self.G_i_list[self.l], iids)
                ##################################
            

            if(self.cl_crossLayer == "True"):
                # # Try out different weights for cross-layer contrastive loss and E_G contrastive loss.
                # loss_cl = loss_orig_1st_last + loss_recon_1st_last + 2 * loss_s
                # loss_cl =  loss_orig_1st_last +  loss_recon_1st_last +  loss_s
                # loss_cl = 0.25 * loss_orig_1st_last + 0.25 * loss_recon_1st_last + 0.5 * loss_s
                # loss_cl = 0.5 * loss_orig_1st_last + 0.5 * loss_recon_1st_last
                loss_cl = self.cl_crossLayer_weight * loss_orig_1st_last + (1-self.cl_crossLayer_weight) * loss_s
                
            else:
                loss_cl = loss_s
            
            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()
            # print("loss_r",loss_r)

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2
            

            # My model total loss
            loss = loss_r + self.lambda_1 * loss_cl + loss_reg
                     

            return loss, loss_r, self.lambda_1 * loss_cl
