import random
import time
import numpy as np
import torch 
import torch.optim as optim
import argparse
from utils.bert import get_config, BertModel, BertForMaskedLM, show_base_PCA
from module import Train_Module
from dataload import DATA, MyDataset 
import datetime
from sklearn.metrics.cluster import adjusted_rand_score
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score
from sklearn.cluster import SpectralClustering 


random.seed(10)
torch.manual_seed(1234)
np.random.seed(1234)

parser = argparse.ArgumentParser(description='RNABERT')
parser.add_argument('--mag',  type=int, default=1,
                    help='enumerate')
parser.add_argument('--epoch', '-e', type=int, default=200,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--batch', '-b', type=int, default=20,
                    help='Number of batch size')
parser.add_argument('--maskrate', '-m', type=float, default=0.0,
                    help='mask rate')
parser.add_argument('--pretraining', '-pre', type=str, help='use pretrained weight')
parser.add_argument('--outputweight', type=str, help='output path for weights')
parser.add_argument('--algorithm', type=str, default="global", help='algorithm method')
# 通过指定不同的数据集来激活不同的任务
parser.add_argument('--data_mlm', '-d', type=str, nargs='*', help='data for mlm training')
parser.add_argument('--data_mul', type=str, nargs='*', help='data for mul training')
parser.add_argument('--data_alignment', type=str, nargs='*', help='data for alignment test')
parser.add_argument('--data_clustering', type=str, nargs='*', help='data for clustering test')
parser.add_argument('--data_showbase', type=str, nargs='*', help='data for base embedding')
parser.add_argument('--data_embedding', type=str, nargs='*', help='data for base embedding')
parser.add_argument('--embedding_output', type=str, nargs='*', help='output file for base embedding')
parser.add_argument('--show_aln', action='store_true')

args = parser.parse_args()
batch_size = args.batch
current_time = datetime.datetime.now()

print("start...")
class TRAIN:
    """The class for controlling the training process of SFP"""
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.module = Train_Module(config)
    
    def model_device(self, model):
        # Move model to device
        print("device: ", self.device)
        print('-----start-------')
        model.to(self.device)
        if self.device == 'cuda':
            model = torch.nn.DataParallel(model) # make parallel
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return model

    def train_MLM_SFP(self, model, optimizer, dl_MLM_SFP, num_epochs, task_type):
        # dl_xxx 是一个dataloader
        # task_type目前有效的为MLM和MUL
        for epoch in range(num_epochs):
            model.train()
            epoch_mlm_loss = 0.0
            epoch_ssl_loss = 0.0
            epoch_mlm_correct = 0.0
            epoch_ssl_correct = 0.0
            epoch_sfp_loss=0.0
            epoch_sfp_correct = 0.0
            epoch_mul_loss = 0.0

            iteration = 1
            t_epoch_start = time.time()
            t_iter_start = time.time()
            data_num = 0
            for batch in dl_MLM_SFP:
                optimizer.zero_grad()
                if task_type == "MLM" or task_type == "SFP":
                    # 每个batch返回一系列seq pair
                    low_seq_0, masked_seq_0, family_0, seq_len_0, low_seq_1, masked_seq_1, family_1, seq_len_1 = batch
                    # seq_0 and 1 30x440, fa_0 and 1 all 0, seq neq masked seq
                elif task_type == "MUL":
                    # 多了两列信息common index for 0 and 1
                    low_seq_0, masked_seq_0, family_0, seq_len_0, low_seq_1, masked_seq_1, family_1, seq_len_1, common_index_0, common_index_1 = batch

                masked_seq_0 = masked_seq_0.to(self.device)
                low_seq_0 = low_seq_0.to(self.device)
                masked_seq_1 = masked_seq_1.to(self.device)
                low_seq_1 = low_seq_1.to(self.device)

                masked_seq = torch.cat((masked_seq_0, masked_seq_1), axis=0)  # 60,440
                prediction_scores, prediction_scores_ss, encoded_layers =  model(masked_seq)  # 送入模型预测打分
                prediction_scores0, prediction_scores1 = torch.split(prediction_scores, int(prediction_scores.shape[0]/2))
                # (torch.Size([60, 440, 6]), torch.Size([60, 440, 8]), torch.Size([60, 440, 120]))
                prediction_scores_ss0, prediction_scores_ss1 = torch.split(prediction_scores_ss, int(prediction_scores_ss.shape[0]/2))
                encoded_layers0, encoded_layers1 = torch.split(encoded_layers, int(encoded_layers.shape[0]/2))

                loss = 0
                # MLM LOSS
                mlm_loss_0, mlm_correct_0 = self.module.train_MLM(low_seq_0, masked_seq_0, prediction_scores0)
                mlm_loss_1, mlm_correct_1 = self.module.train_MLM(low_seq_1, masked_seq_1, prediction_scores1)
                mlm_loss = (mlm_loss_0 + mlm_loss_1)/2  # 两部分seq 预测loss的平均
                mlm_loss = torch.tensor(0.0) if  torch.isnan(mlm_loss) else mlm_loss 
                mlm_correct = (mlm_correct_0 + mlm_correct_1)/2
                epoch_mlm_loss += mlm_loss.item() * batch_size  # 为什么要乘batch_size
                epoch_mlm_correct += mlm_correct
                if task_type == "MLM":    
                    loss += mlm_loss

                # SFP LOSS
                if task_type == "SFP":   # 这是什么类型任务，看一下module.train_xxx系列 
                    # z0, z1是否embedding？
                    z0_list, z1_list =  self.module.em(encoded_layers0, seq_len_0), self.module.em(encoded_layers1, seq_len_1)
                    sfp_loss, sfp_correct = self.module.train_SFP(low_seq_0, seq_len_0, low_seq_1, seq_len_1, family_0, family_1, z0_list, z1_list)
                    sfp_loss = torch.tensor(0.0) if  torch.isnan(sfp_loss) else sfp_loss 
                    epoch_sfp_loss += sfp_loss.item() * batch_size
                    epoch_sfp_correct += sfp_correct
                    loss += sfp_loss

                # MULTIPLE LOSS
                if task_type == "MUL":
                    common_index_0 = common_index_0.to(self.device)
                    common_index_1 = common_index_1.to(self.device)
                    z0_list, z1_list =  self.module.em(encoded_layers0, seq_len_0), self.module.em(encoded_layers1, seq_len_1)
                    mul_loss = self.module.train_MUL(z0_list, z1_list, common_index_0, common_index_1, seq_len_0, seq_len_1)
                    mul_loss = torch.tensor(0.0) if  torch.isnan(mul_loss) else mul_loss 
                    epoch_mul_loss += mul_loss.item()
                    loss +=  mul_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            t_epoch_finish = time.time()
            epoch_mlm_loss = epoch_mlm_loss / len(dl_MLM_SFP.dataset)  # 30
            epoch_mlm_correct = epoch_mlm_correct / len(dl_MLM_SFP)  # 1
            epoch_sfp_loss = epoch_sfp_loss  / len(dl_MLM_SFP.dataset)
            epoch_sfp_correct = epoch_sfp_correct / len(dl_MLM_SFP.dataset)
            epoch_mul_loss = epoch_mul_loss
            print('Epoch {}/{} | MLM Loss: {:.4f} MLM Acc: {:.4f}| SFP Loss: {:.4f} SFP Acc: {:.4f}| MUL Loss: {:.4f}| time: {:.4f} sec.'.format(epoch+1, num_epochs,
                                                                        epoch_mlm_loss, epoch_mlm_correct, epoch_sfp_loss, epoch_sfp_correct, epoch_mul_loss, time.time() - t_epoch_start))
            t_epoch_start = time.time()
        if args.outputweight:  # 模型输出
            torch.save(model.state_dict(), args.outputweight + '{0:%m_%d_%H_%M}'.format(current_time))  # checkpoints
            torch.save(model.state_dict(), args.outputweight)
        return model

    # make feature vector 
    def make_feature(self, model, dataloader, seqs):
        # 模型推理一系列seqs，转embedding返回
        model.eval()
        torch.backends.cudnn.benchmark = True
        batch_size = dataloader.batch_size
        encoding = []
        for batch in dataloader:
            data, label, seq_len= batch  # data 36x440 padding 0, label 36 all zero, seq_len 36 all 19
            inputs = data.to(self.device)
            prediction_scores, prediction_scores_ss, encoded_layers =  model(inputs)  # BertForMaskedLM.forward
            # p_s 36x440x6, p_s_ss 36x440x8, e_l 36x440x120, e_l是需要的encoding
            encoding.append(encoded_layers.cpu().detach().numpy())
        encoding = np.concatenate(encoding, 0)  # 126x440x120

        embedding = []
        for e, seq in zip(encoding, seqs):
            embedding.append(e[:len(seq)].tolist())  # 只截取需要长度，nx120，padding部分emb不需要

        return embedding 

    def validateOnCompleteTestData(self, test_loader, simirality_matrix):
        # 测试数据上的完整验证
        # accuracy and rand index
        nmi = normalized_mutual_info_score
        ari = adjusted_rand_score
        homo = homogeneity_score
        com = completeness_score 
        true_labels = np.concatenate([d[1].cpu().numpy() for i,d in enumerate(test_loader)], 0)
        
        # km = KMeans(n_clusters=len(np.unique(true_labels)), n_init=20, n_jobs=4)
        # y_pred = km.fit_predict(simirality_matrix)

        # ac = AgglomerativeClustering(n_clusters=len(np.unique(true_labels)), affinity='precomputed', linkage='average')
        # ac = AgglomerativeClustering(n_clusters=None,affinity='precomputed', linkage='average', distance_threshold=0.45)
        # y_pred = ac.fit_predict(1+ (-1 * simirality_matrix))

        # y_pred = y_pred.tolist()
        # true_labels = true_labels.tolist()
        # import collections
        # c = collections.Counter(y_pred)
        # y_pred_new = []
        # true_labels_new = []
        # for i, j in zip(y_pred, true_labels):
        #     if c[i] >= 2:
        #         y_pred_new.append(i)
        #         true_labels_new.append(j)
        # print(len(y_pred_new))
        # y_pred = np.array(y_pred_new)
        # true_labels = np.array(true_labels_new)

        sc=SpectralClustering(n_clusters=len(np.unique(true_labels)))  # 先降维再聚类，效率高
        y_pred=sc.fit(simirality_matrix).labels_

        print(' '*8 + '|==>  nmi: %.4f ,  ari: %.4f,  com: %.4f,  homo: %.4f  <==|'
                      % (nmi(true_labels, y_pred), ari(true_labels, y_pred), com(true_labels, y_pred), homo(true_labels, y_pred)))
        return ari(true_labels, y_pred)

    def align(self, model, dl):
        # 输入的seq_0, seq_1组合后做embedding，再test_align
        model.eval()
        pred_match = 0
        ref_match = 0
        TP = 0
        for batch in dl:
            low_seq_0, masked_seq_0, family_0, seq_len_0, low_seq_1, masked_seq_1, family_1, seq_len_1, common_index_0, common_index_1 = batch
            low_seq_0 = low_seq_0.to(self.device)
            low_seq_1 = low_seq_1.to(self.device)
            low_seq = torch.cat((low_seq_0, low_seq_1), axis=0)
            
            start = time.time()
            prediction_scores, prediction_scores_ss, encoded_layers =  model(low_seq)
            elapsed_time = time.time() - start
            # print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

            prediction_scores0, prediction_scores1 = torch.split(prediction_scores, int(prediction_scores.shape[0]/2))
            encoded_layers0, encoded_layers1 = torch.split(encoded_layers, int(encoded_layers.shape[0]/2))
            z0_list, z1_list =  self.module.em(encoded_layers0, seq_len_0), self.module.em(encoded_layers1, seq_len_1)
            len_TP, len_pred_match, len_ref_match = self.module.test_align(low_seq_0, low_seq_1, z0_list, z1_list, common_index_0, common_index_1, seq_len_0, seq_len_1, args.show_aln)
            TP += len_TP
            pred_match += len_pred_match
            ref_match += len_ref_match

        PPV = TP /pred_match 
        sens = TP / ref_match
        f1 = 2 * PPV * sens/(PPV + sens)
        if args.show_aln == False:
            print("alignment accuracy : ", f1, "sens : ", sens, "PPV : ", PPV)
        return f1 

    def test(self, ds, test_loader, model):
        model.eval()
        data_num = len(test_loader.dataset)
        simirality_matrix = []
        for i in range(data_num):
            single_seq = MyDataset("CLU", np.tile(ds.low_seq[i],(data_num,1)), np.tile(ds.low_seq[i],(data_num,1)),np.tile(ds.family[i],(data_num,1)), np.tile(ds.seq_len[i], data_num)) 
            single_seq = torch.utils.data.DataLoader(single_seq, batch_size, shuffle=False)
            low = []
            for data0, data1 in zip( test_loader, single_seq):
                x0, label0, seq_len_0 = data0
                x1, label1, seq_len_1 = data1
                x0, label0 = x0.to("cuda"),label0.to("cuda"),
                x1, label1 = x1.to("cuda"),label1.to("cuda"),
                x = torch.cat((x0, x1), axis=0) 
                prediction_scores, prediction_scores_ss, encoded_layers =  model(x)
                encoded_layers0, encoded_layers1 = torch.split(encoded_layers, int(encoded_layers.shape[0]/2))
                z0_list, z1_list =  self.module.em(encoded_layers0, seq_len_0), self.module.em(encoded_layers1, seq_len_1)
                _, logits = self.module.match(z0_list, z1_list)
                low.append(torch.squeeze(logits).to('cpu').detach().numpy().copy())
            simirality_matrix.append(np.concatenate(low, 0))
        currentAcc = self.validateOnCompleteTestData(test_loader, np.array(simirality_matrix))
        return currentAcc

def objective():
    config.hidden_size = config.num_attention_heads * config.multiple    
    train = TRAIN(config)
    model = BertModel(config)
    model = BertForMaskedLM(config, model)
    if args.data_mlm:
        config.adam_lr = 2e-4
    # if args.data_sfp:
    #     model = fix_params(model)
    #     config.adam_lr = config.adam_lr * 0.5
    if args.data_mul:
        # model = fix_params(model)
        config.adam_lr = 1e-4
    model = train.model_device(model)
    if args.pretraining:
        model.load_state_dict(torch.load(args.pretraining))
    optimizer = optim.AdamW([{'params': model.parameters(), 'lr': config.adam_lr}])
    return model , optimizer, train, config

config = get_config(file_path = "./RNA_bert_config.json")  # BERT模型参数
data = DATA(args, config)
model, optimizer, train, config = objective()

#now start training
if args.data_mlm:
    dl_MLM = data.load_data_MLM_SFP(args.data_mlm)
    model = train.train_MLM_SFP(model, optimizer, dl_MLM, args.epoch, "MLM")
# elif args.data_sfp:  # 这是一个废弃的结构对齐任务？
#     dl_SFP = data.load_data_MLM_SFP(args.data_sfp)
#     model = train.train_MLM_SFP(model, optimizer, dl_SFP, args.epoch, "SFP")
if args.data_mul:
    dl_MUL = data.load_data_MUL(args.data_mul, "MUL")
    model = train.train_MLM_SFP(model, optimizer, dl_MUL, args.epoch, "MUL")

if args.data_alignment:  # alignment推理任务？
    dl_alignment = data.load_data_MUL(args.data_alignment, "MUL")
    alignment_accuracy = train.align(model, dl_alignment)
elif args.data_clustering:  # 聚类任务
    _, _, ds, test_dl = data.load_data_CLU(args.data_clustering) 
    train.test(ds, test_dl, model)  # 这个入口名字……

if args.data_showbase:  # 需要用PCA展示的数据集？
    seqs, label, SS,  ds, test_dl  = data.load_data_SHOW(args.data_showbase) 
    features = train.make_feature(model, test_dl)
    features = features.reshape(-1, features.shape[2])
    show_base_PCA(features, label.reshape(-1), SS)

if args.data_embedding:  # 抽取和生成embedding任务
    seqs, label, test_dl  = data.load_data_EMB(args.data_embedding)  # load data to do embedding
    # label是数字化的seq
    features = train.make_feature(model, test_dl, seqs)  # list of list of list, which is 126x19x120
    for i, data_set in enumerate(args.embedding_output):
        with open(data_set, 'w') as f:
            for d in features:
                f.write(str(d) + '\n')