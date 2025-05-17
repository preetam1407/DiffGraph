import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import torch.nn.functional as F  
from params import args
from Model import HGDM, MetaPathLearner
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax
from DataHandler import DataHandler,index_generator
import numpy as np
import pickle
from Utils.Utils import *
from Utils.Utils import contrast
import os
import logging
import datetime
import sys
# from Model import FastGTNWithDiffusion
import warnings
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt



warnings.filterwarnings("ignore")
 

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')




def print_metapaths(mp: MetaPathLearner):
    """
    Given a trained MetaPathLearner, softmax its two weight matrices
    and print the top-2 relations in each hop for each channel.
    """
    # must match the order you built raw_mats in DataHandler.load_dblp_data()
    relation_names = ['ap','pa','cp','pc','pt','tp']

    # conv1.weight, conv2.weight are both (C, E)
    alpha1 = F.softmax(mp.conv1.weight, dim=1).detach().cpu().numpy()  # (C, E)
    alpha2 = F.softmax(mp.conv2.weight, dim=1).detach().cpu().numpy()  # (C, E)
    C, E = alpha1.shape

    print("\n=== Learned Two-hop Meta-paths ===")
    for c in range(C):
        # get indices of top-2 for each hop
        top1_idx = np.argsort(alpha1[c])[::-1][:2]
        top2_idx = np.argsort(alpha2[c])[::-1][:2]

        hops1 = [(relation_names[i], round(alpha1[c,i], 3)) for i in top1_idx]
        hops2 = [(relation_names[i], round(alpha2[c,i], 3)) for i in top2_idx]

        print(f"Channel {c}:")
        print(f"  1st hop: {hops1}")
        print(f"  2nd hop: {hops2}")
    print("=================================\n")

class Coach:
    def __init__(self, handler):
        self.handler = handler
       
        # self.metrics = dict()
        # mets = ['bceLoss','AUC']
        # for met in mets:
        #     self.metrics['Train' + met] = list()
        #     self.metrics['Test' + met] = list()
                # for plotting
        self.train_bce_losses = []
        self.train_diff_losses = []
        # existing metric containers
        self.metrics = dict()
        for met in ['bceLoss','AUC']:
            self.metrics['Train' + met] = []
            self.metrics['Test'  + met] = []

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret
    
    def log_metapath_stats(self, tag):
        mp = self.handler.metapath_learner
        α1 = F.softmax(mp.conv1.weight, dim=1)
        α2 = F.softmax(mp.conv2.weight, dim=1)
        # e.g. l1 norms per channel
        norms1 = α1.abs().sum(dim=1).detach().cpu().numpy()
        norms2 = α2.abs().sum(dim=1).detach().cpu().numpy()
        print(f"[{tag}] MetaPath α1 ‖·‖₁ = {np.round(norms1,3)}")
        print(f"[{tag}] MetaPath α2 ‖·‖₁ = {np.round(norms2,3)}")


    def run(self):
        # for ratio in range(len(self.handler.train_idx)):
        self.train_bce_losses.clear()
        self.train_diff_losses.clear()
        for ratio in range(len(self.handler.train_idx)):
            log('Ratio Type: '+str(ratio))
            accs = []
            micro_f1s = []
            macro_f1s = []
            macro_f1s_val = []
            auc_score_list = []
            for repeat in range(20):
                self.prepareModel()
                log('Repeat: '+str(repeat))

                macroMax = 0
                


                log_format = '%(asctime)s %(message)s'
                logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
                log_save = './History/'
                log_file = f'{args.data}_' + \
                                    f'lr_{args.lr}_batch_{args.batch}_noise_scale_{args.noise_scale}_step_{args.steps}_ratio_{ratio}_public'
                fname = f'{log_file}.txt'
                fh = logging.FileHandler(os.path.join(log_save, fname))
                fh.setFormatter(logging.Formatter(log_format))
                logger = logging.getLogger()
                logger.addHandler(fh)
                # logger.info(args)
                # logger.info('================')  
                args.save_path = log_file 

                val_accs = []
                val_micro_f1s = []
                val_macro_f1s = []
                test_accs = []
                test_micro_f1s = []
                test_macro_f1s = []
                logits_list = []
                test_lbls = t.argmax(self.label[self.test_idx[ratio]], dim=-1)
                # eval_every = 400
                for ep in range(args.epoch):
                    tstFlag = (ep % 1 == 0)
                    # reses = self.trainEpoch(ratio)
                    # log(self.makePrint('Train', ep, reses, tstFlag))
                    reses = self.trainEpoch(ratio)
                # record for plotting
                    self.train_bce_losses.append(reses['bceLoss'])
                    self.train_diff_losses.append(reses['diffLoss'])
                    if tstFlag:
                        val_reses,test_reses = self.testEpoch(ratio)
                        val_accs.append(val_reses['acc'].item())
                        val_macro_f1s.append(val_reses['macro'])
                        val_micro_f1s.append(val_reses['micro'])

                        test_accs.append(test_reses['acc'].item())
                        test_macro_f1s.append(test_reses['macro'])
                        test_micro_f1s.append(test_reses['micro'])
                        logits_list.append(test_reses['logits'])
                        # print("\t[Val_Classification] Macro-F1_epoch: {:.4f} Micro-F1_epoch: {:.4f} Test-acc_epoch: {:.4f}"
                        #     .format(val_reses['macro'],
                        # val_reses['micro'],
                        # val_reses['acc']
                        #     )
                        #     )


                        # print("\t[Test_Classification] Macro-F1_epoch: {:.4f} Micro-F1_epoch: {:.4f} Test-acc_epoch: {:.4f}"
                        #     .format(test_reses['macro'],
                        # test_reses['micro'],
                        # test_reses['acc']
                        #     )
                        #     )
                    
                    
                    
                        # log(self.makePrint('Test', ep, reses, tstFlag))
                        # if (val_reses['macro'] > macroMax):
                        #     macroMax = test_reses['macro']
                        #     self.saveModel()
                        # logger.info(self.makePrint('Test', ep, test_reses, tstFlag))
                        # self.saveHistory()
                   

                        # self.saveHistory()
                        
                        
                        

                max_iter = test_accs.index(max(test_accs))
                accs.append(test_accs[max_iter])
                max_iter = test_macro_f1s.index(max(test_macro_f1s))
                macro_f1s.append(test_macro_f1s[max_iter])
                macro_f1s_val.append(val_macro_f1s[max_iter])

                max_iter = test_micro_f1s.index(max(test_micro_f1s))
                micro_f1s.append(test_micro_f1s[max_iter])

                best_logits = logits_list[max_iter]
                best_proba = softmax(best_logits, dim=1)
                auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                                    y_score=best_proba.detach().cpu().numpy(),
                                                    multi_class='ovr'
                                                    ))
                
                # print("\t[Test_Classification] Macro-F1_one_time: {:.4f} Micro-F1_one_time: {:.4f} Test-AUC_one_time: {:.4f}"
                #             .format(macro_f1s[-1],
                #         micro_f1s[-1],
                #         auc_score_list[-1]
                #             )
                #             )
                

                logger.info("\t[Classification] Macro-F1: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
                    .format(np.mean(macro_f1s),
                            np.std(macro_f1s),
                            np.mean(micro_f1s),
                            np.std(micro_f1s),
                            np.mean(auc_score_list),
                            np.std(auc_score_list)))
                
                plt.figure()
                plt.plot(self.train_bce_losses, label='Train BCE Loss')
                plt.plot(self.train_diff_losses, label='Train Diffusion Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss Curves')
                plt.legend()
                plt.tight_layout()
                plt.savefig('loss_curve.png', dpi=200)
                plt.show()
    # def run(self):
    #     for ratio in range(len(self.handler.train_idx)):
    #         log('Ratio Type: '+str(ratio))
    #         accs = []
    #         micro_f1s = []
    #         macro_f1s = []
    #         macro_f1s_val = []
    #         auc_score_list = []
    #         for repeat in range(20):
    #             self.log_metapath_stats("Init")
    #             self.prepareModel()
    #             log('Repeat: '+str(repeat))

    #             macroMax = 0

    #             # set up file‐logging
    #             log_format = '%(asctime)s %(message)s'
    #             logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    #                                 format=log_format, datefmt='%m/%d %I:%M:%S %p')
    #             log_save = './History/'
    #             log_file = (f'{args.data}_lr_{args.lr}_batch_{args.batch}_'
    #                         f'noise_scale_{args.noise_scale}_step_{args.steps}_'
    #                         f'ratio_{ratio}_public')
    #             fname = f'{log_file}.txt'
    #             fh = logging.FileHandler(os.path.join(log_save, fname))
    #             fh.setFormatter(logging.Formatter(log_format))
    #             logger = logging.getLogger()
    #             logger.addHandler(fh)
    #             args.save_path = log_file

    #             val_accs = []
    #             val_micro_f1s = []
    #             val_macro_f1s = []
    #             test_accs = []
    #             test_micro_f1s = []
    #             test_macro_f1s = []
    #             logits_list = []
    #             test_lbls = t.argmax(self.label[self.test_idx[ratio]], dim=-1)

    #             # ---- EPOCH LOOP ----
    #             for ep in range(args.epoch):
    #                 # 1) train one epoch
    #                 reses = self.trainEpoch(ratio)
    #                 # print training losses
    #                 print(f"[Epoch {ep+1}/{args.epoch}] Train → "
    #                       f"bceLoss={reses['bceLoss']:.4f}, diffLoss={reses['diffLoss']:.4f}")
    #                 self.log_metapath_stats(f"Epoch {ep+1}")
    #                 # 2) evaluate every epoch
    #                 tstFlag = (ep % 1 == 0)
    #                 if tstFlag:
    #                     val_reses, test_reses = self.testEpoch(ratio)

    #                     # print validation & test metrics
    #                     print(f"[Epoch {ep+1}/{args.epoch}] Val   → "
    #                           f"acc={val_reses['acc']:.4f}, "
    #                           f"macro-F1={val_reses['macro']:.4f}, "
    #                           f"micro-F1={val_reses['micro']:.4f}")
    #                     print(f"[Epoch {ep+1}/{args.epoch}] Test  → "
    #                           f"acc={test_reses['acc']:.4f}, "
    #                           f"macro-F1={test_reses['macro']:.4f}, "
    #                           f"micro-F1={test_reses['micro']:.4f}")

    #                     # accumulate for final summary
    #                     val_accs.append(val_reses['acc'].item())
    #                     val_macro_f1s.append(val_reses['macro'])
    #                     val_micro_f1s.append(val_reses['micro'])
    #                     test_accs.append(test_reses['acc'].item())
    #                     test_macro_f1s.append(test_reses['macro'])
    #                     test_micro_f1s.append(test_reses['micro'])
    #                     logits_list.append(test_reses['logits'])

    #             # ---- end of epochs, pick best run on test set ----
    #             max_iter = test_accs.index(max(test_accs))
    #             accs.append(test_accs[max_iter])
    #             macro_f1s.append(test_macro_f1s[max_iter])
    #             macro_f1s_val.append(val_macro_f1s[max_iter])
    #             micro_f1s.append(test_micro_f1s[test_micro_f1s.index(max(test_micro_f1s))])

    #             best_logits = logits_list[max_iter]
    #             best_proba = softmax(best_logits, dim=1)
    #             auc_score_list.append(
    #                 roc_auc_score(
    #                     y_true=test_lbls.detach().cpu().numpy(),
    #                     y_score=best_proba.detach().cpu().numpy(),
    #                     multi_class='ovr'
    #                 )
    #             )

    #             # log final summary for this repeat
    #             logger.info(
    #                 "\t[Classification] Macro-F1: {:.4f} var: {:.4f}  "
    #                 "Micro-F1_mean: {:.4f} var: {:.4f}  auc {:.4f}".format(
    #                     np.mean(macro_f1s), np.std(macro_f1s),
    #                     np.mean(micro_f1s), np.std(micro_f1s),
    #                     np.mean(auc_score_list), np.std(auc_score_list)
    #                 )
    #             )


    def prepareModel(self):
        self.initial_feature = self.handler.feature_list
        self.dim = self.initial_feature.shape[1]
        self.train_idx = self.handler.train_idx
        self.test_idx = self.handler.test_idx
        self.val_idx = self.handler.val_idx
        self.label = self.handler.labels
        self.nbclasses = self.label.shape[1]
        # if args.model == 'fastgtn':
        #     # how many raw edge‐types did LoadData build?
        #     num_edge_type = len(self.handler.he_adjs)
        #     # total nodes in full het‐graph
        #     N_tot = self.handler.N_tot
        #     self.model = FastGTNWithDiffusion(
        #         f_dim       = self.dim,
        #         num_edge_type = num_edge_type,
        #         w_in        = args.latdim,
        #         num_class   = self.nbclasses,
        #         num_nodes   = N_tot,
        #         args        = args
        #     ).to(device)
        # else:
        self.model = HGDM(self.dim).to(device)
        params = list(self.model.parameters()) + list(self.handler.metapath_learner.parameters())
        self.opt = t.optim.Adam(params, lr=args.lr, weight_decay=0)

        # at the top of Coach.trainEpoch, before calling cal_loss:

        # print(">> model is:", type(self.model))
        # print(">> he_adjs[0] is:", type(self.handler.he_adjs[0]))

    def trainEpoch(self,i):

        trnLoader = index_generator(batch_size=args.batch, indices=self.train_idx[i])
       
        epBCELoss, epDFLoss = 0, 0
        self.label = self.handler.labels
        steps = trnLoader.num_iterations()
       
        for i in range(trnLoader.num_iterations()):
            # self.handler.he_adjs = self.handler.get_he_adjs()
            train_idx_batch = trnLoader.next()
            train_idx_batch.sort()
            ancs=t.LongTensor(train_idx_batch)

            nll_loss,diffloss = self.model.cal_loss(ancs, self.label,self.handler.he_adjs,self.initial_feature)
    
            loss = nll_loss +  diffloss
            epBCELoss += nll_loss.item()
            
            epDFLoss += diffloss.item()
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # log('Step %d/%d: bceloss = %.3f, diffloss = %.3f    ' % (i, steps, nll_loss,diffloss), save=False,
            #     oneline=True)
        ret = dict()
        ret['bceLoss'] = epBCELoss / steps
        ret['diffLoss'] = epDFLoss / steps
        
        
        return ret
   

    # def trainEpoch(self, ratio_idx):
    #     loader = index_generator(batch_size=args.batch,
    #                              indices=self.train_idx[ratio_idx])
    #     steps = loader.num_iterations()
    #     total_bce, total_diff = 0.0, 0.0

    #     for step in range(steps):
    #         # refresh learned metapaths each step
    #         self.handler.he_adjs = self.handler.get_he_adjs()

    #         batch = loader.next()
    #         batch.sort()
    #         ancs = t.LongTensor(batch).to(device)

    #         nll, diffloss = self.model.cal_loss(
    #             ancs, self.label, self.handler.he_adjs, self.initial_feature
    #         )
    #         loss = nll + diffloss
    #         total_bce  += nll.item()
    #         total_diff += diffloss.item()

    #         self.opt.zero_grad()
    #         loss.backward()
    #         self.opt.step()

    #         # **PRINT TRAIN STEP METRICS**
    #         if step % 10 == 0:
    #             print(f"  [Step {step}/{steps}]  bce={nll.item():.4f}, diff={diffloss.item():.4f}")

    #     avg_bce  = total_bce  / steps
    #     avg_diff = total_diff / steps
    #     print(f"[Epoch Train] avg bce={avg_bce:.4f}, avg diff={avg_diff:.4f}")
    #     return {'bceLoss': avg_bce, 'diffLoss': avg_diff}

    def testEpoch(self,i):
        labels = self.handler.labels
        test_idx = self.handler.test_idx[i]
        with t.no_grad():

            embeds,scores = self.model.get_allembeds(self.handler.he_adjs,self.initial_feature)
            val_acc,val_f1_macro,val_f1_micro,test_acc,test_f1_macro,test_f1_micro,test_logits=evaluate(embeds,scores, args.ratio[i], self.train_idx[i], self.val_idx[i], self.test_idx[i], labels, self.nbclasses)
            val_ret = dict()
            val_ret['acc'] = val_acc
            val_ret['macro'] = val_f1_macro
            val_ret['micro'] = val_f1_micro

            test_ret = dict()
            test_ret['acc'] = test_acc
            test_ret['macro'] = test_f1_macro
            test_ret['micro'] = test_f1_micro
            test_ret['logits'] = test_logits
            return val_ret,test_ret
    
    # def testEpoch(self, ratio_idx):
    #     with t.no_grad():
    #         embeds, scores = self.model.get_allembeds(
    #             self.handler.he_adjs, self.initial_feature
    #         )
    #         val_acc, val_macro, val_micro, \
    #         test_acc, test_macro, test_micro, test_logits = evaluate(
    #             embeds, scores,
    #             args.ratio[ratio_idx],
    #             self.train_idx[ratio_idx],
    #             self.val_idx[ratio_idx],
    #             self.test_idx[ratio_idx],
    #             self.label, self.nbclasses
    #         )
    #         return (
    #             {'acc': val_acc,   'macro': val_macro,   'micro': val_micro},
    #             {'acc': test_acc,  'macro': test_macro,  'micro': test_micro, 'logits': test_logits}
    #         )

    


    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('./History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        
    def saveModel(self):
        content = {
            'model': self.model,
        }
        t.save(content, './Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        
        ckp = t.load('./Models/' + args.load_model )
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        log('Model Loaded')



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True
    log('Start')
    handler = DataHandler()

    handler.metapath_learner = MetaPathLearner(
    num_edge_types=6,                # AP, PA, CP, PC, PT, TP
    num_channels=args.num_channels
    ).to(device)

    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    coach.run()
    # coach.test()
    print_metapaths(handler.metapath_learner)











import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import torch.nn.functional as F  
from params import args
from Model import HGDM, MetaPathLearner
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax
from DataHandler import DataHandler,index_generator
import numpy as np
import pickle
from Utils.Utils import *
from Utils.Utils import contrast
import os
import logging
import datetime
import sys
# from Model import FastGTNWithDiffusion
import warnings

warnings.filterwarnings("ignore")
 

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')




def print_metapaths(mp: MetaPathLearner):
    """
    Given a trained MetaPathLearner, softmax its two weight matrices
    and print the top-2 relations in each hop for each channel.
    """
    # must match the order you built raw_mats in DataHandler.load_dblp_data()
    relation_names = ['ap','pa','cp','pc','pt','tp']

    # conv1.weight, conv2.weight are both (C, E)
    alpha1 = F.softmax(mp.conv1.weight, dim=1).detach().cpu().numpy()  # (C, E)
    alpha2 = F.softmax(mp.conv2.weight, dim=1).detach().cpu().numpy()  # (C, E)
    C, E = alpha1.shape

    print("\n=== Learned Two-hop Meta-paths ===")
    for c in range(C):
        # get indices of top-2 for each hop
        top1_idx = np.argsort(alpha1[c])[::-1][:2]
        top2_idx = np.argsort(alpha2[c])[::-1][:2]

        hops1 = [(relation_names[i], round(alpha1[c,i], 3)) for i in top1_idx]
        hops2 = [(relation_names[i], round(alpha2[c,i], 3)) for i in top2_idx]

        print(f"Channel {c}:")
        print(f"  1st hop: {hops1}")
        print(f"  2nd hop: {hops2}")
    print("=================================\n")

class Coach:
    def __init__(self, handler):
        self.handler = handler
       
        self.metrics = dict()
        mets = ['bceLoss','AUC']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret
    
    def log_metapath_stats(self, tag):
        mp = self.handler.metapath_learner
        α1 = F.softmax(mp.conv1.weight, dim=1)
        α2 = F.softmax(mp.conv2.weight, dim=1)
        # e.g. l1 norms per channel
        norms1 = α1.abs().sum(dim=1).detach().cpu().numpy()
        norms2 = α2.abs().sum(dim=1).detach().cpu().numpy()
        print(f"[{tag}] MetaPath α1 ‖·‖₁ = {np.round(norms1,3)}")
        print(f"[{tag}] MetaPath α2 ‖·‖₁ = {np.round(norms2,3)}")


    def run(self):
        for ratio in range(len(self.handler.train_idx)):
            log('Ratio Type: '+str(ratio))
            accs = []
            micro_f1s = []
            macro_f1s = []
            macro_f1s_val = []
            auc_score_list = []
            for repeat in range(20):
                self.prepareModel()
                log('Repeat: '+str(repeat))

                macroMax = 0
                


                log_format = '%(asctime)s %(message)s'
                logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
                log_save = './History/'
                log_file = f'{args.data}_' + \
                                    f'lr_{args.lr}_batch_{args.batch}_noise_scale_{args.noise_scale}_step_{args.steps}_ratio_{ratio}_public'
                fname = f'{log_file}.txt'
                fh = logging.FileHandler(os.path.join(log_save, fname))
                fh.setFormatter(logging.Formatter(log_format))
                logger = logging.getLogger()
                logger.addHandler(fh)
                # logger.info(args)
                # logger.info('================')  
                args.save_path = log_file 

                val_accs = []
                val_micro_f1s = []
                val_macro_f1s = []
                test_accs = []
                test_micro_f1s = []
                test_macro_f1s = []
                logits_list = []
                test_lbls = t.argmax(self.label[self.test_idx[ratio]], dim=-1)
                # eval_every = 400
                for ep in range(args.epoch):
                    tstFlag = (ep % 1 == 0)
                    reses = self.trainEpoch(ratio)
                    # log(self.makePrint('Train', ep, reses, tstFlag))
                    if tstFlag:
                        val_reses,test_reses = self.testEpoch(ratio)
                        val_accs.append(val_reses['acc'].item())
                        val_macro_f1s.append(val_reses['macro'])
                        val_micro_f1s.append(val_reses['micro'])

                        test_accs.append(test_reses['acc'].item())
                        test_macro_f1s.append(test_reses['macro'])
                        test_micro_f1s.append(test_reses['micro'])
                        logits_list.append(test_reses['logits'])
                        # print("\t[Val_Classification] Macro-F1_epoch: {:.4f} Micro-F1_epoch: {:.4f} Test-acc_epoch: {:.4f}"
                        #     .format(val_reses['macro'],
                        # val_reses['micro'],
                        # val_reses['acc']
                        #     )
                        #     )


                        # print("\t[Test_Classification] Macro-F1_epoch: {:.4f} Micro-F1_epoch: {:.4f} Test-acc_epoch: {:.4f}"
                        #     .format(test_reses['macro'],
                        # test_reses['micro'],
                        # test_reses['acc']
                        #     )
                        #     )
                    
                    
                    
                        # log(self.makePrint('Test', ep, reses, tstFlag))
                        # if (val_reses['macro'] > macroMax):
                        #     macroMax = test_reses['macro']
                        #     self.saveModel()
                        # logger.info(self.makePrint('Test', ep, test_reses, tstFlag))
                        # self.saveHistory()
                   

                        # self.saveHistory()
                        
                        
                        

                max_iter = test_accs.index(max(test_accs))
                accs.append(test_accs[max_iter])
                max_iter = test_macro_f1s.index(max(test_macro_f1s))
                macro_f1s.append(test_macro_f1s[max_iter])
                macro_f1s_val.append(val_macro_f1s[max_iter])

                max_iter = test_micro_f1s.index(max(test_micro_f1s))
                micro_f1s.append(test_micro_f1s[max_iter])

                best_logits = logits_list[max_iter]
                best_proba = softmax(best_logits, dim=1)
                auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                                    y_score=best_proba.detach().cpu().numpy(),
                                                    multi_class='ovr'
                                                    ))
                
                # print("\t[Test_Classification] Macro-F1_one_time: {:.4f} Micro-F1_one_time: {:.4f} Test-AUC_one_time: {:.4f}"
                #             .format(macro_f1s[-1],
                #         micro_f1s[-1],
                #         auc_score_list[-1]
                #             )
                #             )
                

                logger.info("\t[Classification] Macro-F1: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
                    .format(np.mean(macro_f1s),
                            np.std(macro_f1s),
                            np.mean(micro_f1s),
                            np.std(micro_f1s),
                            np.mean(auc_score_list),
                            np.std(auc_score_list)))


    def prepareModel(self):
        self.initial_feature = self.handler.feature_list
        self.dim = self.initial_feature.shape[1]
        self.train_idx = self.handler.train_idx
        self.test_idx = self.handler.test_idx
        self.val_idx = self.handler.val_idx
        self.label = self.handler.labels
        self.nbclasses = self.label.shape[1]
        # if args.model == 'fastgtn':
        #     # how many raw edge‐types did LoadData build?
        #     num_edge_type = len(self.handler.he_adjs)
        #     # total nodes in full het‐graph
        #     N_tot = self.handler.N_tot
        #     self.model = FastGTNWithDiffusion(
        #         f_dim       = self.dim,
        #         num_edge_type = num_edge_type,
        #         w_in        = args.latdim,
        #         num_class   = self.nbclasses,
        #         num_nodes   = N_tot,
        #         args        = args
        #     ).to(device)
        # else:
        self.model = HGDM(self.dim).to(device)
        params = list(self.model.parameters()) + list(self.handler.metapath_learner.parameters())
        self.opt = t.optim.Adam(params, lr=args.lr, weight_decay=0)

        # at the top of Coach.trainEpoch, before calling cal_loss:

        # print(">> model is:", type(self.model))
        # print(">> he_adjs[0] is:", type(self.handler.he_adjs[0]))

    def trainEpoch(self,i):

        trnLoader = index_generator(batch_size=args.batch, indices=self.train_idx[i])
       
        epBCELoss, epDFLoss = 0, 0
        self.label = self.handler.labels
        steps = trnLoader.num_iterations()
       
        for i in range(trnLoader.num_iterations()):
            # self.handler.he_adjs = self.handler.get_he_adjs()
            train_idx_batch = trnLoader.next()
            train_idx_batch.sort()
            ancs=t.LongTensor(train_idx_batch)

            nll_loss,diffloss = self.model.cal_loss(ancs, self.label,self.handler.he_adjs,self.initial_feature)
    
            loss = nll_loss +  diffloss
            epBCELoss += nll_loss.item()
            
            epDFLoss += diffloss.item()
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # log('Step %d/%d: bceloss = %.3f, diffloss = %.3f    ' % (i, steps, nll_loss,diffloss), save=False,
            #     oneline=True)
        ret = dict()
        ret['bceLoss'] = epBCELoss / steps
        ret['diffLoss'] = epDFLoss / steps
        
        
        return ret
   


    def testEpoch(self,i):
        labels = self.handler.labels
        test_idx = self.handler.test_idx[i]
        with t.no_grad():

            embeds,scores = self.model.get_allembeds(self.handler.he_adjs,self.initial_feature)
            val_acc,val_f1_macro,val_f1_micro,test_acc,test_f1_macro,test_f1_micro,test_logits=evaluate(embeds,scores, args.ratio[i], self.train_idx[i], self.val_idx[i], self.test_idx[i], labels, self.nbclasses)
            val_ret = dict()
            val_ret['acc'] = val_acc
            val_ret['macro'] = val_f1_macro
            val_ret['micro'] = val_f1_micro

            test_ret = dict()
            test_ret['acc'] = test_acc
            test_ret['macro'] = test_f1_macro
            test_ret['micro'] = test_f1_micro
            test_ret['logits'] = test_logits
            return val_ret,test_ret
    


    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('./History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        
    def saveModel(self):
        content = {
            'model': self.model,
        }
        t.save(content, './Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        
        ckp = t.load('./Models/' + args.load_model )
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        log('Model Loaded')



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True
    log('Start')
    handler = DataHandler()

    handler.metapath_learner = MetaPathLearner(
    num_edge_types=6,                # AP, PA, CP, PC, PT, TP
    num_channels=args.num_channels
    ).to(device)

    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    coach.run()
    # coach.test()
    print_metapaths(handler.metapath_learner)











