import torch as t
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import numpy as np
def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret
def reg_pick_embeds(embeds_list):
	reg_loss = 0
	for embeds in embeds_list:
		reg_loss += embeds.square().sum()
	return reg_loss

def contrast(nodes, allEmbeds, allEmbeds2=None):
	if allEmbeds2 is not None:
		pckEmbeds = allEmbeds[nodes]
		scores = t.log(t.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
	else:
		uniqNodes = t.unique(nodes)
		pckEmbeds = allEmbeds[uniqNodes]
		scores = t.log(t.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
	return scores

def cal_infonce_loss(embeds1, embeds2, temp):
        """ InfoNCE Loss (specify nodes for contrastive learning)
        """
        embeds1 = F.normalize(embeds1 + 1e-8, p=2)
        embeds2 = F.normalize(embeds2 + 1e-8, p=2)
        pckEmbeds1 = embeds1
        pckEmbeds2 = embeds2
        nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
        deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
        return -t.log(nume / deno)

def cal_infonce_loss_spec_nodes(embeds1, embeds2, nodes, temp):
        """ InfoNCE Loss (specify nodes for contrastive learning)
        """
        embeds1 = F.normalize(embeds1 + 1e-8, p=2)
        embeds2 = F.normalize(embeds2 + 1e-8, p=2)
        pckEmbeds1 = embeds1[nodes]
        pckEmbeds2 = embeds2[nodes]
        nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
        deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
        return -t.log(nume / deno).mean()


def evaluate_results_nc(embeddings, labels, num_classes):
        print('SVM test')
        svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
        print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                        (macro_f1_mean, macro_f1_std), train_size in
                                        zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
        print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                        (micro_f1_mean, micro_f1_std), train_size in
                                        zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
        print('K-means test')
        nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
        print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
        print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

        return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std
    
def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
        random_states = [182318 + i for i in range(repeat)]
        result_macro_f1_list = []
        result_micro_f1_list = []
        for test_size in test_sizes:
                macro_f1_list = []
                micro_f1_list = []
                for i in range(repeat):
                        X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
                        svm = LinearSVC(dual=False)
                        svm.fit(X_train, y_train)
                        y_pred = svm.predict(X_test)
                        macro_f1 = f1_score(y_test, y_pred, average='macro')
                        micro_f1 = f1_score(y_test, y_pred, average='micro')
                        macro_f1_list.append(macro_f1)
                        micro_f1_list.append(micro_f1)
                result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
                result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
        return result_macro_f1_list, result_micro_f1_list
def kmeans_test(X, y, n_clusters, repeat=10):
        nmi_list = []
        ari_list = []
        for _ in range(repeat):
            kmeans = KMeans(n_clusters=n_clusters)
            y_pred = kmeans.fit_predict(X)
            nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
            ari_score = adjusted_rand_score(y, y_pred)
            nmi_list.append(nmi_score)
            ari_list.append(ari_score)
        return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def evaluate(embeds, scores,ratio, idx_train, idx_val, idx_test, label, nb_classes,
              isTest=True):
        hid_units = embeds.shape[1]
        val_logits = scores[idx_val]
        test_logits = scores[idx_test]

        val_lbls = t.argmax(label[idx_val], dim=-1)
        test_lbls = t.argmax(label[idx_test], dim=-1)


        preds = t.argmax(val_logits, dim=1)

        val_acc = t.sum(preds == val_lbls).float() / val_lbls.shape[0]
        val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
        val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

        # val_accs.append(val_acc.item())
        # val_macro_f1s.append(val_f1_macro)
        # val_micro_f1s.append(val_f1_micro)

            # test
        
        preds = t.argmax(test_logits, dim=1)

        test_acc = t.sum(preds == test_lbls).float() / test_lbls.shape[0]
        test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
        test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

        return val_acc,val_f1_macro,val_f1_micro,test_acc,test_f1_macro,test_f1_micro,test_logits


 