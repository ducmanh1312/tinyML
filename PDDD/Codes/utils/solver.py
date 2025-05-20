import os
import math
from math import isnan
import re
import pickle
import gensim
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD

import models


class Solver(object):
    def __init__(self, visual_model, textual_model, train_config, dev_config, test_config, train_data_loader,
                 dev_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
        self.visual_model = visual_model
        # self.textual_model = textual_model

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)

        # Final list
        for name, param in self.model.named_parameters():

            # Bert freezing customizations
            if self.train_config.data == "mosei":
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            elif self.train_config.data == "ur_funny":
                if "bert" in name:
                    param.requires_grad = False

            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            print('\t' + name, param.requires_grad)

        # Initialize weight of Embedding matrix with Glove embeddings
        if not self.train_config.use_bert:
            if self.train_config.pretrained_emb is not None:
                self.model.embed.weight.data = self.train_config.pretrained_emb
            self.model.embed.requires_grad = False

        if torch.cuda.is_available() and cuda:
            self.model.cuda()
            # self.textual_model.cuda()
            self.visual_model.cuda()

        if self.is_train:
            self.visual_optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.visual_model.parameters()),
                lr=self.train_config.learning_rate)
            # self.textual_optimizer = self.train_config.optimizer(
            #     filter(lambda p: p.requires_grad, self.textual_model.parameters()),
            #     lr=self.train_config.learning_rate)

        self.model.load_state_dict(torch.load(f'/media/icnlab/Data/Manh/tinyML/PDDD/model/ConvNeXt.std'), strict=False)
        #self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_2023-09-21_22:09:19.std'))

    @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 1

        self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")

        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()

        # textual_model = LSTM(class_num=118, vocab_size=500000, embedding_dim=128, hidden_dim=768, num_layers=8,
        #                      dropout=0.5)
        # visual_model = ResNet18()

        best_valid_loss = float('inf')
        visual_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.visual_optimizer, gamma=0.5)
        # textual_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.textual_optimizer, gamma=0.5)

        train_losses = []
        valid_losses = []
        best_acc = 0
        best_epoch = 0
        # self.train_config.n_epoch =
        for e in range(self.train_config.n_epoch):
            # self.textual_model.train()
            self.visual_model.train()

            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            train_loss_recon = []
            train_loss_sp = []
            train_loss = []
            for batch in tqdm(self.train_data_loader):
                self.visual_model.zero_grad()
                # self.textual_model.zero_grad()
                vision, y, text_input = batch
                input_ids, attention_masks, token_type_ids = text_input['input_ids'].squeeze(1), text_input[
                    'attention_mask'].squeeze(1), text_input['token_type_ids'].squeeze(1)

                batch_size = input_ids.size(0)
                input_ids = to_gpu(input_ids)
                attention_masks = to_gpu(attention_masks)
                token_type_ids = to_gpu(token_type_ids)
                vision = to_gpu(vision)
                y = to_gpu(y)

                # textual_feature = self.textual_model(input_ids)
                visual_feature = self.visual_model(vision)

                # textual_feature = textual_feature[0]
                # masked_output = torch.mul(attention_masks.unsqueeze(2), textual_feature)
                mask_len = torch.sum(attention_masks, dim=1, keepdim=True)
                # textual_feature = torch.sum(masked_output, dim=1, keepdim=False)

                y_tilde = self.model(textual_feature, visual_feature)

                if self.train_config.data == "ur_funny":
                    y = y.squeeze()

                cls_loss = criterion(y_tilde, y)
                diff_loss = self.get_diff_loss()
                domain_loss = self.get_domain_loss()
                recon_loss = self.get_recon_loss()
                cmd_loss = self.get_cmd_loss()

                if self.train_config.use_cmd_sim:
                    similarity_loss = cmd_loss
                else:
                    similarity_loss = domain_loss

                # loss = cls_loss + \
                #     self.train_config.diff_weight * diff_loss + \
                #     self.train_config.sim_weight * similarity_loss + \
                #     self.train_config.recon_weight * recon_loss
                loss = cls_loss

                loss.backward()

                torch.nn.utils.clip_grad_value_([param for param in self.visual_model.parameters() if param.requires_grad],
                                                self.train_config.clip)
                # torch.nn.utils.clip_grad_value_([param for param in self.textual_model.parameters() if param.requires_grad],
                #                                 self.train_config.clip)
                self.visual_optimizer.step()
                # self.textual_optimizer.step()

                train_loss_cls.append(cls_loss.item())
                train_loss_diff.append(diff_loss.item())
                train_loss_recon.append(recon_loss.item())
                train_loss.append(loss.item())
                train_loss_sim.append(similarity_loss.item())

            train_losses.append(train_loss)
            print(f"Training loss: {round(np.mean(train_loss), 4)}")

            valid_loss, valid_acc = self.eval(mode="dev")
            print("current epoch is %d, valid acc is %f, valid loss is %f" % (e, valid_acc, valid_loss))
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = e
            print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Found new best model on dev set!")
                if not os.path.exists('../checkpoints'): os.makedirs('../checkpoints')
                torch.save(self.visual_model.state_dict(), f'checkpoints/visual_model_{self.train_config.name}.std')
                # torch.save(self.textual_model.state_dict(), f'checkpoints/textual_model_{self.train_config.name}.std')
                torch.save(self.visual_optimizer.state_dict(), f'checkpoints/visual_optimizer_{self.train_config.name}.std')
                # torch.save(self.textual_optimizer.state_dict(), f'checkpoints/textual_optimizer_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.visual_model.load_state_dict(torch.load(f'checkpoints/visual_model_{self.train_config.name}.std'))
                    self.visual_optimizer.load_state_dict(torch.load(f'checkpoints/visual_optimizer_{self.train_config.name}.std'))
                    # self.textual_model.load_state_dict(torch.load(f'checkpoints/textual_model_{self.train_config.name}.std'))
                    # self.textual_optimizer.load_state_dict(torch.load(f'checkpoints/textual_optimizer_{self.train_config.name}.std'))
                    visual_lr_scheduler.step()
                    # textual_lr_scheduler.step()
                    print(f"Current visual learning rate: {self.visual_optimizer.state_dict()['param_groups'][0]['lr']}")
                    # print(f"Current textual learning rate: {self.textual_optimizer.state_dict()['param_groups'][0]['lr']}")

            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break

        self.eval(mode="test", to_print=True)
        print("Best Acc is %f, the epoch is %d" % (best_acc, best_epoch))

    def eval(self, mode=None, to_print=False):
        assert (mode is not None)
        self.model.eval()
        self.visual_model.eval()
        # self.textual_model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

            if to_print:
                self.model.load_state_dict(torch.load(
                    f'checkpoints/model_{self.train_config.name}.std'))

        with torch.no_grad():

            for batch in dataloader:
                self.model.zero_grad()
                self.visual_model.zero_grad()
                # self.textual_model.zero_grad()

                vision, y, text_input = batch
                input_ids, attention_masks, token_type_ids = text_input['input_ids'].squeeze(1), text_input[
                    'attention_mask'].squeeze(1), text_input['token_type_ids'].squeeze(1)

                batch_size = input_ids.size(0)
                input_ids = to_gpu(input_ids)
                attention_masks = to_gpu(attention_masks)
                token_type_ids = to_gpu(token_type_ids)
                vision = to_gpu(vision)
                y = to_gpu(y)

                textual_feature = self.textual_model(input_ids)
                visual_feature = self.visual_model(vision)

                textual_feature = textual_feature[0]
                masked_output = torch.mul(attention_masks.unsqueeze(2), textual_feature)
                mask_len = torch.sum(attention_masks, dim=1, keepdim=True)
                textual_feature = torch.sum(masked_output, dim=1, keepdim=False)

                y_tilde = self.model(textual_feature, visual_feature)

                if self.train_config.data == "ur_funny":
                    y = y.squeeze()

                cls_loss = self.criterion(y_tilde, y)
                loss = cls_loss

                eval_loss.append(loss.item())
                y_pred.extend(torch.argmax(y_tilde, dim=-1).detach().cpu().numpy())
                y_true.extend(y.detach().cpu().numpy())
        from sklearn.metrics import accuracy_score

        eval_loss = np.mean(eval_loss)
        # y_true = np.concatenate(y_true, axis=0).squeeze()
        # y_pred = np.concatenate(y_pred, axis=0).squeeze()

        # accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)
        accuracy = accuracy_score(y_true, y_pred)

        return eval_loss, accuracy

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        """

        if self.train_config.data == "ur_funny":
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true

            if to_print:
                print("Confusion Matrix (pos/neg) :")
                print(confusion_matrix(test_truth, test_preds))
                print("Classification Report (pos/neg) :")
                print(classification_report(test_truth, test_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))

            return accuracy_score(test_truth, test_preds)

        else:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
            corr = np.corrcoef(test_preds, test_truth)[0][1]
            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)

            f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')

            # pos - neg
            binary_truth = (test_truth[non_zeros] > 0)
            binary_preds = (test_preds[non_zeros] > 0)

            if to_print:
                print("mae: ", mae)
                print("corr: ", corr)
                print("mult_acc: ", mult_a7)
                print("Classification Report (pos/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))

            # non-neg - neg
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 0)

            if to_print:
                print("Classification Report (non-neg/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))

            return accuracy_score(binary_truth, binary_preds)

    def get_domain_loss(self, ):

        if self.train_config.use_cmd_sim:
            return 0.0

        # Predicted domain labels
        domain_pred_t = self.model.domain_label_t
        domain_pred_v = self.model.domain_label_v
        # domain_pred_a = self.model.domain_label_a

        # True domain labels
        domain_true_t = to_gpu(torch.LongTensor([0] * domain_pred_t.size(0)))
        domain_true_v = to_gpu(torch.LongTensor([1] * domain_pred_v.size(0)))
        # domain_true_a = to_gpu(torch.LongTensor([2]*domain_pred_a.size(0)))

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_cmd_loss(self, ):

        if not self.train_config.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_v, 5)
        # loss += self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_a, 5)
        # loss += self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_v, 5)
        loss = loss

        return loss

    def get_diff_loss(self):

        shared_t = self.model.utt_shared_t
        shared_v = self.model.utt_shared_v
        # shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_v = self.model.utt_private_v
        # private_a = self.model.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        # loss += self.loss_diff(private_a, shared_a)

        # Across privates
        # loss += self.loss_diff(private_a, private_t)
        # loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss

    def get_recon_loss(self, ):

        loss = self.loss_recon(self.model.utt_t_recon, self.model.utt_t_orig)
        loss += self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
        # loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss = loss / 2.0
        return loss





