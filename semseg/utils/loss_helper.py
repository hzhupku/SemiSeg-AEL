import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import scipy.ndimage as nd



def get_criterion(cfg, cons=False):
    cfg_criterion = cfg['criterion']
    aux_weight = cfg['net']['aux_loss']['loss_weight'] if cfg['net'].get('aux_loss', False) else 0
    ignore_index = cfg['dataset']['ignore_label']
    if cfg_criterion['type'] == 'ohem':
        criterion = CriterionOhem(aux_weight, ignore_index=ignore_index,
                                  **cfg_criterion['kwargs'])
    else:
        criterion = Criterion(aux_weight, ignore_index=ignore_index, **cfg_criterion['kwargs'])
    if cons:
        gamma = cfg['criterion']['cons']['gamma']
        sample = cfg['criterion']['cons'].get('sample', False)
        gamma2 = cfg['criterion']['cons'].get('gamma2',1)
        criterion = Criterion_cons(gamma, sample=sample, gamma2=gamma2, 
                                   ignore_index=ignore_index, **cfg_criterion['kwargs'])
    return criterion

class Criterion(nn.Module):
    def __init__(self, aux_weight, ignore_index=255, thresh=0.7, min_kept=100000,use_weight=False):
        super(Criterion, self).__init__()
        self._aux_weight = aux_weight
        self._ignore_index = ignore_index
        self.use_weight = use_weight
        if not use_weight:
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            weights = torch.FloatTensor(
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]).cuda()
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self._criterion1 = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert len(preds) == 2 and main_h == aux_h and main_w == aux_w and main_h == h and main_w == w
            if self.use_weight:
                loss1 = self._criterion(main_pred, target) + self._criterion1(main_pred, target)
            else:
                loss1 = self._criterion(main_pred, target)
            loss2 = self._criterion(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion(preds, target)
        return loss

class Criterion_cons(nn.Module):
    def __init__(self, gamma, sample=False, gamma2=1, 
                ignore_index=255, thresh=0.7, min_kept=100000, use_weight=False):
        super(Criterion_cons, self).__init__()
        self.gamma = gamma
        self.gamma2 = float(gamma2)
        self._ignore_index = ignore_index
        self.sample = sample
        self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    def forward(self, preds, conf, gt, dcp_criterion=None):

        #conf = F.softmax(conf)
        ce_loss = self._criterion(preds,gt)
        conf = torch.pow(conf,self.gamma)

        if self.sample:
            dcp_criterion = 1 - dcp_criterion
            dcp_criterion = dcp_criterion / (torch.max(dcp_criterion)+1e-12) 
            dcp_criterion = torch.pow(dcp_criterion, self.gamma2)
            pred_map = preds.max(1)[1].float()

            sample_map = torch.zeros_like(pred_map).float()
            h, w = pred_map.shape[-2], pred_map.shape[-1]

            for idx in range(len(dcp_criterion)):
                prob = 1 - dcp_criterion[idx]
                rand_map = torch.rand(h, w).cuda()*(pred_map == idx)
                rand_map = (rand_map>prob)*1.0
                sample_map += rand_map
            conf = conf * (sample_map)
        conf = conf/(conf.sum() + 1e-12)

        loss = conf * ce_loss
        return loss.sum()

class CriterionOhem(nn.Module):
    def __init__(self, aux_weight, thresh=0.7, min_kept=100000,  ignore_index=255, use_weight=False):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self._criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept,use_weight)
        self._criterion2 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert len(preds) == 2 and main_h == aux_h and main_w == aux_w and main_h == h and main_w == w

            loss1 = self._criterion1(main_pred, target)
            loss2 = self._criterion2(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion1(preds, target)
        return loss

class OhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0/factor, 1.0/factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0/factor, 1.0/factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor*factor)  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept)-1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(target.get_device())

        return new_target

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=256,
                 use_weight=False, reduce=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507]).cuda()
            #weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="none",
                                                       ignore_index=ignore_index)           
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            #print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

