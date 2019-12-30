from __future__ import division

import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

import wargs
from tools.utils import wlog, PAD, schedule_bow_lambda
from models.nn_utils import MaskSoftmax, MyLogSoftmax, Linear


class EMBLossCriterion(nn.Module):

    def __init__(self, trg_emb=None):
        super(EMBLossCriterion, self).__init__()
        wlog('using the embedding-based loss')
        assert trg_emb is not None, 'embedding loss needs target embedding'
        self.trg_word_emb = trg_emb.we
        # self.euclidean_dist = nn.PairwiseDistance(p=2, eps=1e-06, keepdim=True)

    def forward(self, prob_BLV, gold_BL, gold_mask_BL, bow_BN=None, bow_mask_BN=None):
        batch_size, max_L = gold_BL.size()
        E, bow_N = self.trg_word_emb.weight.size(1), bow_BN.size(1)
        gold_BLE = self.trg_word_emb(gold_BL) * gold_mask_BL[:, :, None]
        bow_BNE = self.trg_word_emb(bow_BN) * bow_mask_BN[:, :, None]
        gold_BLNE = gold_BLE[:, :, None, :].expand((-1, -1, bow_N, -1))
        bow_BLNE = bow_BNE[:, None, :, :].expand((-1, max_L, -1, -1))
        dist = F.pairwise_distance(bow_BLNE.permute(0, 3, 1, 2), gold_BLNE.permute(0, 3, 1, 2), p=2)
        # dist = dist.reshape(batch_size, max_L, bow_N).sum(-1)   # [B, L]
        # pred_p_t = tc.gather(prob_BLV, dim=-1, index=gold_BL[:, :, None]).squeeze(-1) # [B, L]
        # pred_p_t = pred_p_t * gold_mask_BL  # [B, L]
        # return ( pred_p_t * dist ).sum()
        prob_BLNV = prob_BLV[:, :, None, :].expand((-1, -1, bow_N, -1))
        bow_BLN = bow_BN[:, None, :].expand((-1, max_L, -1))
        pred_p_t = tc.gather(prob_BLNV, dim=-1, index=bow_BLN.unsqueeze(-1)).squeeze(-1)  # [B,L,N]
        # pred_p_t = tc.gather(prob_BLV, dim=-1, index=gold_BL.unsqueeze(-1)).squeeze(-1) # [B,L,N]
        # loss = ( ( - tc.log(pred_p_t) ) * dist ).sum(-1)
        # loss = (pred_p_t[:, :, None] * dist ).sum(-1)
        loss = (tc.abs(dist - pred_p_t)).sum(-1)
        loss = gold_mask_BL * loss
        return loss.sum()


class BOWLossCriterion(nn.Module):

    def __init__(self, output_size):
        super(BOWLossCriterion, self).__init__()
        weight = tc.ones(output_size, requires_grad=False)
        weight[PAD] = 0  # do not predict padding, same with ingore_index
        self.crit = nn.NLLLoss(weight, ignore_index=PAD, reduction='sum')
        wlog('using the bag of words loss')
        self.sigmoid = nn.Sigmoid()
        # self.ctx_map_vocab = Linear(2 * input_size, output_size, bias=True)
        # self.softmax = MaskSoftmax()

    def forward(self, pred_BLV, gold_mask_BL, bow_BN, bow_mask_BN):
        # p_b = sigmoid(sum_{t=1}^{M} s_t)
        bow_prob = self.sigmoid((pred_BLV * gold_mask_BL[:, :, None]).sum(1))  # [B, V]
        # bow_prob = self.softmax((pred_BLV * gold_mask_BL).sum(1), gold_mask_BL)
        bow_N = bow_BN.size(1)
        bow_ll_BNV = tc.log(bow_prob + 1e-20)[:, None, :].expand(-1, bow_N, -1)
        bow_ll_BNV = bow_ll_BNV * bow_mask_BN[:, :, None]
        bow_ll_flat_nV = bow_ll_BNV.view(-1, bow_ll_BNV.size(-1))
        return self.crit(bow_ll_flat_nV, bow_BN.view(-1))


class NLLLossCriterion(nn.Module):

    def __init__(self, output_size):
        weight = tc.ones(output_size, requires_grad=False)
        weight[PAD] = 0  # do not predict padding, same with ingore_index
        self.criterion = nn.NLLLoss(weight, ignore_index=PAD, reduction='sum')
        # self.criterion = nn.NLLLoss(weight, ignore_index=PAD, size_average=False)

    def forward(self, pred_ll, target):
        return self.criterion(pred_ll, target.view(-1))


# 0 standard Label Smoothing Loss
class LabelSmoothingCriterion(nn.Module):
    """ exp0 标准label smoothing """

    def __init__(self, output_size, label_smoothing=0.1):
        super(LabelSmoothingCriterion, self).__init__()
        assert 0. < label_smoothing <= 1., 'label smoothing value should be in [0, 1]'
        wlog('NLL loss with label_smoothing: {}'.format(label_smoothing))
        # all non-true labels are uniformly set to low-confidence
        self.smoothing_value = label_smoothing / (output_size - 2)  # pad 和 correct word
        one_hot = tc.full((output_size,), self.smoothing_value)
        one_hot[PAD] = 0.
        self.register_buffer('one_hot', one_hot.unsqueeze(0))  # 为什么用register_buffer
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing

    def forward(self, pred_ll, target):
        target = target.view(-1, 1)
        non_pad_mask = target.ne(PAD)
        nll_loss = -pred_ll.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -pred_ll.sum(dim=-1, keepdim=True)[non_pad_mask]
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        eps_i = self.label_smoothing / pred_ll.size(-1)
        loss = (1. - self.label_smoothing) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss
    # return loss


class DistanceLabelSmoothingCriterion(nn.Module):
    """ exp1 使用超参去固定a,b参数，达到映射到最佳距离的效果 """

    def __init__(self, src_emb, alpha, beta):
        super(DistanceLabelSmoothingCriterion, self).__init__()
        self.weight = src_emb.we
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_ll, target):
        target = target.view(-1, 1)
        non_pad_mask = target.ne(PAD)
        nll_loss = -pred_ll.gather(dim=-1, index=target)[non_pad_mask]
        pred_loss, position = tc.max(pred_ll, 1)
        pred_loss, position = -pred_loss[non_pad_mask.squeeze(1)], position[non_pad_mask.squeeze(1)]
        # gold_embedding = F.embedding(target, self.weight)
        # pred_embedding = F.embedding(position, self.weight)
        gold_embedding = self.weight(target[non_pad_mask].squeeze())
        pred_embedding = self.weight(position)
        distance = F.pairwise_distance(gold_embedding, pred_embedding)
        x = tc.sigmoid(self.alpha * distance + self.beta) / 2
        loss = (x + 0.5) * nll_loss + (0.5 - x) * pred_loss
        return loss.sum(), nll_loss.sum()
        # return loss


class DistanceAutoMLCriterion(nn.Module):
    """ exp2 设计两个大小为|V|的vector，每一位分别是词表中每一个词的alpha，
    和beta，从而学习每个词不同的映射区间，映射到最有效的
     alpha = w_alpha[w1] beta = w_beta[w1]
     x = sigmoid(alpha*distance+beta)/2
     loss = (x+0.5)*nll_loss + (0.5-x)*pre_loss
     """

    def __init__(self, src_emb):
        super(DistanceAutoMLCriterion, self).__init__()
        self.weight = src_emb.we
        self.alpha = nn.Parameter(tc.ones([src_emb.n_vocab], dtype=tc.float32))
        self.beta = nn.Parameter(tc.ones([src_emb.n_vocab], dtype=tc.float32))

    def forward(self, pred_ll, target):
        target = target.view(-1, 1)
        non_pad_mask = target.ne(PAD)
        nll_loss = -pred_ll.gather(dim=-1, index=target)[non_pad_mask]
        pred_loss, position = tc.max(pred_ll, 1)
        pred_loss, position = -pred_loss[non_pad_mask.squeeze(1)], position[non_pad_mask.squeeze(1)]
        gold_embedding = self.weight(target[non_pad_mask])
        pred_embedding = self.weight(position)
        # distance = F.pairwise_distance(gold_embedding, pred_embedding)
        distance = F.cosine_similarity(gold_embedding, pred_embedding)
        alpha = self.alpha[target[non_pad_mask]]
        beta = self.beta[target[non_pad_mask]]
        x = tc.sigmoid(alpha * distance + beta) / 2
        loss = (x + 0.5) * nll_loss + (0.5 - x) * pred_loss
        return loss.sum(), nll_loss.sum()


class DistanceAutoMLSmallMatrixCriterion(nn.Module):
    """
    exp3 直接将距离映射到到另一个值，
    f(distance) = w1(relu(w2*distance))
    x = F.sigmoid(f(distance))/2
    loss = (0.5+x)*nll_loss + (0.5-x)*pre_loss
    """

    def __init__(self, src_emb):
        super(DistanceAutoMLSmallMatrixCriterion, self).__init__()
        self.weight = src_emb.we
        self.w1 = nn.Linear(1, 512)
        self.w2 = nn.Linear(512, 1)

    def forward(self, pred_ll, target):
        target = target.view(-1, 1)
        non_pad_mask = target.ne(PAD)
        nll_loss = -pred_ll.gather(dim=-1, index=target)[non_pad_mask]
        pred_loss, position = tc.max(pred_ll, 1)
        pred_loss, position = -pred_loss[non_pad_mask.squeeze(1)], position[non_pad_mask.squeeze(1)]
        gold_embedding = self.weight(target[non_pad_mask].squeeze())
        pred_embedding = self.weight(position)
        distance = F.pairwise_distance(gold_embedding, pred_embedding)
        map_dist = self.w2(tc.relu(self.w1(distance.unsqueeze(1)))).squeeze()
        x = tc.sigmoid(map_dist) / 2
        loss = (0.5 + x) * nll_loss + (0.5 - x) * pred_loss
        return loss.sum(), nll_loss.sum()


class Word2VecDistanceCriterion(nn.Module):
    """
    exp4 利用word2vec预训练好的embedding计算距离
    直接将距离映射到到另一个值，
    distance = ||E(w1)-E(w2)||
    x = F.sigmoid(f(distance))/2
    f(distance) = w1(relu(w2*distance))
    loss = (0.5+x)*nll_loss + (0.5-x)*pre_loss

    改版

    """

    def __init__(self, wv_emb):
        super(Word2VecDistanceCriterion, self).__init__()
        self.emb = nn.Embedding.from_pretrained(wv_emb)
        self.emb.weight.requires_grad = False
        # self.w1 = nn.Linear(3, 512)
        # self.w2 = nn.Linear(512, 1)
        self.w1 = nn.Linear(200, 512)
        self.w2 = nn.Linear(512, 1)
        self.label_smoothing = 0.1

    def forward(self, pred_ll, target):
        target = target.view(-1, 1)
        non_pad_mask = target.ne(PAD)
        nll_loss = -pred_ll.gather(dim=-1, index=target)[non_pad_mask]
        pred_loss, position = tc.max(pred_ll, 1)
        pred_loss, position = -pred_loss[non_pad_mask.squeeze(1)], position[non_pad_mask.squeeze(1)]
        gold_embedding = self.emb(target[non_pad_mask])
        pred_embedding = self.emb(position)
        emb_cat = tc.cat((gold_embedding, pred_embedding), dim=1)

        map_dist = self.w2(tc.relu(self.w1(emb_cat)))
        x = tc.sigmoid(map_dist) / 2
        smooth_loss = -pred_ll.sum(dim=-1, keepdim=True)[non_pad_mask]
        smooth_loss = smooth_loss.sum()
        eps_i = self.label_smoothing / pred_ll.size(-1)
        loss = (0.45 + x) * nll_loss + (0.45 - x) * pred_loss
        loss = loss.sum() + eps_i * smooth_loss
        return loss, nll_loss.sum()


class HighDimDistance(nn.Module):
    """
    exp5 之前的方法将两个embedding的关系通过简单的欧式距离反映，
    并不能准确两个词语之间的复杂关系，通过两个embedding的拼接
    过全连接层学到的距离标量可能更加符合我们需要的‘更近’的距离。
    distance = W_matrix(|E(w1):E(w2)|) + B_bias
    x = F.sigmoid(distance)/2
    loss = (0.5+x)*nll_loss + (0.5-x)*pre_loss
    """

    def __init__(self, src_emb):
        super(HighDimDistance, self).__init__()
        self.weight = src_emb.we
        self.w1 = nn.Linear(1024, 300)
        self.w2 = nn.Linear(300, 1)

    def forward(self, pred_ll, target):
        target = target.view(-1, 1)
        non_pad_mask = target.ne(PAD)
        nll_loss = -pred_ll.gather(dim=-1, index=target)[non_pad_mask]
        pred_loss, position = tc.max(pred_ll, 1)
        pred_loss, position = -pred_loss[non_pad_mask.squeeze(1)], position[non_pad_mask.squeeze(1)]
        gold_embedding = self.weight(target[non_pad_mask])
        pred_embedding = self.weight(position)
        emb_cat = tc.cat((gold_embedding, pred_embedding), dim=1)
        map_dist = self.w2(tc.relu(self.w1(emb_cat)))
        x = tc.sigmoid(map_dist.squeeze()) / 2
        loss = (0.5 + x) * nll_loss + (0.5 - x) * pred_loss
        # print(0.5 - x)
        # print(loss.sum())
        return loss.sum(), nll_loss.sum()


class CosineDistance(nn.Module):

    def __init__(self, wv_emb):
        super(CosineDistance, self).__init__()
        self.V = wv_emb.size(0)
        self.emb = nn.Embedding.from_pretrained(wv_emb)
        self.emb.weight.requires_grad = False

    def forward(self, pred_ll, target, prob):
        target = target.view(-1, 1)
        non_pad_mask = target.ne(PAD)
        nll_loss = -pred_ll.gather(dim=-1, index=target)[non_pad_mask]
        gold_embedding = self.emb(target[non_pad_mask])
        # dist = F.cosine_similarity(gold_embedding[:, None, :], self.emb.weight[None, :, :], dim=2)
        dist = tc.norm(gold_embedding[:, None, :] - self.emb.weight[None, :, :], p=2, dim=2)
        # loss_cos = tc.abs(dist - prob[non_pad_mask.squeeze()])
        # loss_cos_sum = loss_cos.sum(1) / self.V
        # loss = nll_loss + loss_cos_sum
        # print(dist.size())
        dist_max, _ = tc.max(dist, dim=1)
        dist = tc.softmax((dist_max[:, None] - tc.abs(dist)), dim=1)
        pred_ll = -pred_ll
        loss = (dist * pred_ll[non_pad_mask.squeeze()]).sum(dim=1)

        return loss.sum(), nll_loss.sum()


class MultiGPULossCompute(nn.Module):

    def __init__(self, generator, criterion, input_size, output_size, trg_word_emb=None, bowMapper=None,
                 loss_norm='tokens', bow_crit=None, emb_crit=None, rl_crit=None, chunk_size=5, device_ids=None):

        super(MultiGPULossCompute, self).__init__()
        if bow_crit is not None or emb_crit is not None:
            wlog('using the bow loss')
            self.init_lambda, self.max_lambda, self.warmup_steps = 0., 1., 20000
            self.lambda_step = (self.max_lambda - self.init_lambda) / self.warmup_steps
            self.decay_factor = self.max_lambda * (self.warmup_steps ** 0.5)
        self.loss_norm = loss_norm
        self.chunk_size = chunk_size  # ？？
        self.device_ids = device_ids
        self.rl_crit_single = rl_crit
        # self.trg_word_emb = trg_word_emb.we
        self.output_size = output_size
        self.generator = generator
        self.criterion = criterion
        self.bowMapper = nn.parallel.replicate(bowMapper, devices=self.device_ids) if bowMapper is not None else None

    '''
    Compute the loss in shards for efficiency
        outputs: the predict outputs from the model
        gold: correct target sentences in current batch
    '''

    def forward(self, feed_BLO, e_idx, n_upds, gold_BL=None, gold_mask_BL=None, noise=None,
                bow_BN=None, bow_mask_BN=None, context_BLH=None):

        # feed_BLO: (batch_size, y_Lm1, out_size)
        # gold_flat_n, gold_mask_flat_n = gold_BL.view(-1), gold_mask_BL.view(-1)
        # (batch_size, y_Lm1, out_size)
        word_norm = gold_mask_BL.sum().item() if self.loss_norm == 'tokens' else gold_BL.size(
            0)  # 总loss除以总单词词数，总单词词数叫为word_norm
        bow_norm = bow_mask_BN.sum().item() if self.loss_norm == 'tokens' else bow_BN.size(0)  # bow 为去掉begin，打乱顺序的list
        word_norm, bow_norm = float(word_norm), float(bow_norm)
        # if contexts is not None and self.bow_loss is False: contexts = contexts.detach()

        batch_nll, batch_ok_ytoks = 0., 0
        generator = nn.parallel.replicate(self.generator, devices=self.device_ids)
        out_scatter = nn.parallel.scatter(feed_BLO, target_gpus=self.device_ids)  # push on all gpus
        # out_scatter: tuple (outs_gpu0, outs_gpu1, outs_gpu2, outs_gpu3)
        outs_grad = [[] for _ in out_scatter]
        gold_BLs = nn.parallel.scatter(gold_BL, target_gpus=self.device_ids)
        gold_mask_BLs = nn.parallel.scatter(gold_mask_BL, target_gpus=self.device_ids)
        # bow_BNE = self.trg_word_emb(bow_BN)
        bow_BNs = nn.parallel.scatter(bow_BN, target_gpus=self.device_ids)
        # bow_BNEs = nn.parallel.scatter(bow_BNE, target_gpus=self.device_ids)
        bow_mask_BNs = nn.parallel.scatter(bow_mask_BN, target_gpus=self.device_ids)

        for col_idx in range(0, out_scatter[0].size(1), self.chunk_size):
            # Predict distributions
            chunk_outs = [o[:, col_idx: col_idx + self.chunk_size].clone().detach().requires_grad_(True) for o in
                          out_scatter]
            ll_rsts = nn.parallel.parallel_apply(generator[:len(chunk_outs)], chunk_outs)
            paras = []
            for ll_rst, gold_Bl, gold_mask_Bl, bow_BN, bow_mask_BN, chunk_out in zip(
                    ll_rsts, gold_BLs, gold_mask_BLs, bow_BNs, bow_mask_BNs, chunk_outs):
                chunk_gold = gold_Bl[:, col_idx: col_idx + self.chunk_size]
                chunk_gold_mask = gold_mask_Bl[:, col_idx: col_idx + self.chunk_size]
                chunk_gold_flat_n = chunk_gold.contiguous().view(-1)
                _, prob_BLV, ll_BLV = ll_rst  # prob probability of words , negative log value of probability
                prob_flat_nV = prob_BLV.view(-1, prob_BLV.size(-1))
                # ok prediction count in one minibatch
                batch_ok_ytoks += (prob_flat_nV.max(dim=-1)[1]).eq(chunk_gold_flat_n).masked_select(
                    chunk_gold_flat_n.ne(PAD)).sum().item()  # 一个batch 单词对了几个
                ll_BLV = ll_BLV * chunk_gold_mask[:, :, None]
                # print(ll_BLV.size())
                ll_flat_nV = ll_BLV.view(-1, ll_BLV.size(-1))
                paras.append((ll_flat_nV, chunk_gold_flat_n, prob_flat_nV))

            # Compute loss.
            criterion = nn.parallel.replicate(self.criterion, devices=self.device_ids)
            list_loss_celoss = nn.parallel.parallel_apply(criterion[:len(chunk_outs)], paras)
            loss = [a[0].unsqueeze(0) for a in list_loss_celoss]
            # for l in loss:
            #    print('{}:{}'.format(l[0].get_device(), l[0].item()))
            # print(loss)
            loss_gather = nn.parallel.gather(loss, target_device=self.device_ids[0])

            # loss_gather = nn.parallel.gather(list_loss_celoss, target_device=self.device_ids[0])
            # print('gather: {}'.format(loss_gather.sum().item()))
            # chunk_gold_mask = gold_mask_BL[:, col_idx : col_idx + self.chunk_size]
            # chunk_word_norm = chunk_gold_mask.sum().item() if self.loss_norm == 'tokens' else chunk_gold_mask.size(0)
            # loss_gather = loss_gather.sum().div(float(chunk_word_norm))
            loss_gather = loss_gather.sum().div(word_norm)
            # loss_gather.backward(retain_graph=True)

            ce_loss = sum([a[1].item() for a in list_loss_celoss])
            batch_nll += ce_loss
            loss_gather.backward(retain_graph=True)
            # batch_nll += loss_gather.item()
            # Backprop loss to output of transformer
            for j, l in enumerate(list_loss_celoss):
                outs_grad[j].append(chunk_outs[j].grad.data.clone())
            # batch_nll += loss_gather.item()

        # Backprop all loss thpough transformer.
        outs_grad = [tc.cat(og, dim=1) * mask[:, :, None] for og, mask in zip(outs_grad, gold_mask_BLs)]
        o1 = feed_BLO
        o2 = nn.parallel.gather(outs_grad, target_device=self.device_ids[0])
        o1.backward(gradient=o2)

        total_bow_loss = 0.

        return batch_nll, batch_ok_ytoks, total_bow_loss
