from __future__ import division

import sys
import copy
import time
import numpy as np
import torch as tc

import wargs
from tools.utils import *


class Nbs(object):

    def __init__(self, model, tvcb_i2w, k=10, ptv=None, noise=False,
                 print_att=False, batch_sample=False):

        self.k = k
        self.ptv = ptv
        self.noise = noise
        self.xs_mask = None
        self.tvcb_i2w = tvcb_i2w
        self.print_att = print_att
        self.C = [0] * 4
        self.batch_sample = batch_sample
        self.encoder, self.decoder, self.classifier = model.encoder, model.decoder, model.decoder.classifier
        debug('Batch sampling by beam search ... {}'.format(batch_sample))

    def beam_search_trans(self, x_BL, x_mask=None, y_mask=None):

        # print '-------------------- one sentence ............'
        self.trgs_len = y_mask.sum(0).data.int().tolist() if y_mask is not None else None
        if isinstance(x_BL, list):
            x_BL = tc.tensor(x_BL).long().unsqueeze(0)
        elif isinstance(x_BL, tuple):
            x_BL = x_BL[1].unsqueeze(0)
        self.B, self.srcL = x_BL.size(0), x_BL.size(1)
        if x_mask is None:
            x_mask = tc.ones((1, self.srcL), requires_grad=False)
            if wargs.gpu_id is not None: x_mask = x_mask.cuda()
        assert not (self.batch_sample ^ (self.trgs_len is not None)), 'sample ^ trgs_len'

        self.beam, self.hyps = [], [[] for _ in range(self.B)]
        self.batch_tran_cands = [[] for _ in range(self.B)]
        self.attent_probs = [[] for _ in range(self.B)] if self.print_att is True else None

        self.maxL = y_mask.size(1) if self.batch_sample is True else 2 * self.srcL
        # get initial state of decoder rnn and encoder context
        if wargs.gpu_id is not None and not x_BL.is_cuda: x_BL = x_BL.cuda()
        self.enc_src0 = self.encoder(x_BL, xs_mask=x_mask)
        self.s0, self.uh0 = self.decoder.init_state(self.enc_src0, xs_mask=x_mask)
        # if wargs.dec_layer_cnt > 1: self.s0 = [self.s0] * wargs.dec_layer_cnt
        # (1, trg_nhids), (1, src_len, src_nhids*2)
        init_beam(self.beam, cnt=self.maxL, s0=self.s0)

        self.batch_search()
        # best_trans w/o <bos> and <eos> !!!

        # batch_tran_cands: [(trans, loss, attend)]
        for bidx in range(self.B):
            debug([(a[0], a[1]) for a in self.batch_tran_cands[bidx]])
            best_trans, best_loss = self.batch_tran_cands[bidx][0][0], self.batch_tran_cands[bidx][0][1]
            debug('Src[{}], maskL[{}], hyp (w/o EOS)[{}], maxL[{}], loss[{}]'.format(
                int(x_mask[:, bidx].sum()), self.srcL, len(best_trans), self.maxL, best_loss))
        debug('Average location of bp [{}/{}={:6.4f}]'.format(self.C[1], self.C[0], self.C[1] / self.C[0]))
        debug('Step[{}] stepout[{}]'.format(*self.C[2:]))

        # return filter_reidx(best_trans, self.tvcb_i2w), best_loss, attent_matrix
        return self.batch_tran_cands

    # @exeTime
    def batch_search(self):

        # s0: (B, trg_nhids), enc_src0: (B, srcL, src_nhids*2), uh0: (B, srcL, align_size)
        hyp_scores = np.zeros(1).astype('float32')
        delete_idx, prevb_id = None, None
        for i in range(1, self.maxL + 1):

            if all([len(a) > 0 for a in self.batch_tran_cands]) is True:
                debug('Early stop~ Normal beam search or sampling in this batch finished.')
                return
            B_prevbs = self.beam[i - 1]
            debug('\n{} Look Beam-{} {}'.format('-' * 20, i - 1, '-' * 20))
            for bidx, sent_prevb in enumerate(B_prevbs):
                debug('Sent {} '.format(bidx))
                for b in sent_prevb:  # do not output state
                    debug(b[0:1] + (None if b[1] is None else b[1].size(), b[-4].size()) + b[-3:])

            debug('\n{} Step-{} {}'.format('#' * 20, i, '#' * 20))
            '''
                [
                    batch_0 : [ (beam_item_0), (beam_item_1), ...,  ]
                    batch_1 : [ (beam_item_0), (beam_item_1), ...,  ]
                    batch_2 : [ (beam_item_0), (beam_item_1), ...,  ]
                    ......
                    batch_39: [ (beam_item_0), (beam_item_1), ...,  ]
                ]
            '''

            n_remainings = len(B_prevbs)
            enc_src, uh, y_im1, prebs_sz, hyp_scores, s_im1, self.true_bidx = \
                [], [], [], [], [], [], []
            for bidx in range(n_remainings):
                prevb = B_prevbs[bidx]
                preb_sz = len(prevb)
                prebs_sz.append(preb_sz)
                hyp_scores += list(zip(*prevb))[0]
                s_im1 += list(zip(*prevb))[-4]
                y_im1 += list(zip(*prevb))[-2]
                self.true_bidx.append(prevb[0][-3])
                if self.enc_src0.dim() == 4:
                    # (L, L, 1, src_nhids) -> (L, L, preb_sz, src_nhids)
                    enc_src, uh = self.enc_src0.expand(1, 1, preb_sz, 1), self.uh0.expand(1, 1, preb_sz, 1)
                elif self.enc_src0.dim() == 3:
                    # (src_sent_len, B, src_nhids) -> (src_sent_len, B*preb_sz, src_nhids)
                    enc_src.append(self.enc_src0[bidx].repeat(preb_sz, 1, 1))
                    uh.append(self.uh0[bidx].repeat(preb_sz, 1, 1))
            enc_src, uh = tc.cat(enc_src, dim=1), tc.cat(uh, dim=1)

            cnt_bp = (i >= 2)
            if cnt_bp is True: self.C[0] += sum(prebs_sz)
            hyp_scores = np.array(hyp_scores)
            s_im1 = tc.stack(s_im1)

            def track_ys(cur_bidx):
                y_part_seqs = []
                for b in self.beam[cur_bidx - 1][0]:
                    seq, bp = [b[-2]], b[-1]
                    for i in reversed(range(0, cur_bidx - 1)):
                        _, _, _, _, w, backptr = self.beam[i][0][bp]
                        seq.append(w)
                        bp = backptr
                    y_part_seqs.append(seq[::-1])
                return y_part_seqs

            y_part_seqs = track_ys(i)  # (preb_sz, trg_part_L)
            y_part_seqs = tc.tensor(y_part_seqs, requires_grad=False).view(-1, i)
            if wargs.gpu_id is not None: y_part_seqs = y_part_seqs.cuda()

            debug(y_im1)
            debug(y_part_seqs)
            y_im1 = self.decoder.trg_word_emb(y_part_seqs)[1][:, -1, :]
            step_output = self.decoder.step(s_im1, enc_src, uh, y_im1)
            a_i, s_i, y_im1, alpha_ij = step_output[:4]
            # a_i: (n_remainings*p, enc_hid_size), s_i: (n_remainings*p, dec_hid_size),
            # y_im1: (n_remainings*p, trg_wemb_size), alpha_ij: (n_remainings*p, srcL)

            self.C[2] += 1
            # (preb_sz, out_size)
            logit = self.decoder.step_out(y_im1, a_i, s_i)
            self.C[3] += 1

            # (n_remainings*prevb_sz, vocab_size)
            # wlog('bleu sampling, noise {}'.format(self.noise))
            next_ces = self.classifier(logit, noise=self.noise)
            next_ces = next_ces.cpu().data.numpy()
            voc_size = next_ces.shape[1]
            cand_scores = hyp_scores[:, None] + next_ces

            a, split_idx = 0, []
            for preb_sz in prebs_sz[:-1]:
                a += preb_sz
                split_idx.append(a)
            next_ces_B_prevb = np.split(cand_scores, split_idx, axis=0)  # [B: (prevb, vocab)]
            debug(len(next_ces_B_prevb))
            _s_i, _alpha_ij = [], []
            for _idx in prebs_sz[:-1]:
                _s_i.append(s_i[:_idx])
                _alpha_ij.append(alpha_ij[:_idx].t())
                s_i, alpha_ij = s_i[_idx:], alpha_ij[_idx:]
            _s_i.append(s_i)  # [B: (prevb, dec_hid_size)]
            _alpha_ij.append(alpha_ij.t())  # # [B: (srcL, preb_sz)]
            next_step_beam, del_batch_idx = [], []
            import copy
            if self.batch_sample is True: _next_ces_B_prevb = copy.deepcopy(next_ces_B_prevb)  # bak
            for bidx in range(n_remainings):
                true_id, next_ces_prevb = self.true_bidx[bidx], next_ces_B_prevb[bidx]
                if self.attent_probs is not None: self.attent_probs[true_id].append(_alpha_ij[bidx])
                if len(self.hyps[true_id]) == self.k: continue
                # if len(self.hyps[true_id]) == self.k: continue  # have finished this sentence
                debug('Sent {}, {} hypos left ----'.format(bidx, self.k - len(self.hyps[true_id])))
                if self.batch_sample is True:
                    debug(next_ces_prevb.shape)
                    debug(next_ces_prevb[:, :8])
                    if i < self.trgs_len[true_id] - 1:
                        '''here we make the score of <e> so large that <e> can not be selected'''
                        next_ces_prevb[:, EOS] = [float('+inf')] * prebs_sz[bidx]
                    elif i == self.trgs_len[true_id] - 1:
                        '''here we make the score of <e> so large that <e> can not be selected'''
                        next_ces_prevb[:, EOS] = [float('-inf')] * prebs_sz[bidx]
                    else:
                        debug('Impossible ...')
                        import sys
                        sys.exit(0)
                    debug(next_ces_prevb[:, :8])
                cand_scores_flat = next_ces_prevb.flatten()
                ranks_flat = part_sort(cand_scores_flat, self.k - len(self.hyps[true_id]))
                prevb_id = ranks_flat // voc_size
                # debug('For beam [{}], pre-beam ids: {}'.format(i, prevb_id))
                word_indices = ranks_flat % voc_size
                costs = cand_scores_flat[ranks_flat] if self.batch_sample is False else \
                    _next_ces_B_prevb[bidx].flatten()[ranks_flat]
                next_beam_cur_sent = []
                for _j, b in enumerate(
                        zip(costs, _s_i[bidx][prevb_id], [true_id] * len(prevb_id), word_indices, prevb_id)):
                    delete_idx = []
                    bp = b[-1]
                    if wargs.len_norm == 0:
                        score = (b[0], None)
                    elif wargs.len_norm == 1:
                        score = (b[0] / i, b[0])
                    elif wargs.len_norm == 2:  # alpha length normal
                        lp, cp = lp_cp(bp, i, bidx, self.beam)
                        score = (b[0] / lp + cp, b[0])
                    if cnt_bp: self.C[1] += (bp + 1)
                    if b[-2] == EOS:
                        # assert self.batch_sample is False, 'Impossible ...'
                        delete_idx.append(bp)
                        self.hyps[true_id].append(score + b[-3:] + (i,))  # contains <b> and <e>
                        debug('Gen hypo {} {} {}'.format(bidx, true_id, self.hyps[true_id][-1]))
                        # because i starts from 1, so the length of the first beam is 1, no <bos>
                        if len(self.hyps[true_id]) == self.k:
                            # output sentence, early stop, best one in k
                            debug('Early stop! see {} hyps ending with EOS.'.format(self.k))
                            sorted_hyps = sorted(self.hyps[true_id], key=lambda tup: tup[0])
                            for hyp in sorted_hyps: debug('{}'.format(hyp))
                            # best_hyp = sorted_hyps[0]
                            # debug('Best hyp length (w/ EOS)[{}]'.format(best_hyp[-1]))
                            del_batch_idx.append(bidx)
                            self.batch_tran_cands[true_id] = [back_tracking(self.beam, bidx, hyp, \
                                                                            self.attent_probs[
                                                                                true_id] if self.attent_probs is not None \
                                                                                else None) for hyp in sorted_hyps]
                    # should calculate when generate item in current beam
                    else:
                        if wargs.len_norm == 2:
                            next_beam_cur_sent.append((b[0], _alpha_ij[bidx][:, bp]) + b[1:])
                        else:
                            next_beam_cur_sent.append(b)
                if len(next_beam_cur_sent) > 0: next_step_beam.append(next_beam_cur_sent)
            self.beam[i] = next_step_beam

            if len(del_batch_idx) < n_remainings:
                self.enc_src0 = self.enc_src0[list(filter(lambda x: x not in del_batch_idx, range(n_remainings)))]
                self.uh0 = self.uh0[list(filter(lambda x: x not in del_batch_idx, range(n_remainings)))]
            del enc_src, uh, y_im1, y_part_seqs, s_im1  # free the tensor

        # no early stop, back tracking
        n_remainings = len(self.beam[self.maxL])  # loop ends, how many sentences left
        self.no_early_best(n_remainings)

    def no_early_best(self, n_remainings):

        if n_remainings == 0: return
        debug('==Start== No early stop ...')
        for bidx in range(n_remainings):
            true_id = self.true_bidx[bidx]
            debug('Sent {}, true id {}'.format(bidx, true_id))
            hyps = self.hyps[true_id]
            if len(hyps) == self.k: continue  # have finished this sentence
            # no early stop, back tracking
            if len(hyps) == 0:
                debug('No early stop, no hyp with EOS, select k hyps length {} '.format(self.maxL))
                for hyp in self.beam[self.maxL][bidx]:
                    if wargs.len_norm == 0:
                        score = (hyp[0], None)
                    elif wargs.len_norm == 1:
                        score = (hyp[0] / self.maxL, hyp[0])
                    elif wargs.len_norm == 2:  # alpha length normal
                        lp, cp = lp_cp(hyp[-1], self.maxL, bidx, self.beam)
                        score = (hyp[0] / lp + cp, hyp[0])
                    hyps.append(score + hyp[-3:] + (self.maxL,))
            else:
                debug('No early stop, no enough {} hyps with EOS, select the best '
                      'one from {} hyps.'.format(self.k, len(hyps)))
                # best_hyp = sorted_hyps[0]
            sorted_hyps = sorted(hyps, key=lambda tup: tup[0])
            for hyp in sorted_hyps: debug('{}'.format(hyp))
            debug('Sent {}: Best hyp length (w/ EOS)[{}]'.format(bidx, sorted_hyps[0][-1]))
            self.batch_tran_cands[true_id] = [back_tracking(self.beam, bidx, hyp, \
                                                            self.attent_probs[true_id] if self.attent_probs is not None \
                                                                else None) for hyp in sorted_hyps]
        debug('==End== No early stop ...')
