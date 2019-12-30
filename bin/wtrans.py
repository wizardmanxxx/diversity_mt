# coding=utf-8
from __future__ import division

import os
import io
import sys
import time
from tools.utils import init_dir, wlog
import argparse
import torch as tc
import torch.nn as nn
from gensim.models import Word2Vec, KeyedVectors
import sys

sys.path.append(os.getcwd())
import wargs

from tools.inputs import Input
from translate import Translator
from tools.utils import load_model, wlog, dec_conf, init_dir, append_file
from tools.inputs_handler import extract_vocab, wrap_tst_data

if __name__ == '__main__':

    # A = argparse.ArgumentParser(prog='NMT translator ... ')
    # A.add_argument('-m', '--model-file', required=True, dest='model_file', help='model file')
    # A.add_argument('-i', '--input-file', dest='input_file', default=None,
    #                help='name of file to be translated')
    # A.add_argument('-g', '--gpu-ids', type=int, dest='gpu_ids', nargs='+', default=[0],
    #                help='which gpu device to decode on')

    A = argparse.ArgumentParser(prog='NMT translator ... ')
    A.add_argument('-m', '--model-file', default='/home/wen/test_nmt/wmodel/model_type_5_e21_upd23000.pt',
                   dest='model_file', help='model file')
    A.add_argument('-i', '--input-file', dest='input_file', default='hass',
                   help='name of file to be translated')
    A.add_argument('-g', '--gpu-ids', type=int, dest='gpu_ids', nargs='+', default=[0],
                   help='which gpu device to decode on')

    '''
    A.add_argument('--search-mode', dest='search_mode', default=2, help='0: Greedy, 1: naive beam search')
    A.add_argument('--beam-size', dest='beam_size', default=wargs.beam_size, help='beamsize')
    A.add_argument('--len-norm', dest='len_norm', type=int, default=1,
                   help='During searching, whether we normalize accumulated loss by length.')

    '''

    args = A.parse_args()
    model_file = args.model_file
    '''
    search_mode = args.search_mode
    beam_size = args.beam_size
    lenNorm = args.len_norm
    '''

    if wargs.share_vocab is False:
        wlog('Starting load both vocabularies ... ')
        assert os.path.exists(wargs.src_vcb) and os.path.exists(wargs.trg_vcb), 'need vocabulary ...'
        src_vocab = extract_vocab(None, wargs.src_vcb)
        trg_vocab = extract_vocab(None, wargs.trg_vcb)
    else:
        wlog('Starting load shared vocabularies ... ')
        assert os.path.exists(wargs.src_vcb), 'need shared vocabulary ...'
        trg_vocab = src_vocab = extract_vocab(None, wargs.src_vcb)
    n_src_vcb, n_trg_vcb = src_vocab.size(), trg_vocab.size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(n_src_vcb, n_trg_vcb))

    # wv = KeyedVectors.load('word_vector_en', mmap='r')
    # voc = list(wv.vocab)
    # weight = tc.zeros(n_trg_vcb, 100)
    # for i, k in trg_vocab.idx2key.items():
    #     if k not in wv:
    #         continue
    #     weight[i, :] = tc.Tensor(wv[k])
    # dic = {
    #     'w2v': weight
    # }
    # tc.save(dic, '../zh_en_w2v_embedding.pt')

    model_dict, e_idx, e_bidx, n_steps, optim = load_model(model_file)
    from models.embedding import WordEmbedding

    src_emb = WordEmbedding(n_src_vcb, wargs.d_src_emb,
                            position_encoding=wargs.position_encoding, prefix='Src')
    trg_emb = WordEmbedding(n_trg_vcb, wargs.d_trg_emb,
                            position_encoding=wargs.position_encoding, prefix='Trg')
    from models.model_builder import build_NMT

    nmtModel = build_NMT(src_emb, trg_emb)
    if args.gpu_ids is not None:
        wlog('push model onto GPU {} ... '.format(args.gpu_ids[0]), 0)
        nmtModel.to(tc.device(type='cuda', index=args.gpu_ids[0]))
    else:
        wlog('push model onto CPU ... ', 0)
        nmtModel.to(tc.device('cpu'))
    wlog('done.')
    nmtModel.load_state_dict(model_dict)
    wlog('\nFinish to load model.')

    dec_conf()

    nmtModel.eval()
    tor = Translator(nmtModel, src_vocab.idx2key, trg_vocab.idx2key, print_att=wargs.print_att,
                     gpu_ids=args.gpu_ids)

    # input_file = '{}{}.{}'.format(wargs.val_tst_dir, args.input_file, wargs.val_src_suffix)
    input_file = '/home/wen/test_nmt/data/iwslt14.tokenized.de-en/train.trg'
    input_abspath = os.path.realpath(input_file)
    wlog('Translating test file {} ... '.format(input_abspath))
    # ref_file = '{}{}.{}'.format(wargs.val_tst_dir, args.input_file, wargs.val_ref_suffix)
    # ref_file = '/home/wen/test_nmt/data/iwslt14.tokenized.de-en/train.src'
    # ref_file_abspath = os.path.realpath(ref_file)
    test_src_tlst, _, sorted_index = wrap_tst_data(input_abspath, src_vocab, char=wargs.src_char, is_sort=True)
    # # tc.save(sorted_index, 'index_list.pt')
    # index = tc.load('../index_list.pt')
    wargs.test_batch_size = 20000
    test_input_data = Input(test_src_tlst[:150000], None, batch_size=wargs.test_batch_size, batch_type='token',
                            batch_sort=False,
                            gpu_ids=args.gpu_ids)

    batch_tst_data = None
    # if os.path.exists(ref_file_abspath) and False:
    #     wlog('With force decoding test file {} ... to get alignments'.format(input_file))
    #     wlog('\t\tRef file {}'.format(ref_file))
    #     from tools.inputs_handler import wrap_data
    #
    #     tst_src_tlst, tst_trg_tlst = wrap_data(wargs.val_tst_dir, args.input_file,
    #                                            wargs.val_src_suffix, wargs.val_ref_suffix,
    #                                            src_vocab, trg_vocab, False, False, 1000000)
    #     batch_tst_data = Input(tst_src_tlst, tst_trg_tlst, batch_size=wargs.test_batch_size, batch_sort=False)

    rst = tor.token_batch_trans_file(test_input_data, batch_tst_data=batch_tst_data)
    trans, tloss, wloss, sloss, alns = rst['translation'], rst['total_loss'], \
                                       rst['word_level_loss'], rst['sent_level_loss'], rst['total_aligns']
    if wargs.search_mode == 0:
        p1 = 'greedy'
    elif wargs.search_mode == 1:
        p1 = 'nbs'
    p2 = 'gpu' if args.gpu_ids is not None else 'cpu'

    outdir = 'wout_{}_{}'.format(p1, p2)
    init_dir(outdir)
    outprefix = '{}/{}'.format(outdir, args.input_file)
    # wout_nbs_gpu_wb_wvalid/nist06_
    file_out = "{}_e{}_b{}_upd{}_k{}".format(outprefix, e_idx, e_bidx, n_steps, wargs.beam_size)

    mteval_bleu = tor.write_file_eval(file_out, trans, args.input_file, alns)
    bleus_record_fname = '{}/record_bleu.log'.format(outdir)
    bleu_content = 'epoch [{}], batch[{}], BLEU score : {}'.format(e_idx, e_bidx, mteval_bleu)
    with io.open(bleus_record_fname, mode='a', encoding='utf-8') as f:
        f.write(bleu_content + '\n')
        f.close()

    sfig = '{}/{}'.format(outdir, 'record_bleu.sfig')
    sfig_content = ('{} {} {} {} {}').format(
        e_idx,
        e_bidx,
        wargs.search_mode,
        wargs.beam_size,
        mteval_bleu
    )
    append_file(sfig, sfig_content)
