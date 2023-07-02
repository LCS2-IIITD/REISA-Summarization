import time
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from model import Model
from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from train_util import *
from rouge import Rouge
import argparse
import numpy as np
from train_util import get_cuda


class Beam(object):
    def __init__(self, start_id, end_id, unk_id, hidden_state, context):
        h,c = hidden_state                                             
        self.tokens = T.LongTensor(config.beam_size,1).fill_(start_id)  
        self.scores = T.FloatTensor(config.beam_size,1).fill_(-30)   
        self.tokens, self.scores = get_cuda(self.tokens), get_cuda(self.scores)
        self.scores[0][0] = 0                                         
        self.hid_h = h.unsqueeze(0).repeat(config.beam_size, 1)       
        self.hid_c = c.unsqueeze(0).repeat(config.beam_size, 1)      
        self.context = context.unsqueeze(0).repeat(config.beam_size, 1) 
        self.sum_temporal_srcs = None
        self.prev_s = None
        self.done = False
        self.end_id = end_id
        self.unk_id = unk_id

    def get_current_state(self):
        tokens = self.tokens[:,-1].clone()
        for i in range(len(tokens)):
            if tokens[i].item() >= config.vocab_size:
                tokens[i] = self.unk_id
        return tokens


    def advance(self, prob_dist, hidden_state, context, sum_temporal_srcs, prev_s):
        n_extended_vocab = prob_dist.size(1)
        h, c = hidden_state
        log_probs = T.log(prob_dist+config.eps)                         

        scores = log_probs + self.scores                               
        scores = scores.view(-1,1)                                   
        best_scores, best_scores_id = T.topk(input=scores, k=config.beam_size, dim=0)   
        self.scores = best_scores                                      
        beams_order = best_scores_id.squeeze(1)/n_extended_vocab      
        best_words = best_scores_id%n_extended_vocab                
        self.hid_h = h[beams_order]                                    
        self.hid_c = c[beams_order]                                
        self.context = context[beams_order]
        if sum_temporal_srcs is not None:
            self.sum_temporal_srcs = sum_temporal_srcs[beams_order]   
        if prev_s is not None:
            self.prev_s = prev_s[beams_order]                         
        self.tokens = self.tokens[beams_order]                        
        self.tokens = T.cat([self.tokens, best_words], dim=1)         
        if best_words[0][0] == self.end_id:
            self.done = True
    def get_best(self):
        best_token = self.tokens[0].cpu().numpy().tolist()        
        try:
            end_idx = best_token.index(self.end_id)
        except ValueError:
            end_idx = len(best_token)
        best_token = best_token[1:end_idx]
        return best_token

    def get_all(self):
        all_tokens = []
        for i in range(len(self.tokens)):
            all_tokens.append(self.tokens[i].cpu().numpy())
        return all_tokens


def beam_search(enc_hid, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, model, start_id, end_id, unk_id):

    batch_size = len(enc_hid[0])
    beam_idx = T.LongTensor(list(range(batch_size)))
    beams = [Beam(start_id, end_id, unk_id, (enc_hid[0][i], enc_hid[1][i]), ct_e[i]) for i in range(batch_size)]  
    n_rem = batch_size                                               
    sum_temporal_srcs = None                                        
    prev_s = None
    for t in range(config.max_dec_steps):
        x_t = T.stack(
            [beam.get_current_state() for beam in beams if beam.done == False]      
        ).contiguous().view(-1)                                                    
        x_t = model.embeds(x_t)                                             
        dec_h = T.stack(
            [beam.hid_h for beam in beams if beam.done == False]                  
        ).contiguous().view(-1,config.hidden_dim)
        dec_c = T.stack(
            [beam.hid_c for beam in beams if beam.done == False]               
        ).contiguous().view(-1,config.hidden_dim)                               
        ct_e = T.stack(
            [beam.context for beam in beams if beam.done == False]                 
        ).contiguous().view(-1,2*config.hidden_dim)                               
        if sum_temporal_srcs is not None:
            sum_temporal_srcs = T.stack(
                [beam.sum_temporal_srcs for beam in beams if beam.done == False]
            ).contiguous().view(-1, enc_out.size(1))                              
        if prev_s is not None:
            prev_s = T.stack(
                [beam.prev_s for beam in beams if beam.done == False]
            ).contiguous().view(-1, t, config.hidden_dim)                          
        s_t = (dec_h, dec_c)
        enc_out_beam = enc_out[beam_idx].view(n_rem,-1).repeat(1, config.beam_size).view(-1, enc_out.size(1), enc_out.size(2))
        enc_pad_mask_beam = enc_padding_mask[beam_idx].repeat(1, config.beam_size).view(-1, enc_padding_mask.size(1))
        extra_zeros_beam = None
        if extra_zeros is not None:
            extra_zeros_beam = extra_zeros[beam_idx].repeat(1, config.beam_size).view(-1, extra_zeros.size(1))
        enc_extend_vocab_beam = enc_batch_extend_vocab[beam_idx].repeat(1, config.beam_size).view(-1, enc_batch_extend_vocab.size(1))
        final_dist, (dec_h, dec_c), ct_e, sum_temporal_srcs, prev_s = model.decoder(x_t, s_t, enc_out_beam, enc_pad_mask_beam, ct_e, extra_zeros_beam, enc_extend_vocab_beam, sum_temporal_srcs, prev_s)       
        final_dist = final_dist.view(n_rem, config.beam_size, -1)                 
        dec_h = dec_h.view(n_rem, config.beam_size, -1)                           
        dec_c = dec_c.view(n_rem, config.beam_size, -1)                          
        ct_e = ct_e.view(n_rem, config.beam_size, -1)                    
        if sum_temporal_srcs is not None:
            sum_temporal_srcs = sum_temporal_srcs.view(n_rem, config.beam_size, -1)
        if prev_s is not None:
            prev_s = prev_s.view(n_rem, config.beam_size, -1, config.hidden_dim)  
        active = []        
        for i in range(n_rem):
            b = beam_idx[i].item()
            beam = beams[b]
            if beam.done:
                continue
            sum_temporal_srcs_i = prev_s_i = None
            if sum_temporal_srcs is not None:
                sum_temporal_srcs_i = sum_temporal_srcs[i]                             
            if prev_s is not None:
                prev_s_i = prev_s[i]                                                
            beam.advance(final_dist[i], (dec_h[i], dec_c[i]), ct_e[i], sum_temporal_srcs_i, prev_s_i)
            if beam.done == False:
                active.append(b)
        if len(active) == 0:
            break
        beam_idx = T.LongTensor(active)
        n_rem = len(beam_idx)
    predicted_words = []
    for beam in beams:
        predicted_words.append(beam.get_best())
    return predicted_words

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class Evaluate(object):
    def __init__(self, data_path, opt, batch_size = config.batch_size):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path, self.vocab, mode='eval',
                               batch_size=batch_size, single_pass=True)
        self.opt = opt
        time.sleep(5)
    def setup_valid(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        checkpoint = T.load(os.path.join(config.save_model_path, self.opt.load_model))
        self.model.load_state_dict(checkpoint["model_dict"])

    def evaluate_batch(self, print_sents = False):
        self.setup_valid()
        batch = self.batcher.next_batch()
        start_id = self.vocab.word2id(data.START_DECODING)
        end_id = self.vocab.word2id(data.STOP_DECODING)
        unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()
        while batch is not None:
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(batch)
            with T.autograd.no_grad():
                enc_batch = self.model.embeds(enc_batch)
                enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)
            with T.autograd.no_grad():
                pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, self.model, start_id, end_id, unk_id)
            for i in range(len(pred_ids)):
                decoded_words = data.outputids2words(pred_ids[i], self.vocab, batch.art_oovs[i])
                if len(decoded_words) < 2:
                    decoded_words = "xxx"
                else:
                    decoded_words = " ".join(decoded_words)
                decoded_sents.append(decoded_words)
                abstract = batch.original_abstracts[i]
                article = batch.original_articles[i]
                ref_sents.append(abstract)
                article_sents.append(article)
            batch = self.batcher.next_batch()
        load_file = self.opt.load_model
        self.print_original_predicted(decoded_sents, ref_sents, article_sents, load_file)
        scores = rouge.get_scores(decoded_sents, ref_sents, avg = True)
        print(load_file, "scores:", scores)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_from", type=str, default="model")
    parser.add_argument("--load_model", type=str, default=None)
    opt = parser.parse_args()
    eval_processor = Evaluate(config.test_data_path, opt)
    eval_processor.evaluate_batch()