import time
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from model import Model
from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from train_util import *
from torch.distributions import Categorical
from rouge import Rouge
from numpy import random
import argparse
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
from sklearn.metrics.pairwise import cosine_similarity

class Train(object):
    def __init__(self, opt):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        self.opt = opt
        self.start_id = self.vocab.word2id(data.START_DECODING)
        self.end_id = self.vocab.word2id(data.STOP_DECODING)
        self.pad_id = self.vocab.word2id(data.PAD_TOKEN)
        self.unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        time.sleep(5)
    def reward_bert(self, gensum, refsum):
        sentences = [gensum, refsum]
        sentence_embeddings = model.encode(sentences)
        return cosine_similarity(gensum, refsum)[0]

    def save_model(self, iter):
        save_path = "config/%02d.checkpoint" % iter
        T.save({
            "iter": iter + 1,
            "model_dict": self.model.state_dict(),
            "trainer_dict": self.trainer.state_dict()
        }, save_path)

    def setup_train(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        self.trainer = T.optim.Adam(self.model.parameters(), lr=config.lr)
        start_iter = 0
        if self.opt.load_model is not None:
            load_model_path = os.path.join(config.save_model_path, self.opt.load_model)
            checkpoint = T.load(load_model_path)
            start_iter = checkpoint["iter"]
            self.model.load_state_dict(checkpoint["model_dict"])
            self.trainer.load_state_dict(checkpoint["trainer_dict"])
            print("Loaded model at " + load_model_path)
        if self.opt.new_lr is not None:
            self.trainer = T.optim.Adam(self.model.parameters(), lr=self.opt.new_lr)
        return start_iter

    def warm_train(self, enc_out, enc_hidden, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, batch):
        dec_batch, max_dec_len, dec_lens, target_batch = get_dec_data(batch)                      
        step_losses = []
        s_t = (enc_hidden[0], enc_hidden[1])                                                      
        x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(self.start_id))                           
        prev_s = None                                                                             
        sum_temporal_srcs = None                                                                  
        for t in range(min(max_dec_len, config.max_dec_steps)):
            use_gound_truth = get_cuda((T.rand(len(enc_out)) > 0.25)).long()                    
            x_t = use_gound_truth * dec_batch[:, t] + (1 - use_gound_truth) * x_t                 
            x_t = self.model.embeds(x_t)
            final_dist, s_t, ct_e, sum_temporal_srcs, prev_s = self.model.decoder(x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s)
            target = target_batch[:, t]
            log_probs = T.log(final_dist + config.eps)
            step_loss = F.nll_loss(log_probs, target, reduction="none", ignore_index=self.pad_id)
            step_losses.append(step_loss)
            x_t = T.multinomial(final_dist, 1).squeeze()                                           
            is_oov = (x_t >= config.vocab_size).long()                                            
            x_t = (1 - is_oov) * x_t.detach() + (is_oov) * self.unk_id                             

        losses = T.sum(T.stack(step_losses, 1), 1)                                                  
        batch_avg_loss = losses / dec_lens                                                         
        mle_loss = T.mean(batch_avg_loss)                                                          
        return mle_loss

    def main_train(self, enc_out, enc_hidden, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, article_oovs, greedy):
        s_t = enc_hidden                                                                          
        x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(self.start_id))                            
        prev_s = None                                                                             
        sum_temporal_srcs = None                                                                  
        inds = []                                                                                 
        decoder_padding_mask = []                                                                  
        log_probs = []                                                                             
        mask = get_cuda(T.LongTensor(len(enc_out)).fill_(1))                                       
        for t in range(config.max_dec_steps):
            x_t = self.model.embeds(x_t)
            probs, s_t, ct_e, sum_temporal_srcs, prev_s = self.model.decoder(x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s)
            if greedy is False:
                multi_dist = Categorical(probs)
                x_t = multi_dist.sample()                                                        
                log_prob = multi_dist.log_prob(x_t)
                log_probs.append(log_prob)
            else:
                _, x_t = T.max(probs, dim=1)                                                      
            x_t = x_t.detach()
            inds.append(x_t)
            mask_t = get_cuda(T.zeros(len(enc_out)))                                                
            mask_t[mask == 1] = 1                                                                  
            mask[(mask == 1) + (x_t == self.end_id) == 2] = 0                                     
            decoder_padding_mask.append(mask_t)
            is_oov = (x_t>=config.vocab_size).long()                                               
            x_t = (1-is_oov)*x_t + (is_oov)*self.unk_id                                           

        inds = T.stack(inds, dim=1)
        decoder_padding_mask = T.stack(decoder_padding_mask, dim=1)
        if greedy is False:                                                                       
            log_probs = T.stack(log_probs, dim=1)
            log_probs = log_probs * decoder_padding_mask                                            
            lens = T.sum(decoder_padding_mask, dim=1)                                              
            log_probs = T.sum(log_probs, dim=1) / lens  # (bs,)                                     
        decoded_strs = []
        for i in range(len(enc_out)):
            id_list = inds[i].cpu().numpy()
            oovs = article_oovs[i]
            S = data.outputids2words(id_list, self.vocab, oovs)                                     
            try:
                end_idx = S.index(data.STOP_DECODING)
                S = S[:end_idx]
            except ValueError:
                S = S
            if len(S) < 2:                                                                          
                S = ["xxx"]
            S = " ".join(S)
            decoded_strs.append(S)

        return decoded_strs, log_probs

    def reward_function(self, decoded_sents, original_sents):
        rouge = Rouge()
        try:
            scores = rouge.get_scores(decoded_sents, original_sents)
            bertscore = self.reward_bert(decoded_sents, original_sents)
        except Exception:
            print("Error")
            bertscore = 0.1
            scores = []
            for i in range(len(decoded_sents)):
                try:
                    score = rouge.get_scores(decoded_sents[i], original_sents[i])
                    # with BertClient() as bc:
                        # bertscore = bc.encode([decoded_sents[i], original_sents[i]])
                except Exception:
                    score = [{"rouge-l":{"f":0.0}}]
                scores.append(score[0])
        rouge_l_f1 = [score["rouge-l"]["f"] for score in scores]
        rouge_l_f1 = get_cuda(T.FloatTensor(rouge_l_f1))
        return ((0.8*rouge_l_f1) + (1 - 0.8 * bertscore))

    def train_one_batch(self, batch, iter):
        enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, context = get_enc_data(batch)

        enc_batch = self.model.embeds(enc_batch)                                              
        enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)


        if self.opt.train_mle == "yes":        
            mle_loss = self.warm_train(enc_out, enc_hidden, enc_padding_mask, context, extra_zeros, enc_batch_extend_vocab, batch)
        else:
            mle_loss = get_cuda(T.FloatTensor([0]))
        if self.opt.train_rl == "yes":                                                        
            sample_sents, RL_log_probs = self.main_train(enc_out, enc_hidden, enc_padding_mask, context, extra_zeros, enc_batch_extend_vocab, batch.art_oovs, greedy=False)
            with T.autograd.no_grad():
                greedy_sents, _ = self.main_train(enc_out, enc_hidden, enc_padding_mask, context, extra_zeros, enc_batch_extend_vocab, batch.art_oovs, greedy=True)
            sample_reward = self.reward_function(sample_sents, batch.original_abstracts)
            baseline_reward = self.reward_function(greedy_sents, batch.original_abstracts)
            rl_loss = -(sample_reward - baseline_reward) * RL_log_probs                       
            rl_loss = T.mean(rl_loss)
            batch_reward = T.mean(sample_reward).item()
        else:
            rl_loss = get_cuda(T.FloatTensor([0]))
            batch_reward = 0
        self.trainer.zero_grad()
        (self.opt.mle_weight * mle_loss + self.opt.rl_weight * rl_loss).backward()
        self.trainer.step()
        return mle_loss.item(), batch_reward
    def trainIters(self):
        iter = self.setup_train()
        count = mle_total = r_total = 0
        while iter <= config.max_iterations:
            batch = self.batcher.next_batch()
            try:
                mle_loss, r = self.train_one_batch(batch, iter)
            except KeyboardInterrupt:
                exit()

            mle_total += mle_loss
            r_total += r
            count += 1
            iter += 1

            if iter % 1000 == 0:
                mle_avg = mle_total / count
                r_avg = r_total / count
                print("iter:", iter)
                print("loss:", "%.5f" % mle_avg)
                print("reward:", "%.5f" % r_avg)
                count = mle_total = r_total = 0

            if iter % 500 == 0:
                self.save_model(iter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mle', type=str, default="yes")
    parser.add_argument('--train_rl', type=str, default="no")
    parser.add_argument('--mle_weight', type=float, default=1.0)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--new_lr', type=float, default=None)
    opt = parser.parse_args()
    opt.rl_weight = 1 - opt.mle_weight
    train_processor = Train(opt)
    train_processor.trainIters()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mle', type=str, default="yes")
    parser.add_argument('--train_rl', type=str, default="yes")
    parser.add_argument('--mle_weight', type=float, default=0.29)
    parser.add_argument('--load_model', type=str, default="03.checkpoint")
    parser.add_argument('--new_lr', type=float, default=0.00015)
    opt = parser.parse_args()
    opt.rl_weight = 1 - opt.mle_weight
    train_processor = Train(opt)
    train_processor.trainIters()
