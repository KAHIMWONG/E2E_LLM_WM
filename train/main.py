# <editor-fold desc="head">
import os
import config as cfg
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
import torch
torch.set_num_threads(1)

import math
from tqdm import tqdm
import itertools
import numpy as np
import os.path as op
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

from huggingface_hub import login
login(token='your_huggingface_account')  # TODO

from model import WatermarkEncoder, WatermarkDecoder
from dataset import get_data_loader
# </editor-fold>

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class Trainer:
    def __init__(self):

        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.prompt_len = cfg.prompt_len
        self.batch_size = cfg.batch_size

        self.get_model()
        self.dl = get_data_loader(self.tokenizer, self.prompt_len, self.batch_size)

        self.ce_loss = nn.BCEWithLogitsLoss()
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.scaler = GradScaler()  # for float16 training
        self.step = cfg.init_step

        self.set_cfg()

    def set_cfg(self):
        self.ce_loss_w = cfg.ce_loss_w
        self.max_gen_len = cfg.max_gen_len
        self.llm_temp = cfg.llm_temp
        self.gss_tau = cfg.gss_tau
        self.top_k_logits = cfg.top_k_logits
        self.wm_delta = cfg.wm_delta
        self.context_win_size = cfg.context_win_size
        # -------------------------------------------------------------
        self.start_semantic_loss_step = cfg.start_semantic_loss_step
        self.start_noise_step = cfg.start_noise_step
        # -------------------------------------------------------------
        # paraphrase prompt
        self.pp_prompt_start = cfg.pp_prompt_start
        self.pp_prompt_end = cfg.pp_prompt_end
        # semantic embed prompt
        self.se_prompt_start = cfg.se_prompt_start
        self.se_prompt_end = cfg.se_prompt_end
        # convert text to embed
        self.pp_ids_start, self.pp_embed_start = self.text2embed(self.pp_prompt_start)
        self.pp_ids_end, self.pp_embed_end = self.text2embed(self.pp_prompt_end)
        self.se_ids_start, self.se_embed_start = self.text2embed(self.se_prompt_start)
        self.se_ids_end, self.se_embed_end = self.text2embed(self.se_prompt_end)
        # -------------------------------------------------------------
        self.print_log_fq = cfg.print_log_fq
        self.save_ckpt_fq = cfg.save_ckpt_fq
        # -------------------------------------------------------------
        self.save_ckpt_dir = cfg.ckpt_dir
        # save config.py
        proj = op.dirname(op.abspath(__file__))
        cfg_py_path = op.join(proj, 'config.py')
        os.system(f'cp {cfg_py_path} {cfg.exp_dir}')

    def get_model(self):
        from transformers import OPTForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side="left")
        self.llm = OPTForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16).cuda()
        # freeze llm
        for name, param in self.llm.named_parameters():
            param.requires_grad = False
        self.llm.eval()
        self.llm_hidden_size = self.llm.config.hidden_size
        self.embed_matrix = self.llm.get_input_embeddings().weight
        self.vocab_size = self.tokenizer.vocab_size
        self.vocab_bit_n = math.ceil(np.log2(self.vocab_size))

        self.enc = WatermarkEncoder(input_dim=self.llm_hidden_size).cuda()
        self.dec = WatermarkDecoder(input_dim=self.llm_hidden_size).cuda()
        self.enc_optim = torch.optim.Adam(self.enc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.dec_optim = torch.optim.Adam(self.dec.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if cfg.ckpt_path is not None:
            ckpt = torch.load(cfg.ckpt_path)
            print('Load checkpoint:', cfg.ckpt_path)
            print('Load encoder:', ckpt['enc'])
            miss, unexpect = self.enc.load_state_dict(ckpt['enc'])
            print('Missed keys:', miss)
            print('Unexpected keys:', unexpect)
            print('Load decoder:', ckpt['dec'])
            miss, unexpect = self.dec.load_state_dict(ckpt['dec'])
            print('Missed keys:', miss)
            print('Unexpected keys:', unexpect)
            print('Load encoder optimizer:', ckpt['enc_opt'])
            self.enc_optim.load_state_dict(ckpt['enc_opt'])
            print('Load decoder optimizer:', ckpt['dec_opt'])
            self.dec_optim.load_state_dict(ckpt['dec_opt'])
        self.enc.train()
        self.dec.train()

    def text2embed(self, text):
        embedding_matrix = self.llm.get_input_embeddings().weight
        ids = self.tokenizer.encode(text, return_tensors="pt")[:, 1:].cuda()
        one_hot = torch.zeros(ids.size(1), embedding_matrix.size(0), dtype=torch.float16).cuda()
        one_hot[torch.arange(ids.size(1)), ids[0]] = 1
        embed = torch.matmul(one_hot, embedding_matrix)
        return ids, embed

    def get_bos_idx(self, ids, bos_id):
        b_idx, bos_idx = torch.where(ids == bos_id)
        if b_idx.size(0) != ids.size(0):
            new_b_idx, new_seq_idx = [], []
            cnt = 0
            for i, b in enumerate(b_idx):
                if cnt == b:
                    new_b_idx.append(b)
                    new_seq_idx.append(bos_idx[i])
                    cnt += 1
            b_idx, bos_idx = torch.tensor(new_b_idx), torch.tensor(new_seq_idx)
        return bos_idx

    def llm_forward(self, input_ids, attn_mask, prompt_len):
        b = input_ids.shape[0]  # batch size

        # with prompt
        wm_embed = self.llm.get_input_embeddings()(input_ids).cuda()
        nwm_embed = self.llm.get_input_embeddings()(input_ids).cuda()
        wm_ids = input_ids.clone()
        nwm_ids = input_ids.clone()

        for _ in range(self.max_gen_len):
            # gen wm embed
            with torch.no_grad():
                logit = self.llm(inputs_embeds=wm_embed, attention_mask=attn_mask).logits[:, -1, :].squeeze(dim=1)

                # previous tokens
                previous_ids = wm_ids[:, -(self.context_win_size - 1):]
                previous_ids = previous_ids[:, None, :].repeat(1, self.top_k_logits, 1)
                # top-k candidate tokens
                _, candidate_ids = torch.topk(logit, self.top_k_logits, dim=-1)
                cat_ids = torch.cat([previous_ids, candidate_ids.unsqueeze(-1)], dim=-1)
                cand_embed = self.embed_matrix[candidate_ids]
                enc_inputs = self.embed_matrix[cat_ids]

            # logit requires grad
            enc_logit = self.enc(enc_inputs)  # [b, top_k]
            # top_k_logit = torch.gather(logit, 1, candidate_ids)
            # top_k_logit = top_k_logit + delta * perturb_top_k_logit
            perturb_logit = torch.zeros_like(logit)

            perturb_logit.scatter_add_(1, candidate_ids, self.wm_delta * enc_logit)

            logit = logit + perturb_logit
            # soft_one_hot = F.softmax(logit/temp, dim=-1).to(embed_matrix.dtype)
            # soft_one_hot = F.gumbel_softmax(F.log_softmax(logit/temp, dim=-1), tau=tau, hard=False, dim=-1).to(
            #     embed_matrix.dtype)
            # cand_max_val, cand_max_index = torch.max(soft_one_hot, dim=-1, keepdim=True)
            # hard_one_hot = torch.zeros_like(soft_one_hot)
            # hard_one_hot[soft_one_hot == cand_max_val] = 1  # [b]
            # mix_one_hot = hard_one_hot * 0.5 + soft_one_hot * 0.5
            # max_index = candidate_ids[torch.arange(b), cand_max_index.squeeze()]
            soft_one_hot = F.gumbel_softmax(F.log_softmax(logit / self.llm_temp, dim=-1), tau=self.gss_tau, hard=False,
                                            dim=-1).to(
                self.embed_matrix.dtype)
            max_val, max_index = torch.max(soft_one_hot, dim=-1, keepdim=True)
            hard_one_hot = torch.zeros_like(soft_one_hot)
            hard_one_hot[soft_one_hot == max_val] = 1  # [b]
            mix_one_hot = hard_one_hot * 0.5 + soft_one_hot * 0.5

            # Get next token embedding
            # pred_embed = mix_one_hot.unsqueeze(1) @ cand_embed
            pred_embed = torch.matmul(mix_one_hot, self.embed_matrix)
            wm_embed = torch.cat([wm_embed, pred_embed.unsqueeze(1)], dim=1)
            wm_ids = torch.cat([wm_ids, max_index], dim=1)

            # gen nwm embed
            with torch.no_grad():
                logit = self.llm(inputs_embeds=nwm_embed, attention_mask=attn_mask).logits[:, -1, :].squeeze(dim=1)

            # soft_one_hot = F.softmax(logit/temp, dim=-1).to(embed_matrix.dtype)
            soft_one_hot = F.gumbel_softmax(F.log_softmax(logit / self.llm_temp, dim=-1), tau=self.gss_tau, hard=False,
                                            dim=-1).to(
                self.embed_matrix.dtype)
            max_val, max_index = torch.max(soft_one_hot, dim=-1, keepdim=True)
            hard_one_hot = torch.zeros_like(soft_one_hot)
            hard_one_hot[soft_one_hot == max_val] = 1  # [b]
            mix_one_hot = hard_one_hot * 0.5 + soft_one_hot * 0.5

            # Embedding of the next token
            pred_embed = torch.matmul(mix_one_hot, self.embed_matrix)
            nwm_embed = torch.cat([nwm_embed, pred_embed.unsqueeze(1)], dim=1)
            nwm_ids = torch.cat([nwm_ids, max_index], dim=1)

            attn_mask = torch.cat((attn_mask, torch.ones(b, 1).cuda()), dim=1)

        # truncate the prompt
        wm_embed = torch.cat([wm_embed[:, 0][:, None, :], wm_embed[:, prompt_len:]], dim=1)
        nwm_embed = torch.cat([nwm_embed[:, 0][:, None, :], nwm_embed[:, prompt_len:]], dim=1)
        wm_ids = torch.cat([wm_ids[:, 0][:, None], wm_ids[:, prompt_len:]], dim=1)
        nwm_ids = torch.cat([nwm_ids[:, 0][:, None], nwm_ids[:, prompt_len:]], dim=1)

        return (wm_ids, wm_embed), (nwm_ids, nwm_embed)

    def get_online_prompt(self, tgt_sen, prompt_type='pp'):
        tgt_ids, tgt_embed = tgt_sen
        # prompt_type = 'pp' or 'se'
        if prompt_type == 'pp':
            s_ids, s_embed = self.pp_ids_start, self.pp_embed_start
            e_ids, e_embed = self.pp_ids_end, self.pp_embed_end
        elif prompt_type == 'se':
            s_ids, s_embed = self.se_ids_start, self.se_embed_start
            e_ids, e_embed = self.se_ids_end, self.se_embed_end
        bos_idx = self.get_bos_idx(tgt_ids, self.tokenizer.bos_token_id)
        p_embed_list, p_ids_list = [], []
        for b in range(tgt_ids.size(0)):
            b_embed = torch.cat([tgt_embed[b, :bos_idx[b] + 1], s_embed, tgt_embed[b, bos_idx[b] + 1:], e_embed], dim=0)
            p_embed_list.append(b_embed)
            b_ids = torch.cat(
                [tgt_ids[b, :bos_idx[b] + 1], s_ids.squeeze(), tgt_ids[b, bos_idx[b] + 1:], e_ids.squeeze()],
                dim=0)
            p_ids_list.append(b_ids)
        p_embed = torch.stack(p_embed_list)
        p_ids = torch.stack(p_ids_list)
        p_attn_mask = (p_ids != self.tokenizer.pad_token_id).long()
        return p_attn_mask, p_embed, p_ids

    def online_pp_forward(self, input_embed, attn_mask):
        # device = input_embed.device
        b = input_embed.shape[0]

        # embed_matrix = model.get_input_embeddings().weight

        output_embed = torch.empty((b, 0, self.embed_matrix.shape[1]), dtype=torch.float).cuda()
        output_id = torch.empty((b, 0), dtype=torch.int).cuda()

        for step in range(self.max_gen_len):
            if step == self.max_gen_len - 1:
                logit = self.llm(inputs_embeds=input_embed, attention_mask=attn_mask).logits[:, -1, :].squeeze(dim=1)
            else:
                with torch.no_grad():
                    logit = self.llm(inputs_embeds=input_embed, attention_mask=attn_mask).logits[:, -1, :].squeeze(
                        dim=1)

            # top-p sampling
            # prob = F.softmax(last_logits_no_wm, dim=-1)
            # sorted_prob, sorted_indices = torch.sort(prob, descending=True)
            # cum_prob = torch.cumsum(sorted_prob, dim=-1)
            # idx_to_rm = cum_prob > 0.8
            # for b_id in range(b):
            #     prob[b_id][sorted_indices[b_id][idx_to_rm[b_id]]] = 0

            # soft_one_hot = F.gumbel_softmax(F.log_softmax(logit / 0.6, dim=-1), tau=0.6, hard=False, dim=-1).to(
            #     embed_matrix.dtype)
            soft_one_hot = F.softmax(logit, dim=-1).to(self.embed_matrix.dtype)
            max_val, max_index = torch.max(soft_one_hot, dim=-1, keepdim=True)
            hard_one_hot = torch.zeros_like(soft_one_hot)
            hard_one_hot[soft_one_hot == max_val] = 1
            mix_one_hot = hard_one_hot * 0.5 + soft_one_hot * 0.5

            embed = torch.matmul(mix_one_hot, self.embed_matrix)
            input_embed = torch.cat([input_embed, embed.unsqueeze(1)], dim=1)

            output_embed = torch.cat([output_embed, embed.unsqueeze(1)], dim=1)
            _, max_idx = torch.max(logit, dim=-1, keepdim=True)
            output_id = torch.cat([output_id, max_idx], dim=1)
            attn_mask = torch.cat((attn_mask, torch.ones(b, 1).cuda()), dim=1)

        return output_embed, output_id

    def paraphrase_forward(self, sen):
        sen_attn_mask, sen_embed, _ = self.get_online_prompt(sen, 'pp')
        pp_embed, pp_ids = self.online_pp_forward(sen_embed, sen_attn_mask)
        # cat bos
        sen_ids = sen[0]
        bos_ids = self.tokenizer.bos_token_id * torch.ones_like(sen_ids[:, :1])
        pp_ids = torch.cat([bos_ids, pp_ids.squeeze(1)], dim=1)
        bos_embed = self.llm.get_input_embeddings()(bos_ids).cuda()  # .to(llm.device)
        pp_embed = torch.cat([bos_embed, pp_embed], dim=1)
        return pp_ids, pp_embed

    def get_semantic_metric(self, sen1, sen2, return_type):
        sen1_attn_mask, sen1_embed_wm, sen1_ids = self.get_online_prompt(sen1, 'se')
        sen2_attn_mask, sen2_embed_nwm, sen2_ids = self.get_online_prompt(sen2, 'se')

        if return_type == 'similarity':
            # get the last feature of llm
            with torch.no_grad():
                sen1_se = self.llm(inputs_embeds=sen1_embed_wm, attention_mask=sen1_attn_mask,
                                   output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                sen2_se = self.llm(inputs_embeds=sen2_embed_nwm, attention_mask=sen2_attn_mask,
                                   output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            sim = self.cos_sim(sen1_se, sen2_se)
            return sim
        elif return_type == 'loss':
            sen1_se = self.llm(inputs_embeds=sen1_embed_wm, attention_mask=sen1_attn_mask,
                               output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            sen2_se = self.llm(inputs_embeds=sen2_embed_nwm, attention_mask=sen2_attn_mask,
                               output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            se_loss = - torch.mean(self.cos_sim(sen1_se, sen2_se))
            return se_loss

    def train(self):
        for epoch in range(cfg.epochs):
            logger = defaultdict(AverageMeter)

            for step, batch in tqdm(enumerate(self.dl)):
                self.enc_optim.zero_grad()
                self.dec_optim.zero_grad()

                input_ids = batch['input_ids'].cuda()
                prompt_len = batch['prompt_len'][0]
                att_masks = (input_ids != self.tokenizer.pad_token_id).long()

                with autocast():  # mixed precision training
                    # generate watermark ans non-watermark sentences
                    wm_sen, nwm_sen = self.llm_forward(input_ids, att_masks, prompt_len)

                    # compute semantic_loss
                    semantic_loss = torch.tensor(0.0).cuda()
                    if self.step >= self.start_semantic_loss_step:
                        semantic_loss = self.get_semantic_metric(wm_sen, nwm_sen, 'loss')

                    # paraphrasing
                    wm_embed, nwm_embed = wm_sen[1], nwm_sen[1]
                    if torch.rand(1) > 0.5 and self.step >= self.start_noise_step:
                        pp_sen = self.paraphrase_forward(wm_sen)
                        semantic_sim = self.get_semantic_metric(pp_sen, wm_sen, 'similarity')
                        valid_rw = semantic_sim > 0.9
                        # get rw_embed with valid rw and replace wm_embed if semantic_sim <= 0.9, using scatter
                        rw_embed = pp_sen[1]
                        valid_rw = valid_rw.unsqueeze(1).unsqueeze(2).expand(-1, wm_embed.size(1), wm_embed.size(2))
                        adapt_rw_embed = torch.where(valid_rw, rw_embed, wm_embed)
                        dec_input = torch.cat([adapt_rw_embed, nwm_embed], dim=0)
                    else:
                        dec_input = torch.cat([wm_embed, nwm_embed], dim=0)

                    # create watermark labels
                    dec_gt = torch.ones(dec_input.size(0)).cuda()
                    dec_gt[-nwm_embed.size(0):] = 0
                    # watermark detection
                    dec_pred = self.dec(dec_input)
                    # compute detection loss
                    detect_loss = self.ce_loss_w * self.ce_loss(dec_pred[:, 0], dec_gt)

                if self.step >= self.start_semantic_loss_step:
                    # Pareto
                    self.scaler.scale(detect_loss).backward(retain_graph=True)
                    vec_d = []
                    for param in itertools.chain(self.enc.parameters()):
                        vec_d.append(param.grad.view(-1))
                    vec_d = torch.cat(vec_d)

                    self.enc_optim.zero_grad()
                    self.scaler.scale(semantic_loss).backward(retain_graph=True)
                    vec_s = []
                    for param in itertools.chain(self.enc.parameters()):
                        vec_s.append(param.grad.view(-1))
                    vec_s = torch.cat(vec_s)

                    # Multiple-Gradient Descent Algorithm
                    if torch.dot(vec_d, vec_s) >= torch.dot(vec_d, vec_d):
                        factor = 1.0
                    elif torch.dot(vec_d, vec_s) >= torch.dot(vec_s, vec_s):
                        factor = 0.0
                    else:
                        factor = (torch.dot(vec_s - vec_d, vec_s) / torch.dot(vec_s - vec_d, vec_s - vec_d)).item()

                    factor = min(factor, 0.01)  # ensure the weight for L_D is not too high

                    vec = factor * vec_d + (1 - factor) * vec_s

                    # Assign the gradients from MGDA
                    grad_position = 0
                    for param in itertools.chain(self.enc.parameters()):
                        param_numel = param.numel()
                        param_grad = vec[grad_position:grad_position + param_numel]
                        param_grad = param_grad.view_as(param)
                        param.grad = param_grad
                        grad_position += param_numel

                else:
                    total_loss = detect_loss
                    self.scaler.scale(total_loss).backward()

                # step
                self.scaler.step(self.enc_optim)
                self.scaler.update()
                self.scaler.step(self.dec_optim)
                self.scaler.update()

                # compute accuracy
                with torch.no_grad():
                    pred = nn.Sigmoid()(dec_pred).squeeze() > 0.5
                    acc = torch.mean((pred == dec_gt).float())

                # log
                losses = {
                    'Semantic Sim': -semantic_loss.item(),
                    'Detection Loss': detect_loss.item(),
                    'ACC': acc.item()}
                for name, loss in losses.items():
                    logger[name].update(loss)

                self.step += 1

                if self.step % self.print_log_fq == 0:
                    print(f"Step {self.step} | Semantic Sim: {logger['Semantic Sim'].avg} | "
                          f"Detection Loss: {logger['Detection Loss'].avg} | ACC: {logger['ACC'].avg}")
                if self.step % self.save_ckpt_fq == 0:
                    self.save_model()

    def save_model(self):
        # save model
        ckpt_path = op.join(self.save_ckpt_dir, f'{self.step}.pth')
        ckpt = {
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict(),
            'enc_opt': self.enc_optim.state_dict(),
            'dec_opt': self.dec_optim.state_dict(),
        }
        torch.save(ckpt, ckpt_path)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
