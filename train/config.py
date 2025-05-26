import datetime
import os
import os.path as op

gpus = '0'

# ============================== Optimize ============================

epochs = 10
lr = 1e-4
weight_decay = 1e-5
prompt_len = 30
batch_size = 8
start_semantic_loss_step = 10000
start_noise_step = 20000
init_step = 0

# ============================== Optimize ============================

# ================================ Model =============================

ce_loss_w = 10.0
max_gen_len = 100
llm_temp = 1.0
gss_tau = 0.1
top_k_logits = 20
wm_delta = 1.25
context_win_size = 10

# ================================ Model =============================


# =============================== Prompt =============================
# paraphrase prompt
pp_prompt_start = '[User] : Rewrite the following text, maintaining the original meaning. \n[Target Text] : '
pp_prompt_end = '\n[User] : Now start rewrite the above text. \n[Rewritten Text] : '
# semantic embed prompt
se_prompt_start = 'This sentence : "A jockey riding a horse." means in one word:"Equestrian".This sentence : "'
se_prompt_end = '" means in one word:"'
# =============================== Prompt =============================

print_log_fq = 100
save_ckpt_fq = 1000

# ================================ Path ==============================

ckpt_path = None
if ckpt_path is not None:
    init_step = int(ckpt_path.split('/')[-1].split('.')[0])
    print('init_step:', init_step)
root = 'your_exp_root'  # TODO:
date_now = datetime.datetime.now()
date_now = 'Log_v%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
exp_dir = op.join(root, 'exp_out', date_now)
ckpt_dir = op.join(exp_dir, 'ckpt')
os.makedirs(exp_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# ================================ Path ==============================
