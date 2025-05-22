from functools import partial

from torch.nn import functional as F
from transformers import LogitsProcessor, LogitsProcessorList, AutoTokenizer, OPTForCausalLM

from utils.transformers_config import TransformersConfig
from watermark.base import BaseWatermark
from watermark.e2e.model import *


class E2EConfig:
    def __init__(self, transformers_config: TransformersConfig, ckpt) -> None:
        # wm cfg
        self.delta = 1.25
        self.k = 20
        self.win_size = 10
        self.ckpt = ckpt
        self.train_llm = OPTForCausalLM.from_pretrained(
            "facebook/opt-1.3b",
            torch_dtype=torch.float16).cuda().eval()
        self.train_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-1.3b", padding_side="left")
        self.train_llm_dim = self.train_llm.config.hidden_size
        self.train_embed_matrix = self.train_llm.get_input_embeddings().weight.cuda()
        # llm cfg
        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs
        self.same_tokenizer = self.generation_tokenizer.vocab_size == self.train_tokenizer.vocab_size
        if self.same_tokenizer:
            self.prefix_length = self.win_size
        else:
            self.prefix_length = self.win_size * 3


class E2ELogitsProcessor(LogitsProcessor):
    def __init__(self, config):
        self.config = config
        self.enc = Enc(input_dim=self.config.train_llm_dim).cuda()
        model_ckpt = torch.load(self.config.ckpt)
        self.enc.load_state_dict(model_ckpt['enc'])
        self.enc.eval()
        self.embed_matrix = self.config.generation_model.get_input_embeddings().weight
        self.same_tokenizer = self.config.same_tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        previous_n = self.config.win_size
        if not self.same_tokenizer:
            previous_n = int(self.config.win_size * 3)

        # previous tokens
        previous_ids = input_ids[:, -(previous_n - 1):]
        previous_ids = previous_ids[:, None, :].repeat(1, self.config.k, 1)
        # top-k candidate tokens
        _, candidate_ids = torch.topk(scores, self.config.k, dim=-1)
        cat_ids = torch.cat([previous_ids, candidate_ids.unsqueeze(-1)], dim=-1)

        if self.same_tokenizer:
            enc_inputs = self.embed_matrix[cat_ids]
        else:
            # token ids to text
            cand_text = self.config.generation_tokenizer.batch_decode(cat_ids[0], skip_special_tokens=True)
            # text to token ids
            train_enc = self.config.train_tokenizer(cand_text, return_tensors="pt", add_special_tokens=False,
                                                    padding=True)
            train_ids = train_enc["input_ids"].cuda()
            train_ids = train_ids[None, :, -self.config.win_size:]
            enc_inputs = self.config.train_embed_matrix[train_ids]

        try:
            enc_logit = self.enc(enc_inputs)
            perturb_logit = torch.zeros_like(scores)
            perturb_logit.scatter_add_(1, candidate_ids, self.config.delta * enc_logit.to(perturb_logit.dtype))
            # perturb logit to dtype of scores
            scores = scores + perturb_logit.to(scores.dtype)
        except:
            pass
            print("Error in perturb_logit")
        return scores


class E2E(BaseWatermark):
    def __init__(self, transformers_config, ckpt):
        self.config = E2EConfig(transformers_config, ckpt)
        self.dec = Dec(input_dim=self.config.train_llm_dim).cuda()
        model_ckpt = torch.load(self.config.ckpt)
        self.dec.load_state_dict(model_ckpt['dec'])
        self.dec.eval()
        self.logits_processor = E2ELogitsProcessor(self.config)
        self.same_tokenizer = self.config.generation_tokenizer == self.config.train_tokenizer

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs
        )

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(
            self.config.device)
        # Generate watermarked text

        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)

        # Decode
        watermarked_text = \
            self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        enc_text = self.config.train_tokenizer(text, return_tensors="pt", add_special_tokens=False)
        enc_ids = enc_text["input_ids"].cuda()
        embed = self.config.train_embed_matrix[enc_ids]

        pred = self.dec(embed.float())
        logit = F.sigmoid(pred)
        # Determine if the z_score indicates a watermark
        is_watermarked = logit > 0.5
        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": logit.item()}
        else:
            return (is_watermarked, logit.item())
