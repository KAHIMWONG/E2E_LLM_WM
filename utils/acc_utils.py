from transformers import AutoTokenizer, OPTForCausalLM

from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from evaluation.tools.text_editor import *


def get_llm(llm_name):
    # opt-1.3b, Llama-2-7b-hf, Llama-2-7b-chat-hf
    if llm_name == 'opt-1.3b':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        llm = OPTForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16).cuda().eval()
        vocab_size = 50272
    elif llm_name == 'Llama-2-7b-hf':
        from transformers import LlamaTokenizer, LlamaForCausalLM
        tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        llm = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', torch_dtype=torch.float16).cuda().eval()
        vocab_size = tokenizer.vocab_size
    elif llm_name == 'Llama-2-7b-chat-hf':
        from transformers import LlamaTokenizer, LlamaForCausalLM
        tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
        llm = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype=torch.float16).cuda().eval()
        vocab_size = tokenizer.vocab_size
    return tokenizer, llm, vocab_size


def get_attack(attack_name):
    if attack_name == 'no_attack':
        return None
    if attack_name == 'delete':
        return WordDeletion(ratio=0.3)
    elif attack_name == 'substitute':
        return SynonymSubstitution(ratio=0.5)
    elif attack_name == 'context_substitute':
        return ContextAwareSynonymSubstitution(
            ratio=0.5,
            tokenizer=BertTokenizer.from_pretrained(
                'google-bert/bert-large-uncased'),
            model=BertForMaskedLM.from_pretrained(
                'google-bert/bert-large-uncased',
                torch_dtype=torch.float16).cuda().eval())
    elif attack_name == 'paraphrase_dipper':
        return DipperParaphraser(
            tokenizer=T5Tokenizer.from_pretrained('google/t5-v1_1-xxl'),
            model=T5ForConditionalGeneration.from_pretrained(
                '/home/jesonwong47/data/WLLM/SIR/dipper/para-paraphrase-ctx-t5-xxl',
                torch_dtype=torch.float16).cuda().eval(),
            lex_diversity=60, order_diversity=0, sent_interval=1, max_new_tokens=100,
            do_sample=True, top_p=0.75, top_k=None)


def print_det_result(nwm_result, wm_result):
    best_calc = DynamicThresholdSuccessRateCalculator(rule='best')
    fpr10_calc = DynamicThresholdSuccessRateCalculator(rule='target_fpr', target_fpr=0.1)
    fpr1_calc = DynamicThresholdSuccessRateCalculator(rule='target_fpr', target_fpr=0.01)
    best_result = best_calc.calculate(wm_result, nwm_result)
    fpr10_result = fpr10_calc.calculate(wm_result, nwm_result)
    fpr1_result = fpr1_calc.calculate(wm_result, nwm_result)
    print('best_result:', best_result)
    print('fpr10_result:', fpr10_result)
    print('fpr1_result:', fpr1_result)
