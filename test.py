import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["OPENAI_API_KEY"] = ''
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from evaluation.pipelines.detection import (
    WatermarkedTextDetectionPipeline, \
    UnWatermarkedTextDetectionPipeline, \
    DetectionPipelineReturnType)
from evaluation.dataset import C4Dataset
from evaluation.tools.text_editor import TruncatePromptTextEditor
from utils.acc_utils import get_llm, get_attack, print_det_result
from utils.quality_utils import assess_quality
from utils.transformers_config import TransformersConfig
from watermark.e2e.e2e import E2E

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(llm_name='opt-1.3b', assess_type='det', assess_name='no_attack', ds_len=-1):
    """
        Test the LLM with given assessment type and assessment name.

        Parameters:
            llm_name: The name of the LLM. 'opt-1.3b', 'Llama-2-7b-hf'
            assess_type: The type of assessment, 'det' for detection, 'qlt' for quality.
            assess_name: The name of the assessment.
                For detection, 'no_attack', 'context_substitute', 'paraphrase_dipper'.
                For quality, 'PPL', 'Log Diversity', 'BLEU', 'pass@1'.
            ds_len: The length of the dataset. If -1, the whole dataset is used.
    """
    print('llm_name:', llm_name)
    print('assess_type:', assess_type)
    print('assess_name:', assess_name)
    print('ds_len:', ds_len)

    trans_cfg = get_trans_cfg(assess_name, assess_type, llm_name)

    wm_scheme = E2E(trans_cfg, ckpt='ckpt/35000.pth')

    with torch.no_grad():
        if assess_type == 'det':
            ds = C4Dataset('dataset/c4/processed_c4.json', sample_n=ds_len)
            attack = get_attack(assess_name)
            if attack is None:
                text_editor_list = [TruncatePromptTextEditor()]
            else:
                text_editor_list = [TruncatePromptTextEditor(), attack]
            wm_result = WatermarkedTextDetectionPipeline(
                dataset=ds,
                text_editor_list=text_editor_list,
                show_progress=True,
                return_type=DetectionPipelineReturnType.SCORES).evaluate(wm_scheme)
            nwm_result = UnWatermarkedTextDetectionPipeline(
                dataset=ds,
                text_editor_list=[],
                show_progress=True,
                text_source_mode='natural',
                return_type=DetectionPipelineReturnType.SCORES).evaluate(wm_scheme)
            print_det_result(nwm_result, wm_result)
            del attack

        elif assess_type == 'qlt':
            pipeline = assess_quality(assess_name, sample_n=ds_len)
            result = pipeline.evaluate(wm_scheme)
            print(result)
            del pipeline

    del trans_cfg


def get_trans_cfg(assess_name, assess_type, llm_name):
    if assess_type == 'qlt' and assess_name == 'BLEU':
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            src_lang="deu_Latn")
        trans_cfg = TransformersConfig(
            model=AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/nllb-200-distilled-600M",
                torch_dtype=torch.float16).cuda().eval(),
            tokenizer=tokenizer,
            device=device,
            vocab_size=256206,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"))
    elif assess_type == 'qlt' and assess_name == 'pass@1':
        tokenizer = AutoTokenizer.from_pretrained(
            "bigcode/starcoder")
        trans_cfg = TransformersConfig(
            model=AutoModelForCausalLM.from_pretrained(
                "bigcode/starcoder",
                torch_dtype=torch.float16).cuda().eval(),
            tokenizer=tokenizer,
            device=device,
            min_length=200,
            max_length=400,
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        tokenizer, llm, vocab_size = get_llm(llm_name)
        trans_cfg = TransformersConfig(
            model=llm,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            device='cuda',
            max_new_tokens=200,
            do_sample=True,
            min_length=230,
            no_repeat_ngram_size=4,
            temperature=0.6,
        )
    return trans_cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, default='Llama-2-7b-hf')
    parser.add_argument('--assess_type', type=str, default='det')
    parser.add_argument('--assess_name', type=str, default='no_attack')
    parser.add_argument('--ds_len', type=int, default=-1)
    args = parser.parse_args()

    test(llm_name=args.llm_name,
         assess_type=args.assess_type,
         assess_name=args.assess_name,
         ds_len=args.ds_len)
