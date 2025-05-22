 # ==========================================================================
# assess_quality.py
# Description: Assess the impact on text quality of a watermarking algorithm
# ==========================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.dataset import C4Dataset, HumanEvalDataset, WMT16DE_ENDataset
from evaluation.pipelines.quality_analysis import DirectTextQualityAnalysisPipeline, QualityPipelineReturnType, \
    ReferencedTextQualityAnalysisPipeline
from evaluation.tools.text_editor import TruncatePromptTextEditor, TruncateTaskTextEditor, CodeGenerationTextEditor
from evaluation.tools.text_quality_analyzer import PPLCalculator, LogDiversityAnalyzer, BLEUCalculator, PassOrNotJudger

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def assess_quality(metric, sample_n=-1):
    if metric == 'PPL':
        my_dataset = C4Dataset('dataset/c4/processed_c4.json', sample_n)
        pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset,
                                                     watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                     unwatermarked_text_editor_list=[],
                                                     analyzer=PPLCalculator(
                                                         model=AutoModelForCausalLM.from_pretrained(
                                                             'meta-llama/Llama-2-13b-hf',
                                                             torch_dtype=torch.float16).cuda().eval(),
                                                         tokenizer=AutoTokenizer.from_pretrained(
                                                             'meta-llama/Llama-2-13b-hf'),
                                                         device=device),
                                                     unwatermarked_text_source='natural', show_progress=True,
                                                     # 'natural', 'generated'
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)
    elif metric == 'Log Diversity':
        my_dataset = C4Dataset('dataset/c4/processed_c4.json', sample_n)
        pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset,
                                                     watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                     unwatermarked_text_editor_list=[],
                                                     analyzer=LogDiversityAnalyzer(),
                                                     unwatermarked_text_source='natural', show_progress=True,
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)
    elif metric == 'BLEU':
        my_dataset = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
        pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset,
                                                         watermarked_text_editor_list=[],
                                                         unwatermarked_text_editor_list=[],
                                                         analyzer=BLEUCalculator(),
                                                         unwatermarked_text_source='generated', show_progress=True,
                                                         return_type=QualityPipelineReturnType.MEAN_SCORES)
    elif metric == 'pass@1':
        my_dataset = HumanEvalDataset('dataset/human_eval/test.jsonl')
        pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset,
                                                         watermarked_text_editor_list=[TruncateTaskTextEditor(),
                                                                                       CodeGenerationTextEditor()],
                                                         unwatermarked_text_editor_list=[TruncateTaskTextEditor(),
                                                                                         CodeGenerationTextEditor()],
                                                         analyzer=PassOrNotJudger(),
                                                         unwatermarked_text_source='generated', show_progress=True,
                                                         return_type=QualityPipelineReturnType.MEAN_SCORES)
    else:
        raise ValueError('Invalid metric')

    return pipeline
