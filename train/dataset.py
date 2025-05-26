from functools import partial

import torch
from datasets import load_dataset, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def tokenize_and_truncate(
        example: dict,
        input_col_name: str = "text",
        completion_length: int = None,
        prompt_length: int = None,
        hf_model_name: str = None,
        tokenizer=None,
        truncate_left=False,
        model_max_length=None,
):
    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"

    # tokenize
    inputs_ids = tokenizer(example[input_col_name], return_tensors="pt")["input_ids"]
    example.update({"untruncated_inputs": inputs_ids})

    # truncate
    desired_comp_len = (inputs_ids.shape[1] - 1) - prompt_length
    slice_length = desired_comp_len if desired_comp_len > 0 else 0

    inputs_ids = inputs_ids[:, : inputs_ids.shape[1] - slice_length]

    example.update({"input_ids": inputs_ids})
    return example


def preprocess_func(
        example: dict,
        tgt_prompt_len: int = None,
        tokenizer=None,
):
    # tokenize
    inputs_ids = tokenizer(example["text"], return_tensors="pt")["input_ids"]
    example.update({'full_input_ids': inputs_ids})

    # truncate
    desired_comp_len = (inputs_ids.shape[1] - 1) - tgt_prompt_len
    slice_length = desired_comp_len if desired_comp_len > 0 else 0
    inputs_ids = inputs_ids[:, : inputs_ids.shape[1] - slice_length]
    example.update({'input_ids': inputs_ids})

    # get len
    prompt_len = inputs_ids.shape[1]
    full_input_ids = example["full_input_ids"]
    ori_len = full_input_ids.shape[1]
    prompt_text = tokenizer.batch_decode(inputs_ids, skip_special_tokens=True)[0]
    left_len = ori_len - prompt_len

    example.update(
        {
            'prompt_text': prompt_text,
            'ori_len': ori_len,
            'prompt_len': prompt_len,
            'left_len': left_len,
        }
    )
    return example


def filter_func(
        example,
        tgt_ori_len=0,
        tgt_prompt_len=0,
        tgt_left_len=0,
):
    ori_len = example['ori_len']
    prompt_len = example['prompt_len']
    left_len = example['left_len']

    conds = all(
        [
            ori_len >= tgt_ori_len,
            prompt_len >= tgt_prompt_len,
            left_len >= tgt_left_len,
        ]
    )
    return conds


def get_data_loader(tokenizer, tgt_prompt_len, batch_size):
    raw_dataset = load_dataset(
        'wikitext',
        'wikitext-103-raw-v1',
        split='train',
        streaming=False  # iterate over the dataset directly without having to download the entire dataset.
    )

    def wikitext_generator():
        # the generator loop
        for ex in raw_dataset:
            yield ex

    dataset = IterableDataset.from_generator(wikitext_generator)

    # tokenize and truncate the row inputs to create prompts according to the strategy spec'd above

    preprocess_func_ = partial(preprocess_func, **dict(
        tokenizer=tokenizer,
        tgt_prompt_len=tgt_prompt_len,
    ))

    # map() allows you to apply a processing function to each example in a dataset, independently or in batches
    dataset = dataset.map(preprocess_func_, batched=False)

    filter_func_ = partial(filter_func, **dict(
        tgt_ori_len=100,
        tgt_prompt_len=tgt_prompt_len,
        tgt_left_len=100,
    ))
    # filter() returns rows that match a specified condition:
    dataset = dataset.filter(filter_func_, batched=False)

    # construct the collator
    def data_collator(batch):
        # Pad input_ids
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long).view(-1) for item in batch]

        # Reverse the sequences, pad them, and then reverse back
        input_ids_reversed = [torch.flip(tensor, [0]) for tensor in input_ids]  # Reverse each sequence
        # pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids_padded_reversed = pad_sequence(input_ids_reversed, batch_first=True,
                                                 padding_value=tokenizer.pad_token_id)
        input_ids_padded = torch.flip(input_ids_padded_reversed, [1])  # Reverse back to original order

        # Collate other data fields dynamically
        collated_batch = {'input_ids': input_ids_padded}
        for key in batch[0].keys():
            if key != 'input_ids':  # Assuming 'input_ids' is handled separately
                collated_batch[key] = [item[key] for item in batch]

        return collated_batch

    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    return train_loader


if __name__ == '__main__':
    from transformers import AutoTokenizer
    batch_size = 8
    min_prompt_tokens = 30
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side="left")

    dl = get_data_loader(tokenizer, min_prompt_tokens, batch_size)

    for batch in dl:
        input_ids = batch['input_ids'].cuda()
        print(batch)
