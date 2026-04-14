import json
import os
import torch
from torch import LongTensor
from torch.utils.data import Dataset, ConcatDataset
from transformers import BertTokenizer


class FakeNewsDataset(Dataset):
    def __init__(self, text_data, labels, generator_labels,sub_generator_labels):
        self.text_data = text_data  # dict: input_ids, token_type_ids, attention_mask
        self.labels = labels
        self.generator_labels = generator_labels
        self.sub_generator = sub_generator_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.text_data['input_ids'][idx],
            'token_type_ids': self.text_data['token_type_ids'][idx],
            'attention_mask': self.text_data['attention_mask'][idx],
            'label': self.labels[idx],
            'generator': self.generator_labels[idx],
            'sub_generator': self.sub_generator[idx],
            'domain_names': self.text_data['domain_names'][idx],
        }


def process_jsonl_file(jsonl_path, tokenizer, max_length=512):
    input_ids, token_type_ids, attention_mask = [], [], []
    labels, generator_names, domain_ids = [], [], []   # ★ 新增列表
    generator_name = extract_generator_name(jsonl_path)

    DOMAIN_MAP = {
        "cmv": 0,
        "eli5": 1,
        "roct": 2,
    }

    error_count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            line_num += 1
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: 文件 {jsonl_path} 第 {line_num} 行")
                print(f"错误内容: {line.strip()[:100]}")
                print(f"错误详情: {str(e)}")
                error_count += 1
                continue

            if 'article' not in obj or 'label' not in obj:
                print(f"警告: 文件 {jsonl_path} 第 {line_num} 行缺少必要字段")
                error_count += 1
                continue

            text = obj['article']
            if isinstance(text, list):
                text = " ".join([str(x) for x in text])
            elif not isinstance(text, str):
                text = str(text)
            text = text.replace('\n', '').replace('\r', '')


            domain_str = obj.get("domain", None)
            domain_id = DOMAIN_MAP.get(domain_str, -1)
            domain_ids.append(domain_id)

            encoded = tokenizer.encode_plus(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True
            )
            input_ids.append(encoded['input_ids'])
            token_type_ids.append(encoded['token_type_ids'])
            attention_mask.append(encoded['attention_mask'])

            label_str = obj['label'].lower()
            label = 1 if label_str == 'human' else 0
            labels.append(label)

            if label == 1:
                generator_names.append(f"{generator_name}_human")
            else:
                generator_names.append(generator_name)
            generator_names.append(generator_name)

    print(f"→ Loaded {jsonl_path}: {len(labels)} samples")
    if error_count > 0:
        print(f"  警告: 跳过 {error_count} 个错误行")

    if not input_ids:
        print(f"严重错误: {jsonl_path} 没有有效数据!")
        return None, None, None

    text_data = {
        'input_ids': LongTensor(input_ids),
        'token_type_ids': LongTensor(token_type_ids),
        'attention_mask': LongTensor(attention_mask),
        'domain_names': LongTensor(domain_ids),
    }
    return text_data, LongTensor(labels), generator_names




def extract_generator_name(path):
    filename = os.path.basename(path).lower()
    for name in ['bloom','davinci002', 'davinci003','flant5base','flant5large',
         'flant5small', 'flant5xl', 'flant5xxl', 'glm130b', 'gpt3.5turbo',
         'gptj6b', 'gptneox','llama13b', 'llama30b', 'llama65b', 'llama6b','gpt41','gpt42',
         'opt1.3b', 'opt125m', 'opt13b', 'opt2.7b','opt30b', 'opt350m','opt6.7b',
         'optiml30b', 'optimlmax1.3b', 't011b', 't03b','llama','grover4.1','cohere','dolly4.1','gpt24.1',
                 'glm130b','gpt4.1turbo','gpt2.1turbo','hc3','gpt2','llama',
                 'bigscience', "eleutherai","flant5","gpt5","llama","opt","human_test1","human_train","claude","deep1",
                 'gpt',]:
        if name in filename:
            return name
    raise ValueError(f"Unknown generator name in path: {path}")


def cache_dataset(jsonl_path, tokenizer, cache_dir='./cache', force=False):
    os.makedirs(cache_dir, exist_ok=True)
    gen_name = extract_generator_name(jsonl_path)
    cache_path = os.path.join(cache_dir, f"{gen_name}.pt")

    if os.path.exists(cache_path) and not force:
        print(f"[Cache exists] Skipping {jsonl_path}")
        return

    print(f"[Processing] {jsonl_path}")
    result = process_jsonl_file(jsonl_path, tokenizer)

    if result[0] is None:
        print(f"错误: 无法处理文件 {jsonl_path}, 跳过缓存")
        return

    text_data, labels, generator_names = result

    torch.save({
        'input_ids': text_data['input_ids'],
        'token_type_ids': text_data['token_type_ids'],
        'attention_mask': text_data['attention_mask'],
        'labels': labels,
        'generator_names': generator_names,
        'domain_names':text_data['domain_names']

    }, cache_path)
    print(f"[Cached] {cache_path}")

def load_cached_dataset(
    name,
    cache_dir='./cache',
    generator_label_map=None,
    sub_generator_label_map=None
    ):
    path = os.path.join(cache_dir, f"{name}.pt")
    data = torch.load(path)

    # 一级标签（大类）
    if generator_label_map is None:
        unique_generators = list(set(data['generator_names']))
        generator_label_map = {g: i for i, g in enumerate(sorted(unique_generators))}

    generator_ids = torch.LongTensor([generator_label_map[g] for g in data['generator_names']])

    # 二级标签（细分小类）
    if sub_generator_label_map is not None:
        sub_generator_ids = torch.LongTensor([
            sub_generator_label_map.get(g, -1)  # 不在 map 中 → -1
            for g in data['generator_names']
        ])
    else:
        sub_generator_ids = torch.LongTensor([-1] * len(data['generator_names']))

    label_counts = torch.bincount(data['labels'])
    pos_count = label_counts[1].item() if len(label_counts) > 1 else 0
    neg_count = label_counts[0].item() if len(label_counts) > 0 else 0
    print(f"[Loaded] {name} — Total: {len(data['labels'])} | Pos: {pos_count} | Neg: {neg_count}")

    return FakeNewsDataset(
        {
            'input_ids': data['input_ids'],
            'token_type_ids': data['token_type_ids'],
            'attention_mask': data['attention_mask'],
            'domain_names': data['domain_names'],  # ★ 把它加回来
        },
        data['labels'],
        generator_ids,
        sub_generator_ids,
    )


def build_combined_dataset(dataset_names, cache_dir='./cache', generator_label_map=None,sub_generator_label_map=None):
    datasets = [load_cached_dataset(name, cache_dir, generator_label_map,sub_generator_label_map) for name in dataset_names]
    return ConcatDataset(datasets)


def build_all_caches():
    tokenizer = BertTokenizer.from_pretrained("./bert-base-english", use_fast=True)
    jsonl_paths = [
        ('bloom', '-'),
        ('davinci002', '-'),
        ('davinci003', '-'),
        ('flant5base', '-'),
        ('flant5large', '-'),
        ('flant5small', '-'),
        ('flant5xl', '-'),
        ('flant5xxl', '-'), #(name,path)


    ]

    for name, path in jsonl_paths:

        if not os.path.exists(path):
            print(f"错误: 文件不存在 {path}, 跳过")
            continue

        cache_dataset(path, tokenizer, cache_dir='./cache')