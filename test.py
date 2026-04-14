from module.model import BertForAIHumanClassification
from module.data import load_cached_dataset,build_all_caches,build_combined_dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time

def evaluate(model, data_loader, device,dataset_name=""):
    model.eval()
    all_preds, all_labels = [], []
    start_time = time.time()
    with torch.no_grad():
        for batch in data_loader:

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['label'].to(device)

            logits = model.test_struct(input_ids, attention_mask, token_type_ids)
            preds  = torch.argmax(logits, dim=1)
            all_preds  += preds.cpu().tolist()
            all_labels += labels.cpu().tolist()


    acc       = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[Eval {dataset_name}] "
          f"Acc: {acc:.4f} | F1: {f1:.4f} | "
          f"Recall: {recall:.4f} | Pre: {precision:.4f} | ")

    print(f"[Eval] Time taken: {elapsed_time:.4f} seconds")
    return acc, f1


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# ==== 1. 路径配置 ====
# model_path = "/home/czp/AIGC copy/models_mlp/model_weights2_400.pth"
# config_path = "/home/czp/AIGC copy/models_mlp/model_weights2_400_config.pth"
model_path = "/home/czp/AIGC_copy/models_tacl-adv2/model_weights2_200.pth"
config_path = "/home/czp/AIGC_copy/models_tacl-adv2/model_weights2_200_config.pth"
build_all_caches()
generator_label_map = {'bloom': 0, 'davinci003': 1, 'flant5base': 2, 'opt2.7b': 3, 'grover': 4,
                           'bloom_human': 0, 'davinci003_human': 1, 'flant5base_human': 2, 'merge2': 0, 'merge3': 1,
                           'merge2_human': 0, 'merge3_human': 1,
                           'grover_human': 4, 'opt2.7b_human': 3}

test_dataset = load_cached_dataset(
        'grover',
        generator_label_map=generator_label_map
    )

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


config = {
    "pretrained_model": "./bert-base-english",
    "dropout": 0.1,
    "lambda_grl": 0.5
}
model = BertForAIHumanClassification(
    pretrained_model=config["pretrained_model"],
    dropout=config["dropout"],
    lambda_grl=config["lambda_grl"]
)

state_dict = torch.load(model_path,map_location=device)
model.load_state_dict(state_dict)
model.to(device)
acc, f1 = evaluate(model, test_loader, device=device, dataset_name="gpt2")