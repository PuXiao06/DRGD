from collections import Counter,defaultdict
import torch, os
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def evaluate(model, data_loader, device,name):
    model.eval()
    all_preds, all_labels = [], []
    per_gen = defaultdict(lambda: {"preds": [], "labels": []})       # 一级 generator
    per_subgen = defaultdict(lambda: {"preds": [], "labels": []})    # 二级 generator

    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['label'].to(device)
            gen_labels     = batch['generator'].to(device)       # 一级 generator
            subgen_labels  = batch['sub_generator'].to(device)   # 二级 generator

            logits = model.test_struct(input_ids, attention_mask, token_type_ids)
            preds  = torch.argmax(logits, dim=1)

            all_preds  += preds.cpu().tolist()
            all_labels += labels.cpu().tolist()

            # for g, p, l in zip(gen_labels.cpu().tolist(), preds.cpu().tolist(), labels.cpu().tolist()):
            #     per_gen[g]["preds"].append(p)
            #     per_gen[g]["labels"].append(l)


            # for sg, p, l in zip(subgen_labels.cpu().tolist(), preds.cpu().tolist(), labels.cpu().tolist()):
            #     if sg >= 0:
            #         per_subgen[sg]["preds"].append(p)
            #         per_subgen[sg]["labels"].append(l)


    acc       = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)



    print(f"[Eval Overall]--{name} Acc: {acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")
    # print(f"-----[Eval Class-wise]-----")
    # print(f"  AI (label=0)     → Precision: {ai_prec:.4f} | Recall: {ai_rec:.4f}")
    # print(f"  Human (label=1)  → Precision: {human_prec:.4f} | Recall: {human_rec:.4f}")


    # for g, data in per_gen.items():
    #     if len(set(data["labels"])) > 1:
    #         acc_g = accuracy_score(data["labels"], data["preds"])
    #         f1_g  = f1_score(data["labels"], data["preds"], zero_division=0)
    #         print(f"  [Gen {g}] Acc: {acc_g:.4f} | F1: {f1_g:.4f}")
    #     else:
    #         acc_g = accuracy_score(data["labels"], data["preds"])
    #         print(f"  [Gen {g}] Acc: {acc_g:.4f} | F1: N/A (single class)")


    # for sg, data in per_subgen.items():
    #     if len(set(data["labels"])) > 1:
    #         acc_sg = accuracy_score(data["labels"], data["preds"])
    #         f1_sg  = f1_score(data["labels"], data["preds"], zero_division=0)
    #         print(f"    [SubGen {sg}] Acc: {acc_sg:.4f} | F1: {f1_sg:.4f}")
    #     else:
    #         acc_sg = accuracy_score(data["labels"], data["preds"])
    #         print(f"    [SubGen {sg}] Acc: {acc_sg:.4f} | F1: N/A (single class)")

    return acc