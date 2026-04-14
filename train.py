from torch.utils.data import DataLoader, ConcatDataset
from transformers import AdamW

from tqdm import tqdm
import torch, os
import numpy as np
from module.model import BertForAIHumanClassification
import torch.nn as nn
from module.data import load_cached_dataset, build_all_caches, build_combined_dataset
from tools.eval import evaluate
from scipy.stats import wasserstein_distance
from collections import Counter,defaultdict

def compute_wasserstein_distance(content_feats, domain_feats):
    # content_feats, domain_feats: [N, D] numpy arrays
    distances = []
    for i in range(content_feats.shape[1]):
        dist = wasserstein_distance(content_feats[:, i], domain_feats[:, i])
        distances.append(dist)
    return np.mean(distances)



def train(model, train_dataset, test_dataset_1, test_dataset_2, test_dataset_3,device, epochs=5, batch_size=16, lr=2e-5, log_every=10):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader_1 = DataLoader(test_dataset_1, batch_size=batch_size,shuffle=True)
    val_loader_2 = DataLoader(test_dataset_2, batch_size=batch_size)
    val_loader_3 = DataLoader(test_dataset_3, batch_size=batch_size)
    # val_loader_4 = DataLoader(test_dataset_4, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    params_share = model.adv_params()
    optimizer_share = AdamW(params_share, lr=lr)
    optimizer = AdamW(model.parameters() , lr=lr)

    best_acc1 = 0.0
    best_acc2 = 0.0
    best_acc3 = 0.0
    model.to(device)

    best_acc_1, best_f1_1 = 0, 0


    for epoch in range(epochs):
        print(f"\n Epoch {epoch + 1}/{epochs} start...")
        model.train()

        all_preds_c, all_labels_c, all_preds_g, all_labels_g = [], [], [], []



        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            generator_labels = batch['generator'].to(device)


            optimizer.zero_grad()
            pert_c, pert_d, pure_c, pure_d, pure_c_g, pure_d_g, kl_c, kl_d, ai_perturb, human_perturb = model(
                input_ids, attention_mask, token_type_ids, labels, generator_labels
            )


            # ===  phase 1 s  ===
            loss1 = pert_c

            loss2 = pert_d
            loss3 = 0.5 * (kl_c + kl_d)

            loss4 = ai_perturb
            loss = loss1 + loss2 + loss3 + loss4 + 0.8*human_perturb

            loss.backward(retain_graph=True)

            optimizer.step()


            #  ===  phase 2  ===
            pure1 = pure_c

            pure2 = pure_d
            pure3 = pure_c_g

            pure4 = pure_d_g
            pure5 = 0.5 * (kl_c + kl_d)

            optimizer_share.zero_grad()
            pure_loss = pure1 + pure2 + pure3 + pure4 + pure5

            pure_loss.backward()
            optimizer_share.step()

            if (step + 1) % log_every == 0 or (step + 1) == len(train_loader):

              print(f" Step {step + 1}/{len(train_loader)} | Batch Loss_1: {loss.item():.4f}")



            if step % 10 == 0 and step != 0:
                val_acc1 = evaluate(model, val_loader_1, device,'flant5base')
                val_acc2 = evaluate(model, val_loader_2, device,'opt2.7b')
                val_acc3 = evaluate(model, val_loader_3, device,'llama65b')
                best_acc1 = max(best_acc1, val_acc1)
                best_acc2 = max(best_acc2, val_acc2)
                best_acc3 = max(best_acc3, val_acc3)
        print(f"✅--Epoch {epoch + 1} | flant5 | Best Val Acc: {best_acc1:.4f}")
        print(f"✅--Epoch {epoch + 1} | opt | Best Val Acc: {best_acc2:.4f}")
        print(f"✅--Epoch {epoch + 1} | llama | Best Val Acc: {best_acc3:.4f}")







if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print('start to build dataset....')

    build_all_caches()  # first

    # generator_label_map = {'bloom': 0, 'gpt2': 2, 'llama': 0, 'hc3':1,
    #                        'davinci002': 4, 'davinci003': 1, 'gpt3.5turbo': 4,
    #                        'flant5base': 1, 'flant5large': 1, 'flant5small': 1, 'flant5xl': 1, 'flant5xxl': 1,
    #                        't011b': 1, 't03b': 1,
    #                        'glm130b': 0,
    #                        'gptj6b': 6, 'gptneox': 6,
    #                        'llama13b': 4, 'llama30b': 2, 'llama65b': 2, 'llama6b': 4,
    #                        'opt1.3b': 0, 'opt125m': 0, 'opt13b': 0, 'opt2.7b': 2, 'opt30b': 3, 'opt350m': 0,
    #                        'opt6.7b': 0, 'optiml30b': 0, 'optimlmax1.3b': 4,
    #                        'gpt42': 4
    #                        }

    generator_label_map = {'bloom': 0,
                           'davinci002': 1, 'davinci003': 1, 'gpt3.5turbo': 1,
                           'flant5base': 3, 'flant5large': 2, 'flant5small': 2, 'flant5xl': 2,
                           'glm130b': 3,
                           'gptj6b': 2, 'gptneox': 4,
                           'llama13b': 5, 'llama30b': 5, 'llama65b': 3, 'llama6b': 5,
                           'opt2.7b': 3,'opt6.7b': 6, 'optiml30b': 6, 'optimlmax1.3b': 6,
                           'gpt5': 7,'qwen':7,'deep1':7,'wxyy4.5turob':7
                           }
    # train_dataset = build_combined_dataset(
    #     [
    #         'bloom',
    #             'davinci002', 'davinci003', 'gpt3.5turbo',
    #             'flant5base', 'flant5large', 'flant5small', 'flant5xl',
    #             'gptj6b', 'gptneox',
    #                 'llama13b', 'llama30b', 'llama65b', 'llama6b',
    #             'opt2.7b','glm130b',
    #         'opt6.7b', 'optiml30b', 'optimlmax1.3b',
    #     ],)
    train_dataset = build_combined_dataset(
        [
            'bloom',
            'davinci003',
            'gptj6b'
        ] )
    # generator_label_map = {
    #     "bigscience":0,
    #     "eleutherai":1,
    #     "flant5":2,
    #     "glm130b":6,
    #     "gpt":3,
    #     "llama":4,
    #     "opt":5,
    #     "human_test":6,
    #     "human_train":0,
    # }

    sub_generator_label_map = {
        "llama65b": 1
    }

    # train_dataset = build_combined_dataset(
    #     [
    #          "bigscience","eleutherai","flant5","gpt", "llama","opt","human_train",
    #     ],
    #     generator_label_map=generator_label_map,
    #     sub_generator_label_map=sub_generator_label_map
    # )
    test_dataset_1 = build_combined_dataset(
        [ "flant5base"],
        generator_label_map=generator_label_map,
        sub_generator_label_map=sub_generator_label_map
    )
    test_dataset_2 = build_combined_dataset(
        ["opt2.7b"],
        generator_label_map=generator_label_map,
        sub_generator_label_map=sub_generator_label_map
    )
    test_dataset_3 = build_combined_dataset(
        ["llama65b"],
        generator_label_map=generator_label_map,
        sub_generator_label_map=sub_generator_label_map
    )



    print(f"Train size: {len(train_dataset)} | Val size: {len(test_dataset_1)}")
    print(f"→ Generator mapping: {generator_label_map}")




    model = BertForAIHumanClassification()


    train(
        model=model,
        train_dataset=train_dataset,
        test_dataset_1=test_dataset_1,
        test_dataset_2=test_dataset_2,
        test_dataset_3=test_dataset_3,
        device=device,
        epochs=5,
        batch_size=16,
        lr=2e-5
    )