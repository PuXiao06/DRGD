import torch
import torch.nn as nn
from transformers import BertModel
from torch.autograd import Function
from IB import IB_Module
import torch.nn.functional as F
import os

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


def AdaIn(content, style, style_strength=0.4, eps=1e-5):

    content_std, content_mean = torch.std_mean(content, dim=-1, unbiased=False, keepdim=True)
    style_std, style_mean = torch.std_mean(style, dim=-1, unbiased=False, keepdim=True)
    normalized_content = (content - content_mean) / (content_std + eps)
    stylized_content = (normalized_content * style_std) + style_mean
    output = (1 - style_strength) * content + style_strength * stylized_content

    return output

class BertForAIHumanClassification(nn.Module):
    def __init__(self, pretrained_model='./bert-base-english', dropout=0.1, lambda_grl=0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.finetune(self.bert)
        # self.dropout = nn.Dropout(dropout)
        # self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 二分类

        # self.grl = GradientReversalLayer(lambda_grl)
        self.global_reduction = nn.Linear(768, 32)

        self.content_pro = nn.Linear(32, 32)
        self.domain_pro = nn.Linear(32, 32)


        self.CE = nn.Sequential()
        self.CE.add_module('cet_fc1', nn.Linear(32, 16))
        self.CE.add_module('cet_dropout', nn.Dropout(p=0.5))
        self.CE.add_module('cet_fc2', nn.Linear(16, 2))

        # domain classifier
        self.DE = nn.Sequential()
        self.DE.add_module('det_fc1', nn.Linear(32, 16))
        self.DE.add_module('det_dropout', nn.Dropout(p=0.5))
        self.DE.add_module('det_fc2', nn.Linear(16, 6))


        # === 使用 IB 模块 ===
        ib_kl_beta =1e-5
        ib_sample_size = 5
        self.ib_content = IB_Module(
            input_dim=768,
            hidden_dim=512,
            latent_dim=32,
            kl_beta=ib_kl_beta,
            sample_size=ib_sample_size
        )
        self.ib_domain = IB_Module(
            input_dim=768,
            hidden_dim=512,
            latent_dim=32,
            kl_beta=ib_kl_beta,
            sample_size=ib_sample_size
        )
        # ===========================================================


    def finetune(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def save_model(self, save_path,epoch,step):
        os.makedirs(save_path, exist_ok=True)


        torch.save(self.state_dict(), os.path.join(save_path, f"model_weights{epoch}_{step}.pth"))


        config = {
            "pretrained_model": "",  #  tokenizer path，
            "dropout": 0.1,
            "lambda_grl": 0.5,
            "tokenizer_path": ""  # path
        }
        torch.save(config, os.path.join(save_path,f"model_weights{epoch}_{step}_config.pth"))

    def adv_params(self):

        params = []
        for m in [self.content_pro, self.domain_pro, self.ib_content, self.ib_domain]:
            params += [p for p in m.parameters()]
        return params

    def test_struct(self, input_ids, attention_mask=None, token_type_ids=None):

        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token



            content, _, _ = self.ib_content(cls_output, training=False)
            content = content.squeeze(0)
            content = self.content_pro(content)
            # domain, _, _ = self.ib_domain(cls_output, training=False)
            # domain = self.domain_pro(domain[0])
            # sigma = content.std().item()
            # # noise = torch.randn_like(content) * 15.5 * sigma
            contentOut = self.CE(content)


            # content = self.global_reduction(cls_output)
            # domain = self.global_reduction(cls_output)
            # content = self.content_pro(content)
            # domain = self.domain_pro(domain)
            # # noise = torch.randn_like(content)
            # sigma = content.std().item()
            # noise = torch.randn_like(content) *1.5 * sigma
            # contentOut = self.CE(content)

        return contentOut

    def extract_features(self, input_ids, attention_mask, token_type_ids):

        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            cls_output = outputs.last_hidden_state[:, 0, :]



            content_samples, _, _ = self.ib_content(cls_output, training=False)
            domain_samples, _ , _ = self.ib_domain(cls_output, training=False)
            content = self.content_pro(content_samples[0])
            domain = self.domain_pro(domain_samples[0])


            # # MLP
            # content = self.global_reduction(cls_output)
            # domain = self.global_reduction(cls_output)
            # content = self.content_pro(content)
            # domain = self.domain_pro(domain)

            return content.cpu().numpy(),domain.cpu().numpy()


    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, generator_labels=None):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        device = cls_output.device
        if labels is not None:
            labels = labels.to(device)
        if generator_labels is not None:
            generator_labels = generator_labels.to(device)


        content_samples, mu_c, kl_c = self.ib_content(cls_output, training=self.training)
        domain_samples, mu_d, kl_d = self.ib_domain(cls_output, training=self.training)

        self.kl_content_loss = kl_c
        self.kl_domain_loss = kl_d

        content = content_samples.mean(dim=0)
        domain = domain_samples.mean(dim=0)
        content = self.content_pro(content)
        domain = self.domain_pro(domain)




        ai_indices = (labels == 0).nonzero(as_tuple=True)[0] if labels is not None else torch.tensor([],
                                                                                                     dtype=torch.long,
                                                                                                     device=device)
        human_indices = (labels == 1).nonzero(as_tuple=True)[0] if labels is not None else torch.tensor([],
                                                                                                        dtype=torch.long,
                                                                                                        device=device)

        index = torch.randperm(content.shape[0], device=device)
        domain_shuffle = domain[index]
        content_shuffle = content[index]

        domain_sum = domain.clone()

        if ai_indices.numel() > 0:
            domain_sum[ai_indices] = AdaIn(domain[ai_indices], content_shuffle[ai_indices])

        # content_sum = AdaIn(content, domain_shuffle)

        p = 0.3
        strength = 0.15
        mask = (torch.rand(content.shape[0], device=device) < p)
        content_sum = content.clone()
        idxs = mask.nonzero(as_tuple=True)[0]
        if idxs.numel() > 0:
            content_sum[idxs] = AdaIn(content[idxs], domain_shuffle[idxs], style_strength=strength)


        ai_perturb_pred = None
        human_perturb_pred = None


        if labels is not None and self.training:


            if ai_indices.numel() > 0:
                ai_content = content[ai_indices]


                ai_repeat = ai_content.repeat_interleave(domain_shuffle.shape[0], dim=0)
                perturb_repeat = domain_shuffle.repeat((ai_content.shape[0], 1))


                noise = torch.randn_like(ai_repeat) * 0.1


                if perturb_repeat.shape[0] < ai_repeat.shape[0]:
                    repeat_factor = ai_repeat.shape[0] // perturb_repeat.shape[0]
                    perturb_repeat = perturb_repeat.repeat(repeat_factor, 1)

                    if perturb_repeat.shape[0] < ai_repeat.shape[0]:
                        remaining = ai_repeat.shape[0] - perturb_repeat.shape[0]
                        perturb_repeat = torch.cat([perturb_repeat, perturb_repeat[:remaining]], dim=0)


                ai_perturb = AdaIn(ai_repeat + noise, perturb_repeat, style_strength=0.5)
                ai_perturb_pred = self.CE(ai_perturb)


            if human_indices.numel() > 0:
                human_content = content[human_indices]

                noise_human = torch.randn_like(human_content) * 0.02
                human_perturb = human_content + noise_human
                human_perturb_pred = self.CE(human_perturb)



        domainOut = self.DE(domain_sum)
        if generator_labels is not None and ai_indices.numel() > 0:
            pert_d = F.cross_entropy(domainOut[ai_indices], generator_labels[ai_indices])
        else:
            pert_d = torch.tensor(0.0, device=device, dtype=domainOut.dtype)

        contentOut = self.CE(content_sum)
        if labels is not None:
            pert_c = F.cross_entropy(contentOut, labels)
        else:
            pert_c = torch.tensor(0.0, device=device, dtype=contentOut.dtype)


        pure_c_logits = self.CE(content)
        if labels is not None:

            weights = torch.tensor([1.0,0.7], dtype=torch.float).to(device)

            pure_c = F.cross_entropy(pure_c_logits, labels, weight=weights)

        else:
            pure_c = torch.tensor(0.0, device=device, dtype=pure_c_logits.dtype)

        pure_d_logits = self.DE(domain)
        if generator_labels is not None and ai_indices.numel() > 0:
            pure_d = F.cross_entropy(pure_d_logits[ai_indices], generator_labels[ai_indices])
        else:
            pure_d = torch.tensor(0.0, device=device, dtype=pure_d_logits.dtype)


        pure_c_g_logits = self.CE(grad_reverse(domain, 0.5))
        pure_d_g_logits = self.DE(grad_reverse(content, 0.5))


        if ai_indices.numel() > 0:
            pure_c_g = F.cross_entropy(pure_c_g_logits[ai_indices], labels[ai_indices])  # AI/Human 标签
            if generator_labels is not None:
                pure_d_g = F.cross_entropy(pure_d_g_logits[ai_indices], generator_labels[ai_indices])
            else:
                pure_d_g = torch.tensor(0.0, device=device, dtype=pure_d_g_logits.dtype)
        else:
            pure_c_g = torch.tensor(0.0, device=device)
            pure_d_g = torch.tensor(0.0, device=device)



        loss_ai_perturb = 0
        loss_human_perturb = 0


        if ai_perturb_pred is not None and ai_perturb_pred.numel() > 0:

            ai_labels = torch.zeros(ai_perturb_pred.size(0), dtype=torch.long, device=device)
            loss_ai_perturb = F.cross_entropy(ai_perturb_pred, ai_labels)


        if human_perturb_pred is not None and human_perturb_pred.numel() > 0:

            human_labels = torch.ones(human_perturb_pred.size(0), dtype=torch.long, device=device)
            loss_human_perturb = F.cross_entropy(human_perturb_pred, human_labels)


        return pert_c, pert_d, pure_c,pure_d,pure_c_g,pure_d_g, kl_c, kl_d, loss_ai_perturb,loss_human_perturb
