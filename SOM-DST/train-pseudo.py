"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from model import SomDST
from transformers import BertPreTrainedModel, BertModel
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertConfig
from utils.data_utils import prepare_dataset, MultiWozDataset
from utils.data_utils import OP_SET, make_turn_label, postprocessing
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
from utils.model_eval import model_evaluation

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import argparse
import random
from tqdm import tqdm, trange

import json
import time
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss


def main(args):
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)
        
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        
     # logger
    logger_file_name = "logging"
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, "%s.txt"%(logger_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)

    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    print(n_gpu)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    rng = random.Random(args.random_seed)
    torch.manual_seed(args.random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    ontology = json.load(open(args.ontology_data))
    slot_meta = list(ontology.keys())
    logger.info(slot_meta)
    
    op2id = OP_SET[args.op_code]
    print(op2id)
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    train_data_raw = prepare_dataset(data_path=args.train_data_path,
                                     tokenizer=tokenizer,
                                     slot_meta=slot_meta,
                                     n_history=args.n_history,
                                     max_seq_length=args.max_seq_length,
                                     op_code=args.op_code,
                                     use_pseudo=True)

    train_data = MultiWozDataset(train_data_raw,
                                 tokenizer,
                                 slot_meta,
                                 args.max_seq_length,
                                 rng,
                                 ontology,
                                 args.word_dropout,
                                 args.shuffle_state,
                                 args.shuffle_p)
    print("# train examples %d" % len(train_data_raw))

    dev_data_raw = prepare_dataset(data_path=args.dev_data_path,
                                   tokenizer=tokenizer,
                                   slot_meta=slot_meta,
                                   n_history=args.n_history,
                                   max_seq_length=args.max_seq_length,
                                   op_code=args.op_code)
    print("# dev examples %d" % len(dev_data_raw))

    test_data_raw = prepare_dataset(data_path=args.test_data_path,
                                    tokenizer=tokenizer,
                                    slot_meta=slot_meta,
                                    n_history=args.n_history,
                                    max_seq_length=args.max_seq_length,
                                    op_code=args.op_code)
    print("# test examples %d" % len(test_data_raw))

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    model_config.hidden_dropout_prob = args.hidden_dropout_prob

    model = SomDST(model_config, len(op2id), op2id['update'])

    # re-initialize added special tokens ([SLOT], [NULL], [EOS])
    model.encoder.bert.embeddings.word_embeddings.weight.data[1].normal_(mean=0.0, std=0.02)
    model.encoder.bert.embeddings.word_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
    model.encoder.bert.embeddings.word_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)
    model.to(device)

    num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs)
    logger.info("***** Running training *****")
    logger.info(" Num examples = %d", len(train_data_raw))
    logger.info(" Batch size = %d", args.batch_size)
    logger.info(" Num steps = %d", num_train_steps)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    enc_param_optimizer = list(model.encoder.named_parameters())
    enc_optimizer_grouped_parameters = [
        {'params': [p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=args.enc_lr)
    enc_scheduler = get_linear_schedule_with_warmup(enc_optimizer, int(num_train_steps * args.enc_warmup),
                                         num_train_steps)

    dec_param_optimizer = list(model.decoder.parameters())
    dec_optimizer = AdamW(dec_param_optimizer, lr=args.dec_lr)
    dec_scheduler = get_linear_schedule_with_warmup(dec_optimizer, int(num_train_steps * args.dec_warmup),
                                                    num_train_steps)
    
    logger.info(enc_optimizer)
    logger.info(dec_optimizer)

#     if n_gpu > 1:
#         model = torch.nn.DataParallel(model)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=train_data.collate_fn,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)

    loss_fnc = nn.CrossEntropyLoss()
    best_score = {'epoch': 0, 'joint_acc': 0, 'op_acc': 0, 'final_slot_f1': 0}
    logger.info("Training...")
    for epoch in trange(int(args.n_epochs), desc="Epoch"):
        batch_loss = []
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
#             if step > 2:
#                 break
            batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
            input_ids, input_mask, segment_ids, state_position_ids, op_ids, gen_ids, max_value, max_update, \
            op_ids_pseudo, gen_ids_pseudo, max_value_pseudo, max_update_pseudo = batch

            if rng.random() < args.decoder_teacher_forcing:  # teacher forcing
                teacher = gen_ids
            else:
                teacher = None

            model_output_all = model(input_ids=input_ids,
                                        token_type_ids=segment_ids,
                                        state_positions=state_position_ids,
                                        attention_mask=input_mask,
                                        max_value=max_value,
                                        op_ids=op_ids,
                                        max_update=max_update,
                                        teacher=teacher,
                                        use_pseudo=True)
            
            state_scores, gen_scores, state_output, sequence_output, pooled_output = model_output_all
            
            # need to generate gen_scores for the pseudo labels
            gathered_pseudo = []
            for b, a in zip(state_output, op_ids_pseudo.eq(op2id['update'])):  # update
                if a.sum().item() != 0:
                    v = b.masked_select(a.unsqueeze(-1)).view(1, -1, model_config.hidden_size)
                    n = v.size(1)
                    gap = max_update_pseudo - n
                    if gap > 0:
                        zeros = torch.zeros(1, 1*gap, model_config.hidden_size, device=input_ids.device)
                        v = torch.cat([v, zeros], 1)
                else:
                    v = torch.zeros(1, max_update_pseudo, model_config.hidden_size, device=input_ids.device)
                gathered_pseudo.append(v)
            decoder_inputs = torch.cat(gathered_pseudo) # batch, max_update, hidden
            ###########
            if teacher is not None:
                teacher = gen_ids_pseudo
            gen_scores_pseudo = model.decoder(input_ids, decoder_inputs, sequence_output,
                                              pooled_output, max_value_pseudo, teacher)

            loss_s = loss_fnc(state_scores.view(-1, len(op2id)), op_ids.view(-1))
            loss_g = masked_cross_entropy_for_value(gen_scores.contiguous(),
                                                    gen_ids.contiguous(),
                                                    tokenizer.vocab['[PAD]'])
            loss_gt = loss_s + loss_g
            
            loss_s_p = loss_fnc(state_scores.view(-1, len(op2id)), op_ids_pseudo.view(-1))
            loss_g_p = masked_cross_entropy_for_value(gen_scores_pseudo.contiguous(),
                                                      gen_ids_pseudo.contiguous(),
                                                      tokenizer.vocab['[PAD]'])
            loss_p = loss_s_p + loss_g_p
            
            loss = (1 - args.alpha) * loss_gt + args.alpha * loss_p
            
            batch_loss.append(loss.item())

            loss.backward()
            enc_optimizer.step()
            enc_scheduler.step()
            dec_optimizer.step()
            dec_scheduler.step()
            model.zero_grad()

            if step % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, gen_loss : %.3f" \
                      % (epoch+1, args.n_epochs, step,
                         len(train_dataloader), np.mean(batch_loss), loss_s.item(), loss_g.item()))
                batch_loss = []

        if (epoch+1) % args.eval_epoch == 0:
            eval_res = model_evaluation(model, dev_data_raw, tokenizer, slot_meta, epoch+1, device, args.op_code)
            eval_res_test = model_evaluation(model, test_data_raw, tokenizer, slot_meta, epoch+1, device, args.op_code)
            logger.info(eval_res_test)
            logger.info(eval_res)
            if eval_res['joint_acc'] > best_score['joint_acc']:
                best_score = eval_res
                model_to_save = model.module if hasattr(model, 'module') else model
                save_path = os.path.join(args.save_dir, 'model_best.bin')
                torch.save(model_to_save.state_dict(), save_path)
            print("Epoch: %d, Test acc: %.6f, Val acc: %.6f, Best Score : %.6f"% (epoch+1, eval_res_test['joint_acc'], eval_res['joint_acc'], best_score['joint_acc']))
            print("\n")
            print(best_score)
            logger.info("Epoch: %d Test acc: %.6f Val acc: %.6f Best Score : %.6f"% (epoch+1, eval_res_test['joint_acc'], eval_res['joint_acc'], best_score['joint_acc']))
            logger.info(" epoch end ")
            

    print("Test using best model...")
    best_epoch = best_score['epoch']
    ckpt_path = os.path.join(args.save_dir, 'model_best.bin')
    model = SomDST(model_config, len(op2id), op2id['update'])
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    
    res = model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, device, args.op_code,
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=False)
    logger.info(res)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_root", default='data/mwz2.4', type=str)
    parser.add_argument("--train_data", default='train_dials_v2.json', type=str)
    parser.add_argument("--dev_data", default='dev_dials_v2.json', type=str)
    parser.add_argument("--test_data", default='test_dials_v2.json', type=str)
    parser.add_argument("--ontology_data", default='ontology-modified.json', type=str)
    parser.add_argument("--vocab_path", default='assets/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='assets/bert_config_base_uncased.json', type=str)
    parser.add_argument("--save_dir", default='outputs24', type=str)

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=30, type=int) #####
    parser.add_argument("--eval_epoch", default=1, type=int)

    parser.add_argument("--op_code", default="4", type=str)
    parser.add_argument("--slot_token", default="[SLOT]", type=str)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float)
    parser.add_argument("--decoder_teacher_forcing", default=0.5, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    parser.add_argument("--not_shuffle_state", default=False, action='store_true')
    parser.add_argument("--shuffle_p", default=0.5, type=float)

    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--msg", default=None, type=str)
    
    parser.add_argument("--alpha", default=0.4, type=float)

    args = parser.parse_args()
    args.train_data_path = os.path.join(args.data_root, args.train_data)
    args.dev_data_path = os.path.join(args.data_root, args.dev_data)
    args.test_data_path = os.path.join(args.data_root, args.test_data)
    args.ontology_data = os.path.join(args.data_root, args.ontology_data)
    args.shuffle_state = False if args.not_shuffle_state else True
    print('pytorch version: ', torch.__version__)
    print(args)
    main(args)
