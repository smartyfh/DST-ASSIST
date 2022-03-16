"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import re
from copy import deepcopy

flatten = lambda x: [i for s in x for i in s]

OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}


def make_turn_label(slot_meta, last_dialog_state, turn_dialog_state,
                    tokenizer, op_code='4', dynamic=False):
    if dynamic:
        gold_state = turn_dialog_state
        turn_dialog_state = {}
        for x in gold_state:
            s = x.split('-')
            k = '-'.join(s[:2])
            turn_dialog_state[k] = s[2]

    op_labels = ['carryover'] * len(slot_meta)
    generate_y = []
    for idx, slot in enumerate(slot_meta):
        v = turn_dialog_state[slot]
        vv = last_dialog_state[slot]
        if vv != v:
            if v == 'do not care':
                op_labels[idx] = 'dontcare'
            elif v == 'none':
                op_labels[idx] = 'delete'
            else:
                op_labels[idx] = 'update'
                generate_y.append([tokenizer.tokenize(v) + ['[EOS]'], idx])
        else:
            op_labels[idx] = 'carryover'
            
    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
    if len(generate_y) > 0:
        generate_y = sorted(generate_y, key=lambda lst: lst[1])
        generate_y, _ = [list(e) for e in list(zip(*generate_y))]

    if dynamic:
        op2id = OP_SET[op_code]
        generate_y = [tokenizer.convert_tokens_to_ids(y) for y in generate_y]
        op_labels = [op2id[i] for i in op_labels]

    return op_labels, generate_y, gold_state


def postprocessing(slot_meta, ops, last_dialog_state,
                   generated, tokenizer, op_code, gold_gen={}):
    gid = 0
    for st, op in zip(slot_meta, ops):
        if op == 'dontcare':
            last_dialog_state[st] = 'do not care'
        elif op == 'delete':
            last_dialog_state[st] = 'none'
        elif op == 'update':
            g = tokenizer.convert_ids_to_tokens(generated[gid])
            gen = []
            for gg in g:
                if gg == '[EOS]':
                    break
                gen.append(gg)
            gen = ' '.join(gen).replace(' ##', '')
            gid += 1
            gen = gen.replace(' : ', ':').replace('##', '')
            last_dialog_state[st] = gen
            
    return generated, last_dialog_state


def prepare_dataset(data_path, tokenizer, slot_meta,
                    n_history, max_seq_length, diag_level=False, op_code='4', use_pseudo=False):
    if use_pseudo:
        if '2.0' in data_path:
            pred = json.load(open("pseudo_label_aux_20.json"))
        elif '2.4' in data_path:
            pred = json.load(open("pseudo_label_aux_24.json"))
        else:
            raise Exception("version not supported")
        
    dials = json.load(open(data_path))
    
    data = []
    dial_cnt = 0
    for dial_dict in dials:
        dial_cnt += 1
        if dial_cnt % 1000 == 0:
            print(dial_cnt)
        dialog_history = []
        last_dialog_state = {}
        last_dialog_state_pseudo = {}
        last_uttr = ""
        for slot in slot_meta:
            last_dialog_state[slot] = "none"
            last_dialog_state_pseudo[slot] = "none"    
                
        dialogue_idx = dial_dict["dialogue_idx"]
        for ti, turn in enumerate(dial_dict["dialogue"]):
            turn_idx = turn["turn_idx"]
            assert turn_idx == ti
            if (ti + 1) == len(dial_dict["dialogue"]):
                is_last_turn = True
            else:
                is_last_turn = False
            
            turn_uttr = (turn["system_transcript"] + ' ; ' + turn["transcript"]).strip()
            dialog_history.append(last_uttr)
            last_uttr = turn_uttr

            turn_dialog_state = deepcopy(last_dialog_state)
            for tl in turn["turn_label"]:
                turn_dialog_state[tl[0]] = tl[1]
            
            op_labels, generate_y, gold_state = make_turn_label(slot_meta, last_dialog_state,
                                                                turn_dialog_state,
                                                                tokenizer, op_code)
            turn_dialog_state_pseudo = {}
            if use_pseudo:
                turn_dial = dialogue_idx + "_" + str(turn_idx)
                for slot in slot_meta:
                    turn_dialog_state_pseudo[slot] = pred[turn_dial][slot]["pred"]
                    
                op_labels_pseudo, generate_y_pseudo, _ = make_turn_label(slot_meta, last_dialog_state_pseudo,
                                                                turn_dialog_state_pseudo,
                                                                tokenizer, op_code)
            else:
                op_labels_pseudo = None
                generate_y_pseudo = None

            instance = TrainingInstance(dialogue_idx, turn_idx, turn_uttr, 
                                        ' '.join(dialog_history[-n_history:]),
                                        last_dialog_state, op_labels,
                                        generate_y, gold_state, turn_dialog_state, 
                                        max_seq_length, slot_meta, is_last_turn, 
                                        op_labels_pseudo, generate_y_pseudo,
                                        op_code=op_code)
            instance.make_instance(tokenizer)
            data.append(instance)
            last_dialog_state = turn_dialog_state
            last_dialog_state_pseudo = turn_dialog_state_pseudo
#         break
    return data


class TrainingInstance:
    def __init__(self, ID,
                 turn_id,
                 turn_utter,
                 dialog_history,
                 last_dialog_state,
                 op_labels,
                 generate_y,
                 gold_state,
                 cur_turn_state,
                 max_seq_length,
                 slot_meta,
                 is_last_turn,
                 op_labels_pseudo,
                 generate_y_pseudo,
                 op_code='4'):
        self.id = ID
        self.turn_id = turn_id
        self.turn_utter = turn_utter
        self.dialog_history = dialog_history
        self.last_dialog_state = last_dialog_state
        self.gold_p_state = last_dialog_state
        self.generate_y = generate_y
        self.op_labels = op_labels
        self.gold_state = gold_state
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.is_last_turn = is_last_turn
        self.op2id = OP_SET[op_code]
        self.cur_turn_state = cur_turn_state
        self.op_labels_pseudo = op_labels_pseudo
        self.generate_y_pseudo = generate_y_pseudo

    def shuffle_state(self, rng, slot_meta=None):
        new_y = []
        gid = 0
        for idx, aa in enumerate(self.op_labels):
            if aa == 'update':
                new_y.append(self.generate_y[gid])
                gid += 1
            else:
                new_y.append(["dummy"])
                
        new_y_pseudo = []
        gid = 0
        for idx, aa in enumerate(self.op_labels_pseudo):
            if aa == 'update':
                new_y_pseudo.append(self.generate_y_pseudo[gid])
                gid += 1
            else:
                new_y_pseudo.append(["dummy"])
                
        if slot_meta is None:
            temp = list(zip(self.op_labels, self.slot_meta, new_y, self.op_labels_pseudo, new_y_pseudo))
            rng.shuffle(temp)
        else:
            indices = list(range(len(slot_meta)))
            for idx, st in enumerate(slot_meta):
                indices[self.slot_meta.index(st)] = idx
            temp = list(zip(self.op_labels, self.slot_meta, new_y, self.op_labels_pseudo, new_y_pseudo, indices))
            temp = sorted(temp, key=lambda x: x[-1])
        temp = list(zip(*temp))
        self.op_labels = list(temp[0])
        self.slot_meta = list(temp[1])
        self.generate_y = [yy for yy in temp[2] if yy != ["dummy"]]
        self.op_labels_pseudo = list(temp[3])
        self.generate_y_pseudo = [yy for yy in temp[4] if yy != ["dummy"]]

    def make_instance(self, tokenizer, max_seq_length=None,
                      word_dropout=0., slot_token='[SLOT]'):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        state = []
        for s in self.slot_meta:
            state.append(slot_token)
            k = s.split('-')
            v = self.last_dialog_state.get(s)
            if v is not None:
                k.extend(['-', v])
                t = tokenizer.tokenize(' '.join(k))
            else:
                t = tokenizer.tokenize(' '.join(k))
                t.extend(['-', '[NULL]'])
            state.extend(t)
        avail_length_1 = max_seq_length - len(state) - 3
        diag_1 = tokenizer.tokenize(self.dialog_history)
        diag_2 = tokenizer.tokenize(self.turn_utter)
        avail_length = avail_length_1 - len(diag_2)

        if len(diag_1) > avail_length:  # truncated
            avail_length = len(diag_1) - avail_length
            diag_1 = diag_1[avail_length:]

        if len(diag_1) == 0 and len(diag_2) > avail_length_1:
            avail_length = len(diag_2) - avail_length_1
            diag_2 = diag_2[avail_length:]

        drop_mask = [0] + [1] * len(diag_1) + [0] + [1] * len(diag_2) + [0]
        diag_1 = ["[CLS]"] + diag_1 + ["[SEP]"]
        diag_2 = diag_2 + ["[SEP]"]
        segment = [0] * len(diag_1) + [1] * len(diag_2)

        diag = diag_1 + diag_2
        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]
        input_ = diag + state
        segment = segment + [1]*len(state)
        self.input_ = input_

        self.segment_id = segment
        slot_position = []
        for i, t in enumerate(self.input_):
            if t == slot_token:
                slot_position.append(i)
        self.slot_position = slot_position

        input_mask = [1] * len(self.input_)
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)
        if len(input_mask) < max_seq_length:
            self.input_id = self.input_id + [0] * (max_seq_length-len(input_mask))
            self.segment_id = self.segment_id + [0] * (max_seq_length-len(input_mask))
            input_mask = input_mask + [0] * (max_seq_length-len(input_mask))

        self.input_mask = input_mask
        self.op_ids = [self.op2id[a] for a in self.op_labels]
        self.generate_ids = [tokenizer.convert_tokens_to_ids(y) for y in self.generate_y]
        if self.op_labels_pseudo is not None:
            self.op_ids_pseudo = [self.op2id[a] for a in self.op_labels_pseudo]
            self.generate_ids_pseudo = [tokenizer.convert_tokens_to_ids(y) for y in self.generate_y_pseudo]
        else:
            self.op_ids_pseudo = None
            self.generate_ids_pseudo = None


class MultiWozDataset(Dataset):
    def __init__(self, data, tokenizer, slot_meta, max_seq_length, rng,
                 ontology, word_dropout=0.1, shuffle_state=False, shuffle_p=0.5):
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.slot_meta = slot_meta
        self.max_seq_length = max_seq_length
        self.ontology = ontology
        self.word_dropout = word_dropout
        self.shuffle_state = shuffle_state
        self.shuffle_p = shuffle_p
        self.rng = rng

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.shuffle_state and self.shuffle_p > 0.:
            if self.rng.random() < self.shuffle_p:
                self.data[idx].shuffle_state(self.rng, None)
            else:
                self.data[idx].shuffle_state(self.rng, self.slot_meta)
        if self.word_dropout > 0 or self.shuffle_state:
            self.data[idx].make_instance(self.tokenizer,
                                         word_dropout=self.word_dropout)
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        state_position_ids = torch.tensor([f.slot_position for f in batch], dtype=torch.long)
        
        op_ids = torch.tensor([f.op_ids for f in batch], dtype=torch.long)
        op_ids_pseudo = torch.tensor([f.op_ids_pseudo for f in batch], dtype=torch.long)
        
        gen_ids = [b.generate_ids for b in batch]
        gen_ids_pseudo = [b.generate_ids_pseudo for b in batch]
        
        max_update = max([len(b) for b in gen_ids])
        max_value = max([len(b) for b in flatten(gen_ids)])
        for bid, b in enumerate(gen_ids):
            n_update = len(b)
            for idx, v in enumerate(b):
                b[idx] = v + [0] * (max_value - len(v))
            gen_ids[bid] = b + [[0] * max_value] * (max_update - n_update)
        gen_ids = torch.tensor(gen_ids, dtype=torch.long)
        
        # pseudo
        max_update_pseudo = max([len(b) for b in gen_ids_pseudo])
        max_value_pseudo = max([len(b) for b in flatten(gen_ids_pseudo)])
        for bid, b in enumerate(gen_ids_pseudo):
            n_update = len(b)
            for idx, v in enumerate(b):
                b[idx] = v + [0] * (max_value_pseudo - len(v))
            gen_ids_pseudo[bid] = b + [[0] * max_value_pseudo] * (max_update_pseudo - n_update)
        gen_ids_pseudo = torch.tensor(gen_ids_pseudo, dtype=torch.long)
        #

        return input_ids, input_mask, segment_ids, state_position_ids, op_ids, gen_ids, max_value, max_update, op_ids_pseudo, gen_ids_pseudo, max_value_pseudo, max_update_pseudo
