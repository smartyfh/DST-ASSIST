import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import re
import os
from copy import deepcopy


def slot_recovery(slot):
    if "pricerange" in slot:
        return slot.replace("pricerange", "price range")
    elif "arriveby" in slot:
        return slot.replace("arriveby", "arrive by")
    elif "leaveat" in slot:
        return slot.replace("leaveat", "leave at")
    else:
        return slot


class Processor(object):
    def __init__(self, config):
        # MultiWOZ dataset
        if "data/mwz" in config.data_dir:
            fp_ontology = open(os.path.join(config.data_dir, "ontology-modified.json"), "r")
            ontology = json.load(fp_ontology)
            fp_ontology.close()
        else:
            raise NotImplementedError()

        self.ontology = ontology
        self.slot_meta = list(self.ontology.keys()) # must be sorted
        self.num_slots = len(self.slot_meta)
        self.label_map = {slot: {label: i for i, label in enumerate(self.ontology[slot])} for slot in self.slot_meta}
        self.config = config
        
    def _load_data(self, input_file):        
        return json.load(open(input_file, "r"))

    def get_instances(self, data_dir, data_file, tokenizer, use_pseudo_label=False):
        return self._create_instances(self._load_data(os.path.join(data_dir, data_file)), tokenizer, use_pseudo_label)

    def _create_instances(self, dials, tokenizer, use_pseudo_label):
        if use_pseudo_label:
            if '2.0' in self.config.data_dir:
                pred = json.load(open("pseudo_label_aux_20.json"))
            elif '2.4' in self.config.data_dir:
                pred = json.load(open("pseudo_label_aux_24.json"))
            else:
                raise Exception("version not supported")
        instances = []       
        for dial_dict in dials:
            last_uttr = ""
            history_uttr = []
            last_dialogue_state = {}         
            for slot in self.slot_meta:
                last_dialogue_state[slot] = "none"
                
            dialogue_idx = dial_dict["dialogue_idx"]    
            for ti, turn in enumerate(dial_dict["dialogue"]):               
                turn_idx = turn["turn_idx"]
                if (ti + 1) == len(dial_dict["dialogue"]):
                    is_last_turn = True
                else:
                    is_last_turn = False
                system_response = turn["system_transcript"]
                user_utterance = turn["transcript"]
                
                curr_turn_only_label = [] # turn label
                curr_dialogue_state = deepcopy(last_dialogue_state)
                for tl in turn["turn_label"]:
                    curr_dialogue_state[tl[0]] = tl[1]
                    curr_turn_only_label.append(tl[0] + "-" + tl[1])
                curr_dialogue_state_ids = []                
                for slot in self.slot_meta:
                    curr_dialogue_state_ids.append(self.label_map[slot][curr_dialogue_state[slot]])
                    
                history_uttr.append(last_uttr)
                text_curr = (system_response + " " + user_utterance).strip()
                text_hist = ' '.join(history_uttr[-self.config.num_history:])
                last_uttr = text_curr
                
                pseudo_label_ids = None
                if use_pseudo_label:
                    pseudo_label_ids = []
                    turn_dial = dialogue_idx + "_" + str(ti)
                    for slot in self.slot_meta:
                        pseudo_label_ids.append(self.label_map[slot][pred[turn_dial][slot]["pred"]])
                        assert pred[turn_dial][slot]['gt'] == curr_dialogue_state[slot]
               
                instance = TrainingInstance(dialogue_idx, turn_idx, 
                                            text_curr + " none ", text_hist, 
                                            curr_dialogue_state_ids,
                                            curr_turn_only_label, 
                                            curr_dialogue_state, 
                                            last_dialogue_state,
                                            self.config.max_seq_length, 
                                            self.slot_meta, 
                                            is_last_turn,
                                            pseudo_label_ids)
                
                instance.make_instance(tokenizer)
                instances.append(instance)            
                last_dialogue_state = curr_dialogue_state
            
        return instances
            
          
class TrainingInstance(object):
    def __init__(self, ID,
                 turn_id,
                 turn_utter,
                 dialogue_history,
                 label_ids,
                 turn_label,
                 curr_turn_state,
                 last_turn_state,
                 max_seq_length,
                 slot_meta,
                 is_last_turn,
                 pseudo_label_ids):
        self.dialogue_id = ID
        self.turn_id = turn_id
        self.turn_utter = turn_utter
        self.dialogue_history = dialogue_history
        
        self.curr_dialogue_state = curr_turn_state
        self.last_dialogue_state = last_turn_state
        self.gold_last_state = deepcopy(last_turn_state)
        
        self.turn_label = turn_label
        self.label_ids = label_ids
       
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.is_last_turn = is_last_turn
        
        self.pseudo_label_ids = pseudo_label_ids

    def make_instance(self, tokenizer, max_seq_length=None, word_dropout=0.):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
            
        state = []
        for slot in self.slot_meta:
            s = slot_recovery(slot)
            k = s.split('-')
            v = self.last_dialogue_state[slot].lower() # use the original slot name as index
            if v == "none":
                continue
            k.extend([v]) # without symbol "-"
            t = tokenizer.tokenize(' '.join(k))
            state.extend(t)
            
        avail_length_1 = max_seq_length - len(state) - 3
        diag_1 = tokenizer.tokenize(self.turn_utter)
        diag_2 = tokenizer.tokenize(self.dialogue_history)
        avail_length = avail_length_1 - len(diag_1)

        if avail_length <= 0:
            diag_2 = []
        elif len(diag_2) > avail_length:  # truncated
            avail_length = len(diag_2) - avail_length
            diag_2 = diag_2[avail_length:]

        if len(diag_2) == 0 and len(diag_1) > avail_length_1:
            avail_length = len(diag_1) - avail_length_1
            diag_1 = diag_1[avail_length:]
        
        # we keep the order
        drop_mask = [0] + [1] * len(diag_2) + [0] * len(state) + [0] + [1] * len(diag_1) + [0] # word dropout
        diag_2 = ["[CLS]"] + diag_2 + state + ["[SEP]"]
        diag_1 = diag_1 + ["[SEP]"]
        diag = diag_2 + diag_1
        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]
        
        self.input_ = diag
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)
        segment = [0] * len(diag_2) + [1] * len(diag_1)
        self.segment_id = segment
        input_mask = [1] * len(self.input_)
        self.input_mask = input_mask
        

class MultiWozDataset(Dataset):
    def __init__(self, data, tokenizer, word_dropout=0., use_pseudo_label=False):
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.word_dropout = word_dropout
        self.use_pseudo_label = use_pseudo_label
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.word_dropout > 0.:
            self.data[idx].make_instance(self.tokenizer, word_dropout=self.word_dropout)
        return self.data[idx]

    def collate_fn(self, batch):
        def padding(list1, list2, list3, pad_token):
            max_len = max([len(i) for i in list1]) # utter-len
            result1 = torch.ones((len(list1), max_len)).long() * pad_token
            result2 = torch.ones((len(list2), max_len)).long() * pad_token
            result3 = torch.ones((len(list3), max_len)).long() * pad_token
            for i in range(len(list1)):
                result1[i, :len(list1[i])] = list1[i]
                result2[i, :len(list2[i])] = list2[i]
                result3[i, :len(list3[i])] = list3[i]
            return result1, result2, result3
        
        input_ids_list, segment_ids_list, input_mask_list = [], [], []
        for f in batch:
            input_ids_list.append(torch.LongTensor(f.input_id))
            segment_ids_list.append(torch.LongTensor(f.segment_id))
            input_mask_list.append(torch.LongTensor(f.input_mask))
            
        input_ids, segment_ids, input_mask = padding(input_ids_list, 
                                                     segment_ids_list, 
                                                     input_mask_list, 
                                                     torch.LongTensor([0]))
        label_ids = torch.tensor([f.label_ids for f in batch], dtype=torch.long)
        
        if self.use_pseudo_label:
            pseudo_label_ids = torch.tensor([f.pseudo_label_ids for f in batch], dtype=torch.long)
            return input_ids, segment_ids, input_mask, label_ids, pseudo_label_ids
        else:
            return input_ids, segment_ids, input_mask, label_ids