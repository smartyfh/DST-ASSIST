import torch
import numpy as np
import json
import time
from copy import deepcopy
from tqdm import tqdm
import os
from .loss_utils import slot_value_matching, hard_cross_entropy_loss


def model_evaluation(model, test_data, tokenizer, slot_lookup, value_lookup, ontology, ep, is_gt_p_state=False):
    model.eval()
    slot_meta = list(ontology.keys())
    
    device = slot_lookup.device
    
    loss = 0.
    joint_acc = 0.
    joint_turn_acc = 0.
    slot_acc = np.array([0.] * len(slot_meta))

    results = {}
    last_dialogue_state = {}
    wall_times = [] 
    for di, i in enumerate(tqdm(test_data)):
        if i.turn_id == 0 or is_gt_p_state:
            last_dialogue_state = deepcopy(i.gold_last_state)
            
        i.last_dialogue_state = deepcopy(last_dialogue_state)
        i.make_instance(tokenizer)

        input_ids = torch.LongTensor([i.input_id]).to(device)
        input_mask = torch.LongTensor([i.input_mask]).to(device)
        segment_ids = torch.LongTensor([i.segment_id]).to(device)
        label_ids = torch.LongTensor([i.label_ids]).to(device)
        
        start = time.perf_counter()
        with torch.no_grad():
            t_slot_output = model(input_ids=input_ids, attention_mask=input_mask,
                                  token_type_ids=segment_ids, slot_emb=slot_lookup)
            
        _, t_pred_all_distance = slot_value_matching(t_slot_output, value_lookup)
        t_loss, _, t_pred_slot = hard_cross_entropy_loss(t_pred_all_distance, label_ids)

        t_accuracy = t_pred_slot == label_ids
        batch_size = t_pred_slot.size(0)
        t_acc_slot = torch.true_divide(torch.sum(t_accuracy, 0).float(), batch_size).cpu().detach().numpy() # slot accuracy
        t_acc = torch.sum(torch.floor_divide(torch.sum(t_accuracy, 1), len(slot_meta))).float().item() / batch_size # JGA

        loss += t_loss.item()
        joint_acc += t_acc
        slot_acc += t_acc_slot
                       
        end = time.perf_counter()
        wall_times.append(end - start)
        
        ss = {}
        t_turn_label = []
        for s, slot in enumerate(slot_meta):
            v = ontology[slot][t_pred_slot[0, s].item()]
            if v != last_dialogue_state[slot]:
                t_turn_label.append(slot + "-" + v)
            last_dialogue_state[slot] = v
            vv = ontology[slot][i.label_ids[s]]
#             if v == vv:
#                 continue
            # only record wrong slots
            ss[slot] = {}
            ss[slot]["pred"] = v
            ss[slot]["gt"] = vv
        
        if set(t_turn_label) == set(i.turn_label):
            joint_turn_acc += 1

        key = str(i.dialogue_id) + '_' + str(i.turn_id)
        results[key] = ss
        
    loss = loss / len(test_data)
    joint_acc_score = joint_acc / len(test_data)
    joint_turn_acc_score = joint_turn_acc / len(test_data)
    slot_acc_score = slot_acc / len(test_data)    
    latency = np.mean(wall_times) * 1000 # ms
    
    print("------------------------------")
    print('is_gt_p_state: %s' % (str(is_gt_p_state)))
    print("Epoch %d joint accuracy : " % ep, joint_acc_score)
    print("Epoch %d joint turn accuracy : " % ep, joint_turn_acc_score)
    print("Epoch %d slot accuracy : " % ep, np.mean(slot_acc_score))
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    
    if not os.path.exists("pred"):
        os.makedirs("pred")
    json.dump(results, open('pred/preds_%d.json' % ep, 'w'), indent=4)

    scores = {'epoch': ep, 'loss': loss, 'joint_acc': joint_acc_score, 'joint_turn_acc': joint_turn_acc_score, 'slot_acc': slot_acc_score, 'ave_slot_acc': np.mean(slot_acc_score)}
    
    return scores