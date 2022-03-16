import torch
import numpy as np
import json
import time
from copy import deepcopy
from utils.data_utils import OP_SET, make_turn_label, postprocessing
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
import os
from tqdm import tqdm


def model_evaluation(model, test_data, tokenizer, slot_meta, epoch, device, op_code='4',
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=False):
    model.eval()
    op2id = OP_SET[op_code]
    id2op = {v: k for k, v in op2id.items()}

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
    final_joint_acc, final_count, final_slot_F1_pred, final_slot_F1_count = 0, 0, 0, 0
    op_acc = 0
    
    joint_turn_level_acc = 0

    results = {}
    results2 = {}
    last_dialog_state = {}
    ground_dialog_state = {}
    wall_times = []
    for di, i in enumerate(tqdm(test_data)):
        if i.turn_id == 0:
            last_dialog_state = deepcopy(i.gold_p_state)

        if is_gt_p_state is False:
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)
        else:  # ground-truth previous dialogue state
            last_dialog_state = deepcopy(i.gold_p_state)
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)
            
            
        gt_turn_label = []
        for slot in slot_meta:
            if i.gold_p_state[slot] != i.cur_turn_state[slot]:
                gt_turn_label.append(slot + "-" + i.cur_turn_state[slot])
                
        input_ids = torch.LongTensor([i.input_id]).to(device)
        input_mask = torch.LongTensor([i.input_mask]).to(device)
        segment_ids = torch.LongTensor([i.segment_id]).to(device)
        state_position_ids = torch.LongTensor([i.slot_position]).to(device)
        
        ground_dialog_state = deepcopy(last_dialog_state)

        d_gold_op, _, _ = make_turn_label(slot_meta, last_dialog_state, i.gold_state,
                                          tokenizer, op_code, dynamic=True)
        gold_op_ids = torch.LongTensor([d_gold_op]).to(device)

        start = time.perf_counter()
        MAX_LENGTH = 15
        with torch.no_grad():
            # ground-truth state operation
            gold_op_inputs = gold_op_ids if is_gt_op else None
            s, g = model(input_ids=input_ids,
                            token_type_ids=segment_ids,
                            state_positions=state_position_ids,
                            attention_mask=input_mask,
                            max_value=MAX_LENGTH,
                            op_ids=gold_op_inputs)

        _, op_ids = s.view(-1, len(op2id)).max(-1)

        if g.size(1) > 0:
            generated = g.squeeze(0).max(-1)[1].tolist()
        else:
            generated = []

        if is_gt_op:
            pred_ops = [id2op[a] for a in gold_op_ids[0].tolist()]
        else:
            pred_ops = [id2op[a] for a in op_ids.tolist()]
        gold_ops = [id2op[a] for a in d_gold_op]

        if is_gt_gen:
            # ground_truth generation
            gold_gen = {'-'.join(ii.split('-')[:2]): ii.split('-')[-1] for ii in i.gold_state}
        else:
            gold_gen = {}
        generated, last_dialog_state = postprocessing(slot_meta, pred_ops, last_dialog_state,
                                                      generated, tokenizer, op_code, gold_gen)
        end = time.perf_counter()
        wall_times.append(end - start)
        
        t_turn_label = []
        for slot in slot_meta:
            if ground_dialog_state[slot] != last_dialog_state[slot]:
                t_turn_label.append(slot+"-"+last_dialog_state[slot])
                
        if set(gt_turn_label) == set(t_turn_label):
            joint_turn_level_acc += 1.0
        
        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))

        if set(pred_state) == set(i.gold_state):
            joint_acc += 1
        key = str(i.id) + '_' + str(i.turn_id)
        results[key] = [pred_state, i.gold_state]
        ss = {}
        for slot in slot_meta:
            ss[slot] = {}
            ss[slot]["pred"] = last_dialog_state[slot]
            ss[slot]["gt"] = i.cur_turn_state[slot]
#         ss["turn_label"] = gt_turn_label
#         ss["pred_turn_label"] = t_turn_label
        results2[key] = ss
        

        # Compute operation accuracy
        temp_acc = sum([1 if p == g else 0 for p, g in zip(pred_ops, gold_ops)]) / len(pred_ops)
        op_acc += temp_acc

        if i.is_last_turn:
            final_count += 1
            if set(pred_state) == set(i.gold_state):
                final_joint_acc += 1
    

    joint_acc_score = joint_acc / len(test_data)
    joint_turn_level_acc_socre = joint_turn_level_acc / len(test_data)
    op_acc_score = op_acc / len(test_data)
    final_joint_acc_score = final_joint_acc / final_count
    latency = np.mean(wall_times) * 1000
    
    print("------------------------------")
    print('op_code: %s, is_gt_op: %s, is_gt_p_state: %s, is_gt_gen: %s' % \
          (op_code, str(is_gt_op), str(is_gt_p_state), str(is_gt_gen)))
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d op accuracy : " % epoch, op_acc_score)
    print("Final Joint Accuracy : ", final_joint_acc_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    
    if not os.path.exists("pred"):
        os.makedirs("pred")
    json.dump(results2, open('pred/preds_%d.json' % epoch, 'w'), indent=4)
#     per_domain_join_accuracy(results, slot_meta)

    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'op_acc': op_acc_score, 
              'joint_turn_acc':joint_turn_level_acc_socre}
    return scores
