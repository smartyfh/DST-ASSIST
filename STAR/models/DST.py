import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from transformers import BertPreTrainedModel, BertModel
    

class UtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(UtteranceEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)
        
        self.init_weights()
        
    def forward(self, input_ids, attention_mask, token_type_ids, output_attentions=False, output_hidden_states=False):
        return self.bert(input_ids=input_ids, 
                         attention_mask=attention_mask, 
                         token_type_ids=token_type_ids, 
                         output_attentions=output_attentions, 
                         output_hidden_states=output_hidden_states)

    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores  
    

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])
            

class SlotSelfAttention(nn.Module):
    "A stack of N layers"
    def __init__(self, layer, N):
        super(SlotSelfAttention, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)    
    
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x = self.norm(x)
        return x + self.dropout(sublayer(x))
    
    
class SlotAttentionLayer(nn.Module):
    "SlotAttentionLayer is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(SlotAttentionLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.ReLU() # use gelu or relu

    def forward(self, x):
        return self.w_2(self.dropout(self.gelu(self.w_1(x)))) 
    

class UtteranceAttention(nn.Module):
    def __init__(self, attn_head, model_output_dim, dropout=0.):
        super(UtteranceAttention, self).__init__()
        
        self.attn_head = attn_head
        self.model_output_dim = model_output_dim
        self.dropout = dropout
        self.attn_fun = MultiHeadAttention(attn_head, model_output_dim, dropout=0.)
        
    def forward(self, query, value, attention_mask=None):
        num_query = query.size(0)
        batch_size = value.size(0)
        seq_length = value.size(1)
        
        expanded_query = query.unsqueeze(0).expand(batch_size, *query.shape)
        if attention_mask is not None:
            expanded_attention_mask = attention_mask.view(-1, seq_length, 1).expand(value.size()).float()
            new_value = torch.mul(value, expanded_attention_mask)
            attn_mask = attention_mask.unsqueeze(1).expand(batch_size, num_query, seq_length)
        else:
            new_value = value
            attn_mask = None
        
        attended_embedding = self.attn_fun(expanded_query, new_value, new_value, mask=attn_mask)
        
        return attended_embedding
        
        
class Decoder(nn.Module):
    def __init__(self, attn_head, bert_output_dim, dropout_prob, num_self_attention_layer):
        super(Decoder, self).__init__()
        # slot utterance attention
        self.slot_utter_attn = UtteranceAttention(attn_head, bert_output_dim, dropout=0.)
        
        # MLP
        self.SlotMLP = nn.Sequential(nn.Linear(bert_output_dim * 2, bert_output_dim),
                                     nn.ReLU(),
                                     nn.Dropout(p=dropout_prob),
                                     nn.Linear(bert_output_dim, bert_output_dim))
        
        # basic modues, attention dropout is 0.1 by default
        attn = MultiHeadAttention(attn_head, bert_output_dim)
        ffn = PositionwiseFeedForward(bert_output_dim, bert_output_dim, dropout_prob)
        
        ### attention layer, multiple self attention layers
        self.slot_self_attn = SlotSelfAttention(SlotAttentionLayer(bert_output_dim, deepcopy(attn), 
                                                                   deepcopy(ffn), dropout_prob),
                                                                   num_self_attention_layer)
        # prediction
        self.pred = nn.Sequential(nn.Dropout(p=dropout_prob), 
                                  nn.Linear(bert_output_dim, bert_output_dim),
                                  nn.LayerNorm(bert_output_dim))                                  
     
    def forward(self, sequence_output, attention_mask, slot_embedding):         
        # slot utterance attention
        slot_utter_emb = self.slot_utter_attn(slot_embedding, sequence_output, attention_mask)
        
        batch_size = sequence_output.size(0)
        # concat
        slot_utter_embedding = torch.cat((slot_embedding.unsqueeze(0).repeat(batch_size, 1, 1), slot_utter_emb), 2)
        
        # MLP
        slot_utter_embedding2 = self.SlotMLP(slot_utter_embedding)
        
        # slot self attention
        hidden_slot = self.slot_self_attn(slot_utter_embedding2) 
        
        # prediction
        hidden = self.pred(hidden_slot)
        
        return hidden # [batch_size, num_slots, dim]
        

class BeliefTracker(nn.Module):
    def __init__(self, pretrained_model_type, attn_head, dropout_prob=0., num_self_attention_layer=0):
        super(BeliefTracker, self).__init__()
        
        self.encoder = UtteranceEncoding.from_pretrained(pretrained_model_type)
        self.hidden_size = self.encoder.config.hidden_size
        self.decoder = Decoder(attn_head, self.hidden_size, dropout_prob, num_self_attention_layer)

    def forward(self, input_ids, attention_mask, token_type_ids, slot_emb):        
        # encoder, a pretrained model, output is a tuple
        sequence_output = self.encoder(input_ids, attention_mask, token_type_ids)[0] # [batch_size, seq_length, dim]    
        
        # decoder, slot utterance attention, followed by a linear layer        
        slot_output = self.decoder(sequence_output, attention_mask, slot_emb)       
          
        return slot_output