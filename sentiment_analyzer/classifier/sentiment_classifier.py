import torch
from torch import nn
from transformers import BertModel

pretrained_model_name = 'bert-base-cased'
class_names = ['negative', 'neutral', 'positive']
MAX_LEN = 160


class BertCNN(nn.Module):
    def __init__(self, n_classes):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels=128, kernel_size=3),
            nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels=128, kernel_size=4),
            nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels=128, kernel_size=5)
        ])
        self.pooling_layer = nn.AdaptiveMaxPool1d(1)
        self.dense_layer = nn.Linear(128*3, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.output_layer = nn.Linear(128, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = conv_layer(bert_output.transpose(1, 2))
            pool_output = self.pooling_layer(conv_output)
            pool_output = pool_output.view(pool_output.size(0), -1)
            conv_outputs.append(pool_output)
        concat_output = torch.cat(conv_outputs, dim=1)
        dense_output = self.dense_layer(concat_output)
        dropout_output = self.dropout(dense_output)
        output = self.output_layer(dropout_output)
        return output
