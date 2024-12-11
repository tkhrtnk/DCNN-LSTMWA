import torch
from torch import nn
from torchvision.models.alexnet import AlexNet_Weights, alexnet
from torchvision.models.efficientnet import EfficientNet_B6_Weights, efficientnet_b6
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    batch_size = 16
    seq_len = 8
    input_dim = 4096

    # seq_len個のsegmentの出力（サイズ:input_dim）を合わせた行列を想定
    input = torch.randn(batch_size, seq_len, input_dim, device=device)
    print(f'{input.size()=}\n')

    model = LSTMEmoClf(seq_len=seq_len).to(device)
    print(f'Model structure: ', model, '\n\n')

    logits = model(input)
    print(f'{logits.size()=}')
    pred_prob = nn.Softmax(dim=0)(logits)
    y_pred = pred_prob.argmax(dim=1)
    print(f'Predicted class: {y_pred}')

    # for name, param in model.named_parameters():
    #     print(f'Layer: {name} | Size: {param.size()}\n')

class LSTMEmoClf(nn.Module):
    def __init__(self, num_classes, seq_len, dropout, extractor_path):
        super(LSTMEmoClf, self).__init__()
        self.seq_len = seq_len
        net = DCNN(num_classes=num_classes, weight_path=extractor_path)
        self.extractor = create_feature_extractor(net, {'net.net.classifier.4': 'feature'})
        self.bilstm_stack = nn.LSTM(
            input_size = 4096,
            hidden_size = 128, # outputもこのサイズ
            batch_first = True,
            num_layers = seq_len, 
            bidirectional = True
        )
        self.self_attention = SelfAttention(input_dim=128*2)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=128*2, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )
    
    def forward(self, x):
        # x.size() -> (batch_size, seq_len, C, H, W)
        feature = torch.zeros(x.size(0), x.size(1), 4096)
        for seq in range(self.seq_len):
            feature[:,seq] = self.extractor(x[:,seq])['feature']
        # feature.size() -> (batch_size, seq_len, x_emb_dim=4096)
        lstm_out, _ = self.bilstm_stack(feature)
        # lstm_out.size() -> (batch_size, seq_len, lstm_emb_dim)
        # lstm_out: segment-level features
        attn_output  = self.self_attention(lstm_out)
        # attn_output.size() -> (batch_size, lstm_emb_dim)
        # attn_output: utterance-level features
        output = self.linear_relu_stack(attn_output)
        # output.size() -> (batch_size, num_classes)
        return output
    
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.key = nn.Linear(input_dim, input_dim)
        self.query = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, x):
        # x.size() -> (batch_size, seq_len, input_dim)
        # keys, queries, valuesも同様
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # scores.size() -> (batch_size, seq_len(query), seq_len(key))
        # scores は普通、対称行列にはならない。異なるアフィン変換をkeyとqueryに施しているため。
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5 + 1e-10)

        # weights.size() -> (batch_size, seq_len(query), seq_len(key))
        # keyのseq_len方向（scoresのdim=2）に沿って、scoresをsoftmaxする
        weights = self.softmax(scores)

        # weighted_sums.size() -> (batch_size, seq_len(query), input_dim)
        weighted_sums = torch.bmm(weights, values)
        
        
        # average_sums.size() -> (batch_size, input_dim)
        # queryでのweighted_sumsの平均
        average_sums = weighted_sums.sum(dim=1)
        
        return average_sums
    
class DCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.5, pretrained=False, weight_path=None):
        super(DCNN, self).__init__()
        # 事前学習された重みをロード
        if pretrained and weight_path is None:
            self.net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1, dropout=dropout)
            print('load: AlexNet_Weights.IMAGENET1K_V1')
        else:
            self.net = alexnet(dropout=dropout)
        # alexnetの線形層の出力を置換して調整
        self.net.classifier[6] = nn.Linear(in_features=self.net.classifier[6].in_features, out_features=num_classes)
        # ファインチューニングしたモデルをロード
        if weight_path is not None:
            self.net = torch.load(weight_path)

    def forward(self, x):
        return self.net(x)
    
class DNN(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(DNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
if __name__ == '__main__':
    main()