import torch
from torch import nn
import argparse
from utils import segment_level_feature_extractor_from_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("extractor_path")
    parser.add_argument("model_path")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    torch.set_default_device(device)

    # 特徴抽出器をロードする
    extractor = segment_level_feature_extractor_from_file(num_classes=4, extractor_path=args.extractor_path, device=device)
    print(f'load extractor: {args.extractor_path}')
    
    # モデルのロード
    model = torch.load(args.model_path)
    print(f'load model: {args.model_path}')

    # 確認(summaryはRNNだと使用不可)
    # summary(model, input_size=(seq_len, 4096))
    # print(f'input_size: ({model.bilstm_stack.num_layers}, {model.bilstm_stack.input_size})')
    # print(f'output_size: {model.linear_relu_stack[5].out_features}')
    # print(f'model structure: \n{model}')
    # print(f'{list(model.parameters())[0]=}')

    id2emo = {0:'Neutral', 1:'Happy', 2:'Sad', 3:'Angry'}

    while True:
        wavpath = input("input wavpath or 'q' to quit> ")
        if wavpath == 'q':
            break
        pred, label = inference(wavpath, extractor, model)
        print(f'prediction:\n {pred}')
        print(f'label(pred): {id2emo[label]}')


def inference(wavpath, extractor, model):
    feature = extractor(wavpath)
    with torch.no_grad():
        pred = model(feature)
    prob = nn.Softmax(dim=1)(pred)
    return prob, int(prob.argmax(1))

if __name__ == "__main__":
    main()