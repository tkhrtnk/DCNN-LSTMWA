import torch
from torch import nn
import argparse
from preprocess import extract_resized_segments_from_file
import torchvision.transforms as T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    torch.set_default_device(device)

    # モデルのロード
    model = torch.load(args.model_path, device)
    print(f'load model: {args.model_path}')

    # 確認(summaryはRNNだと使用不可)
    # summary(model, input_size=(seq_len, 4096))
    # print(f'input_size: ({model.bilstm_stack.num_layers}, {model.bilstm_stack.input_size})')
    # print(f'output_size: {model.linear_relu_stack[5].out_features}')
    # print(f'model structure: \n{model}')
    # print(f'{list(model.parameters())[0]=}')

    id2emo = {0:'Neutral', 1:'Happy', 2:'Sad', 3:'Angry'}
    if 'esd' in args.model_path:
        max_sec = 3.0
    elif 'jtes' in args.model_path:
        max_sec = 4.0
    else:
        max_sec = 3.0

    while True:
        wavpath = input("input wavpath or 'q' to quit> ")
        if wavpath == 'q':
            break
        pred, label = inference(wavpath, model, device, max_sec)
        print(f'prediction:\n {pred}')
        print(f'label(pred): {id2emo[label]}')


def inference(wavpath, model, device, max_sec):
    X = extract_resized_segments_from_file(wavpath, device=device, max_sec=max_sec, 
                                           normalizer=T.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    # X.unsqueeze(0).size() -> [batch_size(1), seq_len, C, H, W]
    with torch.no_grad():
        pred = model(X.unsqueeze(0))
        prob = nn.Softmax(dim=1)(pred)
    return prob, int(prob.argmax(1))

if __name__ == "__main__":
    main()