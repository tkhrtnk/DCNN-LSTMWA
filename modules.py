import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from speechbrain.dataio.preprocess import AudioNormalizer 
from speechbrain.pretrained import EncoderClassifier 
from models import DCNN
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from preprocess import extract_resized_segments_from_file
import torchvision.transforms as T

class ExtractXvector:
    def __init__(self):
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.classifier = EncoderClassifier.from_hparams( 
            source='speechbrain/spkrec-ecapa-voxceleb',
            run_opts={'device': self._device},
        ) 
        self.audio_norm = AudioNormalizer()

    def __call__(self, wav, sr, wav_path=None) -> np.ndarray:
        if wav_path is not None:
            wav, sr = sf.read(wav_path)
        # Amp Normalization -1 ~ 1
        amax = np.amax(np.absolute(wav))
        wav = wav.astype(np.float32) / amax
        # Freq Norm
        wav = self.audio_norm(torch.from_numpy(wav), sr).to(self._device)
        # x-vector Extraction (192)
        embeds = self.classifier.encode_batch(wav).detach().cpu()[0][0]

        return embeds
    
class SegmentLevelFeatureExtractor():
  def __init__(self, num_classes, extractor_path, device=None):
    net = DCNN(num_classes=num_classes, weight_path=extractor_path)
    net.to(device if device is not None else 'cpu')
    net.eval()
    
    # # DEBUG
    # print(f'{get_graph_node_names(net)=}')
    # exit(0)

    self.extractor = create_feature_extractor(net, {'net.net.classifier.4': 'feature1'})

  def __call__(self, x):
    # x: (batch, seq_len, C, H, W) -> output: (batch, seq_len, 4096)
    output = torch.stack([self.extractor(x[:,idx])['feature1'] for idx in range(x.size(1))], dim=1)
    # # DEBUG
    # print(f'{x.size()=}')
    # print(f'{output.size()=}')
    # exit(0)
    return output

class SegmentLevelFeatureExtractorFromFile(SegmentLevelFeatureExtractor):
  def __init__(self, num_classes, extractor_path, device=None):
    # DEBUG
    # print(num_classes)
    # print(extractor_path)
    # exit(0)
    super().__init__(num_classes, extractor_path, device)

  def __call__(self, wavpath):
    x = extract_resized_segments_from_file(wavpath, normalizer=T.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), device=self.device)
    x = x.unsqueeze(0)
    # x: (batch(0), seq_len, C, H, W) -> output: (batch(0), seq_len, 4096)
    output = super().__call__(x)
    # # DEBUG
    # print(f'{output.size()}')
    # exit(0)
    return output