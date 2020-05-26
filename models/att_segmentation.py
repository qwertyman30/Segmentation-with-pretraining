import random
import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
try:
    from deeplab_decoder import Decoder
    from attention_layer import Attention
except ModuleNotFoundError:
    from .deeplab_decoder import Decoder
    from .attention_layer import Attention


class AttSegmentator(nn.Module):

    def __init__(self, num_classes, encoder, att_type='additive', img_size=(512, 512)):
        super().__init__()
        self.low_feat = IntermediateLayerGetter(encoder, {"layer1": "layer1"}).cuda()
        self.encoder = IntermediateLayerGetter(encoder, {"layer4": "out"}).cuda()
        # For resnet18
        encoder_dim = 512
        low_level_dim = 64
        self.num_classes = num_classes

        self.class_encoder = nn.Linear(num_classes, 512)

        self.attention_enc = Attention(encoder_dim, att_type)

        self.decoder = Decoder(2, encoder_dim, img_size, low_level_dim=low_level_dim, rates=[1, 6, 12, 18])

    def forward(self, x, v_class, out_att=False):
        self.low_feat.eval()
        self.encoder.eval()
        with torch.no_grad():
            low_level_feat = self.low_feat(x)['layer1']
            enc_feat = self.encoder(x)['out']

        query = self.class_encoder(v_class)
        shape = enc_feat.shape

        enc_feat = enc_feat.permute(0, 2, 3, 1).contiguous().view(shape[0], -1, shape[1])

        x_enc, attention = self.attention_enc(enc_feat, query)
        x_enc = x_enc.view(shape)

        segmentation = self.decoder(x_enc, low_level_feat)

        if out_att:
            return segmentation, attention
        return segmentation


if __name__ == "__main__":
    from torchvision.models.resnet import resnet18
    pretrained_model = resnet18(num_classes=4).cuda()
    model = AttSegmentator(10, pretrained_model, att_type='dotprod', double_att=True).cuda()
    model.eval()
    print(model)
    image = torch.randn(1, 3, 512, 512).cuda()
    v_class = torch.randn(1, 10).cuda()
    with torch.no_grad():
        output = model.forward(image, v_class)
    print(output.size())
