import torch
from mmseg.models.decode_heads import SegmenterMaskTransformerHead
from torch import nn

from ._dinov3 import DINOv3ViT
from .adapter import BCIA, LightweightDSMEncoder
from .checkpoints import load_backbone_checkpoint


INSERT_AFTER = 17
PREFIX_TOKENS = 5
TILE_SIZE = 512
NUM_CLASSES = 6
IGNORE_INDEX = 255


def make_backbone():
    return DINOv3ViT(
        image_size=TILE_SIZE, hidden_size=1024, intermediate_size=4096,
        num_attention_heads=16, num_hidden_layers=24, num_register_tokens=4,
        freeze=True, out_features=(23,),
    ).eval()


def insert_residual(tokens, residual):
    return torch.cat((tokens[:, :PREFIX_TOKENS],
                      tokens[:, PREFIX_TOKENS:] + residual), dim=1)


class BCIAPeft(nn.Module):
    def __init__(self, checkpoint=None, *, random_init=False):
        super().__init__()
        if (checkpoint is not None) == random_init:
            raise ValueError("Provide one checkpoint or explicitly set random_init=True for smoke tests")
        self.backbone = make_backbone()
        if random_init:
            nn.init.normal_(self.backbone.embeddings.register_tokens, std=0.02)
        else:
            load_backbone_checkpoint(self.backbone, checkpoint)
        self.dsm_encoder = LightweightDSMEncoder()
        self.bcia = BCIA()
        self.decode_head = SegmenterMaskTransformerHead(
            in_channels=1024, channels=1024, num_classes=NUM_CLASSES,
            num_layers=2, num_heads=16, embed_dims=1024, dropout_ratio=0.0,
            ignore_index=IGNORE_INDEX, align_corners=False,
        )
        self.decode_head.init_weights()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _check_inputs(self, rgb, dsm, valid):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, TILE_SIZE, TILE_SIZE):
            raise ValueError("Expected RGB [B,3,512,512]")
        expected = (rgb.shape[0], 1, TILE_SIZE, TILE_SIZE)
        if dsm.shape != expected or valid.shape != expected:
            raise ValueError("Expected DSM and valid [B,1,512,512]")
        if rgb.dtype != torch.float32 or dsm.dtype != torch.float32 or valid.dtype != torch.bool:
            raise TypeError("Expected FP32 RGB/DSM and bool valid")
        weight = self.backbone.embeddings.patch_embeddings.weight
        if not (rgb.device == dsm.device == valid.device == weight.device):
            raise ValueError("RGB, DSM, valid and model must share the device")
        if weight.dtype != torch.float32:
            raise TypeError("This implementation expects an FP32 model")

    def extract_features(self, rgb, dsm, valid, *, bypass=False):
        self._check_inputs(rgb, dsm, valid)
        if bypass:
            return self.backbone(rgb), None

        tokens = self.backbone.embeddings(rgb)
        rope = self.backbone.rope_embeddings(rgb)
        for index, block in enumerate(self.backbone.layer):
            tokens = block(tokens, position_embeddings=rope)
            if index == INSERT_AFTER:
                z_dsm, coverage = self.dsm_encoder(dsm, valid)
                output = self.bcia(tokens[:, PREFIX_TOKENS:], z_dsm, coverage)
                tokens = insert_residual(tokens, output.residual)
        spatial = self.backbone.norm(tokens)[:, PREFIX_TOKENS:]
        feature = spatial.reshape(rgb.shape[0], 32, 32, 1024).permute(0, 3, 1, 2).contiguous()
        return (feature,), output

    def forward(self, rgb, dsm, valid, *, bypass=False, return_adapter=False):
        features, output = self.extract_features(rgb, dsm, valid, bypass=bypass)
        logits = self.decode_head(features)
        return (logits, output) if return_adapter else logits

    def logits_for_tile(self, logits):
        return self.decode_head.predict_by_feat(logits, [dict(img_shape=(TILE_SIZE, TILE_SIZE))])

    def parameter_counts(self):
        return {
            name: dict(total=sum(param.numel() for param in module.parameters()),
                       trainable=sum(param.numel() for param in module.parameters()
                                     if param.requires_grad))
            for name, module in self.named_children()
        }
