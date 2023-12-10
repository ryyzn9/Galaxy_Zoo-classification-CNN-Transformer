import torch
from CNN_T import CNNT

def test():
    v = CNNT(
        depth=6,
        num_classes=3,
        heads=8,
        mlp_dim=1024
    )

    img = torch.randn(1, 1, 32, 32)

    preds = v(img)
    # assert preds.shape == (1, 1000), 'correct logits outputted'

    print("the preads are :",preds)

test()