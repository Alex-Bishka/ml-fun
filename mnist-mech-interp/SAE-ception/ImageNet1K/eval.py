# import os
# os.environ['HF_DATASETS_CACHE'] = '/mnt/SharedData/.hugging-face-cache'

import torch
import timm
from torchvision import transforms
from datasets import load_dataset, load_from_disk

# model = timm.create_model(
#     'convnextv2_large.fcmae_ft_in22k_in1k_384',
#     pretrained=True,
#     num_classes=1000  # ImageNet-1K
# )
# torch.save(model.state_dict(), './convnextv2_large.fcmae_ft_in22k_in1k_384_baseline.pth')

# train_split = load_dataset("ILSVRC/imagenet-1k", split='train', cache_dir=None)
# train_split.save_to_disk('./data/full-train')
# print(train_split)

train_split = load_dataset("ILSVRC/imagenet-1k", split='train', cache_dir='./data')
print(train_split)

validation_split = load_dataset("ILSVRC/imagenet-1k", split='validation', cache_dir='./data')
print(validation_split)

test_split = load_dataset("ILSVRC/imagenet-1k", split='test', cache_dir='./data')
print(test_split)


if False:
    tfms = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=model.default_cfg['mean'],
                            std=model.default_cfg['std'])
    ])
    # data_config = timm.data.resolve_model_data_config(model)
    # tfms = timm.data.create_transform(**data_config, is_training=False)

    def collate(batch):
        imgs = torch.stack([tfms(x['image']) for x in batch]).cuda()
        labels = torch.tensor([x['label'] for x in batch]).cuda()
        return imgs, labels

    # loader = torch.utils.data.DataLoader(train_split, batch_size=32, collate_fn=collate)
    loader = torch.utils.data.DataLoader(validation_split, batch_size=32, collate_fn=collate)

    model.eval().cuda()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            print("Labels from this batch:", labels)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {correct/total*100:.2f}%") 