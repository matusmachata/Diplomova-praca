import os
import random
from torchvision.transforms import ToTensor
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
import pytorch_lightning as pl
import torch
from torchvision import transforms
from PIL import Image
import tifffile as tiff
from glob import glob
from pytorch_lightning.callbacks import ModelCheckpoint
import re
from pathlib import Path
import ssl
from torch.optim.lr_scheduler import ReduceLROnPlateau

data_dir = "/kaggle/input/ddr-dataset-segmentation-only/lesion_segmentation/"
models_path = "/kaggle/working/"
model_path = "/kaggle/input/final_se/pytorch/default/1/wnet_final_se.pt"

train_loss_global = []
epochs = 200
lr = 0.008
accuracy_history = []
test_loss_history = []

# directory_path = 'C:/Users/Matus/Desktop/FEI/diplom-daj-sem-hned/testing'
file_prefix = 'model'  # Assuming your files are like 'file_1.txt', 'file_2.txt', etc.
file_suffix = '.pt'
image_suffix = '.png'
# Dataset and Dataloader
num = 29
output_size = (32 * num, 32 * num)
num_cuts_line = 3
patch_size = int(output_size[0] / (num / num_cuts_line))
transform = transforms.ToTensor()  # Convert PIL Image to Tensor
img_channel = 0  # je RGB
saturation_value = 110
BATCH = 2
eta_min = 0.0001
num_samples = 5
allChannels = True
encoder_name = "mobilenet_v2"
encoder_weights = "imagenet"
freeze_encoder = False
output_images = True

OUT_CLASSES = 1

img_channel_1 = 1
img_channel_2 = 2


def crop_and_resize(image, mask, size):
    """Crop image and mask to a square (max possible size) and resize to target size."""
    # Get original dimensions
    w, h = image.size
    min_dim = min(w, h)

    # Calculate crop box (center crop to square)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    # Crop and resize
    image = image.crop((left, top, right, bottom)).resize(size, Image.BILINEAR)
    mask = Image.fromarray(mask).crop((left, top, right, bottom)).resize(size, Image.NEAREST)

    return image, mask


def cut_image_pattern(image, patch_size, height, width):
    """Cuts the image into patches based on the red-line pattern."""
    step_size = patch_size // 2
    patches = []

    # Convert PIL Image to NumPy array
    image = np.array(image)

    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            if ((x // step_size) + (y // step_size)) % 2 == 0:  # Red pattern condition
                if x + patch_size <= width and y + patch_size <= height:
                    patch = image[y:y + patch_size, x:x + patch_size]
                    patches.append(patch)

    return patches


class OIADDRDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(448, 448), img_channel=0):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.tif")))
        self.transform = transform
        self.image_size = image_size
        self.img_channel = img_channel  # Channel selection

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = tiff.imread(self.mask_paths[idx])

        # Convert to numpy and select the specified channel
        # image = np.array(image)[:, :, self.img_channel]
        image = np.array(image)[:, :, [img_channel_1, img_channel_2]]

        # Convert back to PIL Image
        image = Image.fromarray(image)

        # Crop and resize
        image, mask = crop_and_resize(image, mask, self.image_size)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image).unsqueeze(0)  # Add channel dim

        mask = transforms.ToTensor()(mask).float()
        # print(f"Final image shape: {image.shape}")

        return image, mask


class OIADDRDatasetAllChannels(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=448, patch_size=64):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')])
        self.transform = transform
        self.image_size = output_size
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = tiff.imread(self.mask_paths[idx])

        # Resize & crop
        image, mask = crop_and_resize(image, mask, self.image_size)

        # Cut into patches
        image_patches = cut_image_pattern(image, self.patch_size, int(self.image_size[0]), int(self.image_size[1]))
        mask_patches = cut_image_pattern(mask, self.patch_size, int(self.image_size[0]),
                                         int(self.image_size[1]))  # Ensure single channel mask

        # Convert to tensors
        if self.transform:
            image_patches = [self.transform(img) for img in image_patches]
            mask_patches = [transforms.ToTensor()(mask).float() for mask in mask_patches]

        return image_patches, mask_patches


train_dataset = OIADDRDatasetAllChannels(os.path.join(data_dir, "train/image"),
                                         os.path.join(data_dir, "train/label/SE"),
                                         transform=transforms.ToTensor(), image_size=output_size, patch_size=patch_size)
val_dataset = OIADDRDatasetAllChannels(os.path.join(data_dir, "valid/image"),
                                       os.path.join(data_dir, "valid/segmentation_label/SE"),
                                       transform=transforms.ToTensor(), image_size=output_size, patch_size=patch_size)
test_dataset = OIADDRDatasetAllChannels(os.path.join(data_dir, "test/image"), os.path.join(data_dir, "test/label/SE"),
                                        transform=transforms.ToTensor(), image_size=output_size, patch_size=patch_size)


def collate_fn(batch):
    """Custom collate function to handle list of patches per image."""
    images, masks = zip(*batch)  # Separate images and masks
    images = [patch for sublist in images for patch in sublist]
    masks = [patch for sublist in masks for patch in sublist]
    return torch.stack(images), torch.stack(masks)


train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, collate_fn=collate_fn)

T_MAX = epochs * len(train_loader)


# print(T_MAX)

class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        self.jaccard = smp.losses.JaccardLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        self.focal = smp.losses.FocalLoss(mode=smp.losses.BINARY_MODE, alpha=0.25, gamma=2)

    def forward(self, y_pred, y_true):
        return self.dice(y_pred, y_true) + self.focal(y_pred, y_true) + self.jaccard(y_pred, y_true)
        # return self.dice(y_pred, y_true) + self.focal(y_pred, y_true)


class UNet(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,  # Specify encoder name
            encoder_weights=encoder_weights,  # Encoder weights
            in_channels=in_channels,  # Set input channels dynamically
            classes=1  # Number of output channels (adjust based on your task)
        )

    def forward(self, x):
        return self.model(x)


class Wnet(pl.LightningModule):
    def __init__(self, freeze_encoder=freeze_encoder, unfreeze_at_epoch=20):
        super(Wnet, self).__init__()
        self.unet1 = UNet(in_channels=3) if allChannels else UNet(in_channels=2)
        self.unet2 = UNet(in_channels=4)
        self.false_positive_images = 0
        self.total_negative_images = 0
        self.test_iou_history = []
        self.test_loss_history = []
        self.test_f1_history = []

        # self.loss_fn = smp.losses.JaccardLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        self.loss_fn = CombinedLoss()

        self.val_images = []
        self.num_val_images = 10
        self.valid_iou_history = []
        self.validation_step_outputs = []
        self.valid_loss_history = []
        self.avg_val_loss = []
        self.avg_val_iou = []

        self.freeze_encoder = freeze_encoder
        self.unfreeze_at_epoch = unfreeze_at_epoch

        if self.freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        """Freezes the encoder layers"""
        for param in self.unet1.model.encoder.parameters():  # Access encoder through `model`
            param.requires_grad = False
        print("Encoder is frozen.")

    def _unfreeze_encoder(self, layers_to_unfreeze=1):
        """Gradually unfreezes encoder layers by a specified number"""
        unfrozen_count = 0
        for idx, child in enumerate(self.unet1.model.encoder.children()):  # Access encoder through `model`
            if not any(param.requires_grad for param in child.parameters()):  # Check if still frozen
                for param in child.parameters():
                    param.requires_grad = True
                unfrozen_count += 1
            if unfrozen_count >= layers_to_unfreeze:
                break  # Stop after unfreezing the desired number
        print(f"Unfroze {unfrozen_count} more layers.")

    def forward(self, x):
        x1 = self.unet1(x)  # Output mask (1-channel)
        x2_input = torch.cat([x, x1], dim=1)  # Concatenate original image (3-channel) with x1 (1-channel) -> 4-channel
        x2 = self.unet2(x2_input)
        return x2

    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self.loss_fn(preds, masks)

        prob_mask = preds.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp2, fp2, fn2, tn2 = smp.metrics.get_stats(pred_mask.long(), masks.long(), mode="binary")
        per_image_iou2 = smp.metrics.iou_score(tp2, fp2, fn2, tn2, reduction="micro-imagewise")
        dataset_iou2 = smp.metrics.iou_score(tp2, fp2, fn2, tn2, reduction="micro")

        # tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), masks.long(), mode="binary")
        # per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        # dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        self.log("training_per_image_iou", per_image_iou2, prog_bar=True)
        self.log("training_dataset_iou", dataset_iou2, prog_bar=True)
        # self.log("testing_per_image_iou", per_image_iou, prog_bar=True)
        # self.log("testing_dataset_iou", dataset_iou, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self.loss_fn(preds, masks)
        self.valid_loss_history.append(loss.item())

        prob_mask = preds.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), masks.long(), mode="binary")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        self.valid_iou_history.append(dataset_iou.item())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", dataset_iou, prog_bar=True)

        # Store 5 random images, masks, and predictions for visualization
        if len(self.val_images) < 5:
            self.val_images.append((images.cpu(), masks.cpu(), preds.cpu()))
        elif random.random() < 0.1:  # Replace occasionally to get varied examples
            self.val_images[random.randint(0, 4)] = (images.cpu(), masks.cpu(), preds.cpu())

        return loss

    def on_train_epoch_start(self):
        """Unfreezes encoder gradually after a few epochs"""
        if self.current_epoch == self.unfreeze_at_epoch:
            self._unfreeze_encoder()

        total, frozen, trainable = self.count_frozen_layers()
        # print(f"Epoch {self.current_epoch}: Frozen layers: {frozen}/{total}, Trainable: {trainable}")

    def on_validation_epoch_end(self):
        if self.valid_loss_history:
            avg_val_loss = sum(self.valid_loss_history) / len(self.valid_loss_history)
            avg_val_iou = sum(self.valid_iou_history) / len(self.valid_iou_history)
            self.log("avg_val_loss", avg_val_loss, prog_bar=True)
            self.log("avg_val_iou", avg_val_iou, prog_bar=True)
            self.avg_val_loss.append(avg_val_loss)
            self.avg_val_iou.append(avg_val_iou)
            self.valid_loss_history.clear()
            self.valid_iou_history.clear()
            print("avg_val_loss", avg_val_loss)

        if not self.val_images:
            return

        if (self.current_epoch % 2 == 0 and output_images):
            fig, axes = plt.subplots(5, 3, figsize=(10, 15))
            fig.suptitle("Validation Samples")

            for i, (image, mask, pred) in enumerate(self.val_images):
                img = image[0].permute(1, 2, 0).numpy()
                mask = mask[0, 0].numpy()
                pred = torch.sigmoid(pred[0, 0]).numpy()

                axes[i, 0].imshow(img)
                axes[i, 0].set_title("Image")
                axes[i, 1].imshow(mask, cmap="gray")
                axes[i, 1].set_title("Ground Truth")
                axes[i, 2].imshow(pred, cmap="gray")
                axes[i, 2].set_title("Prediction")

                for ax in axes[i]:
                    ax.axis("off")

            plt.show()

        self.val_images.clear()

    def test_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self.loss_fn(preds, masks)

        prob_mask = preds.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp2, fp2, fn2, tn2 = smp.metrics.get_stats(pred_mask.long(), masks.long(), mode="binary")
        per_image_iou2 = smp.metrics.iou_score(tp2, fp2, fn2, tn2, reduction="micro-imagewise")
        # dataset_iou2 = smp.metrics.iou_score(tp2, fp2, fn2, tn2, reduction="micro")
        dataset_iou2 = smp.metrics.iou_score(tp2, fp2, fn2, tn2)
        f1_score = smp.metrics.f1_score(tp2, fp2, fn2, tn2)

        for i in range(masks.shape[0]):
            # resolution = images[i].shape[1:]  # (H, W)
            # print(f"Image {i} resolution: {resolution}")# Iterate over batch
            if masks[i].sum() == 0:  # Ground truth is completely empty
                self.total_negative_images += 1
                if pred_mask[i].sum() >= 1:  # Predicted at least 5 positive pixels
                    self.false_positive_images += 1

        # Compute percentage of false positive images
        false_positive_rate = (self.false_positive_images / self.total_negative_images) * 100

        # Log metrics
        self.log("testing_per_image_iou", per_image_iou2, prog_bar=True)
        self.log("testing_dataset_iou", dataset_iou2.mean(), prog_bar=True)
        self.log("test_f1_score", f1_score.mean(), prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)
        self.log("false_positive_rate", false_positive_rate, prog_bar=True)

        self.test_iou_history.append(dataset_iou2.mean())
        self.test_loss_history.append(loss.item())
        self.test_f1_history.append(f1_score.mean())

        return loss

    def on_test_epoch_end(self):
        """Compute false positive rate over the entire test dataset."""
        false_positive_rate = (self.false_positive_images / self.total_negative_images) * 100

        self.log("final_false_positive_rate", false_positive_rate, prog_bar=True)

        print(f"Final Testing loss: {(sum(self.test_loss_history) / len(self.test_loss_history)):.3f}")
        print(f"Final Testing IoU: {(sum(self.test_iou_history) / len(self.test_iou_history)):.3f}")
        print(f"Final Testing F1: {(sum(self.test_f1_history) / len(self.test_f1_history)):.3f}")
        print(f"Final False Positive Rate: {false_positive_rate:.3f}%")

    def count_frozen_layers(self):
        # Access encoder layers through the model's `encoder`
        total_layers = len(list(self.unet1.model.encoder.parameters()))  # Use `model.encoder`
        frozen_layers = sum(
            1 for param in self.unet1.model.encoder.parameters() if not param.requires_grad)  # Use `model.encoder`
        return total_layers, frozen_layers, total_layers - frozen_layers

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(T_MAX / 6), eta_min=eta_min)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.model.parameters(), lr=lr)


def get_next_file_name(directory, file_prefix, file_suffix):
    files = os.listdir(directory)
    pattern = re.compile(f'{re.escape(file_prefix)}_(\d+){re.escape(file_suffix)}')
    indices = [int(pattern.match(file).group(1)) for file in files if pattern.match(file)]
    next_index = max(indices, default=0) + 1
    return f'{file_prefix}_{next_index}{file_suffix}'


def save_model_state(model, directory, file_prefix, file_suffix):
    # Ensure the directory exists
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Get the next file name based on existing files
    new_file_name = get_next_file_name(directory, file_prefix, file_suffix)
    new_save_path = os.path.join(directory, new_file_name)

    # Save the model's state dictionary
    torch.save(model.state_dict(), new_save_path)
    print(f"Model saved at: {new_save_path}")


def plot_accuracy(model, name):
    plt.figure()
    plt.plot(model.avg_val_loss, label='loss počas trénovania')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('loss počas validacie za epochy')
    plt.legend()
    plt.savefig(name + '_loss' + '.png')
    plt.show()

    plt.figure()
    plt.plot(model.avg_val_iou, label='IoU počas validacie')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU imagewise počas validacie za epochy')
    plt.legend()
    plt.savefig(name + '_Imagewise IoU' + '.png')
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Wnet()
model.to(device)
state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.load_state_dict(state_dict, strict=False)

trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=4, precision=16)
test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
print(test_metrics)

num_full_images = 1
patches_per_image = int(num_cuts_line * 2 - 1)  # or whatever number you used during patching
indices = random.sample(range(0, len(test_dataset), patches_per_image), num_full_images)
index = random.randint(0, 224)
index = 224  # 172 91 151 224 223

fig, axs = plt.subplots(num_full_images, 3, figsize=(12, num_full_images * 4))


def reconstruct_image_from_patches(patches, patch_size, output_size):
    """
    Reconstruct the full image from the list of patches following the red-line pattern.
    """
    height, width = output_size
    step_size = patch_size // 2
    image = np.zeros((height, width, patches[0].shape[0]))  # (H, W, C)

    patch_idx = 0
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            if ((x // step_size) + (y // step_size)) % 2 == 0:
                if x + patch_size <= width and y + patch_size <= height:
                    patch = patches[patch_idx].permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
                    image[y:y + patch_size, x:x + patch_size] = patch
                    patch_idx += 1

    return image


model = model.to(device)
model.eval()

image_patches, mask_patches = test_dataset[index]

image_patches_batch = torch.stack(image_patches).to(device)
pred_patches = model(image_patches_batch)
pred_patches = (pred_patches > 0.5).float()

full_image = reconstruct_image_from_patches(image_patches, patch_size, output_size)
full_true_mask = reconstruct_image_from_patches(mask_patches, patch_size, output_size)
full_pred_mask = reconstruct_image_from_patches(pred_patches, patch_size, output_size)

axs[0].imshow(full_image, cmap="gray")
axs[0].set_title("Originálny obrázok")
axs[1].imshow(full_true_mask, cmap="gray")
axs[1].set_title("Správna maska")
axs[2].imshow(full_pred_mask, cmap="gray")
axs[2].set_title("Predikovaná maska")

plt.tight_layout()
plt.savefig("test_reconstructed.png")
plt.show()
