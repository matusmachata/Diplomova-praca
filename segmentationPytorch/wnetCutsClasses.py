import os
import random
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import tifffile as tiff
from glob import glob
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from torch.optim import lr_scheduler
import re
from pathlib import Path
import ssl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
import matplotlib.colors as mcolors

# ssl._create_default_https_context = ssl._create_unverified_context

# tato one channel haluz berie strasne vela vram
# data_dir = "/kaggle/input/ddr-dataset-segmentation-only/"
data_dir = "/kaggle/input/ddr-dataset-segmentation-only/lesion_segmentation/"
models_path = "/kaggle/working/"

train_loss_global = []
epochs = 100
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
# patch_size = 32*int(num/num_cuts_line)
print(patch_size)
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
classes = 3
mode = "3-class"
foreground_source = "SE"

img_channel_1 = 0
img_channel_2 = 1

OUT_CLASSES = 1


def crop_and_resize(image, masks, size):
    """Crop image and all masks to square and resize."""
    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    image = image.crop((left, top, right, bottom)).resize(size, Image.BILINEAR)

    resized_masks = []
    for mask in masks:
        mask_img = Image.fromarray(mask.astype(np.uint8)).crop((left, top, right, bottom)).resize(size, Image.NEAREST)
        resized_masks.append(np.array(mask_img))

    return image, resized_masks


def cut_image_pattern(array, patch_size, height, width):
    """Cut 2D or 3D numpy array into red-line pattern patches."""
    step_size = patch_size // 2
    patches = []

    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            if ((x // step_size) + (y // step_size)) % 2 == 0:
                if x + patch_size <= width and y + patch_size <= height:
                    patch = array[y:y + patch_size, x:x + patch_size]
                    patches.append(patch)

    return patches


class OIADDRDatasetAllChannels(Dataset):
    def __init__(self, image_dir, mask_dir_EX, mask_dir_SE, transform=None, image_size=(448, 448), patch_size=64,
                 mode="3-class", foreground_source=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.mask_paths_EX = sorted(
            [os.path.join(mask_dir_EX, f) for f in os.listdir(mask_dir_EX) if f.endswith('.tif')])
        self.mask_paths_SE = sorted(
            [os.path.join(mask_dir_SE, f) for f in os.listdir(mask_dir_SE) if f.endswith('.tif')])
        self.transform = transform
        self.image_size = image_size
        self.patch_size = patch_size
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask_ex = tiff.imread(self.mask_paths_EX[idx])
        mask_se = tiff.imread(self.mask_paths_SE[idx])

        # Binarize masks
        mask_ex = (mask_ex > 0).astype(np.uint8)
        mask_se = (mask_se > 0).astype(np.uint8)

        # Apply crop & resize to both masks
        image, [mask_ex, mask_se] = crop_and_resize(image, [mask_ex, mask_se], self.image_size)

        # Build combined mask
        if self.mode == "2-class":
            if hasattr(self, "foreground_source"):
                if self.foreground_source == "EX":
                    mask = mask_ex
                elif self.foreground_source == "SE":
                    mask = mask_se
                else:
                    raise ValueError("foreground_source must be 'EX' or 'SE' when using 2-class mode")
            else:
                # default behavior: combine both
                mask = ((mask_ex | mask_se) > 0).astype(np.uint8)

        elif self.mode == "3-class":
            mask = np.zeros_like(mask_ex, dtype=np.uint8)
            mask[mask_ex == 1] = 1
            mask[mask_se == 1] = 2

        else:
            raise ValueError("Mode must be '2-class' or '3-class'")

        # Cut to patches
        image_np = np.array(image)
        image_patches = cut_image_pattern(image_np, self.patch_size, *self.image_size)
        mask_patches = cut_image_pattern(mask, self.patch_size, *self.image_size)

        # Transform to tensors
        if self.transform:
            image_patches = [self.transform(Image.fromarray(p)) for p in image_patches]
            mask_patches = [torch.tensor(p, dtype=torch.long) for p in mask_patches]

        return image_patches, mask_patches


train_dataset = OIADDRDatasetAllChannels(
    os.path.join(data_dir, "train/image"),
    os.path.join(data_dir, "train/label/EX"),
    os.path.join(data_dir, "train/label/SE"),
    transform=transforms.ToTensor(), image_size=output_size,
    patch_size=patch_size, mode=mode,
    foreground_source=foreground_source
)
val_dataset = OIADDRDatasetAllChannels(
    os.path.join(data_dir, "valid/image"),
    os.path.join(data_dir, "valid/segmentation_label/EX"),
    os.path.join(data_dir, "valid/segmentation_label/SE"),
    transform=transforms.ToTensor(), image_size=output_size,
    patch_size=patch_size, mode=mode,
    foreground_source=foreground_source
)
test_dataset = OIADDRDatasetAllChannels(
    os.path.join(data_dir, "test/image"),
    os.path.join(data_dir, "test/label/EX"),
    os.path.join(data_dir, "test/label/SE"),
    transform=transforms.ToTensor(), image_size=output_size,
    patch_size=patch_size, mode=mode,
    foreground_source=foreground_source
)


# val_dataset = OIADDRDatasetAllChannels(os.path.join(data_dir, "valid/image"),
#                             os.path.join(data_dir, "valid/segmentation_label/EX"), transform=transforms.ToTensor(), image_size=output_size, patch_size=patch_size)
# test_dataset = OIADDRDatasetAllChannels(os.path.join(data_dir, "test/image"), os.path.join(data_dir, "test/label/EX"),
#                              transform=transforms.ToTensor(), image_size=output_size, patch_size=patch_size)


def collate_fn(batch):
    """Custom collate function to handle list of patches per image."""
    images, masks = zip(*batch)  # Separate images and masks
    images = [patch for sublist in images for patch in sublist]
    masks = [patch for sublist in masks for patch in sublist]
    return torch.stack(images), torch.stack(masks)


train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, collate_fn=collate_fn)

T_MAX = epochs * len(train_loader)


# print(T_MAX)

class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True)
        self.jaccard = smp.losses.JaccardLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True)
        self.focal = smp.losses.FocalLoss(mode=smp.losses.MULTICLASS_MODE, alpha=0.25, gamma=2)

    def forward(self, y_pred, y_true):
        return self.dice(y_pred, y_true) + self.focal(y_pred, y_true) + self.jaccard(y_pred, y_true)


class UNet(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,  # Specify encoder name
            encoder_weights=encoder_weights,  # Encoder weights
            in_channels=in_channels,  # Set input channels dynamically
            classes=classes  # Number of output channels (adjust based on your task)
        )

    def forward(self, x):
        return self.model(x)


class Wnet(pl.LightningModule):
    def __init__(self, freeze_encoder=freeze_encoder, unfreeze_at_epoch=20):
        super(Wnet, self).__init__()
        self.unet1 = UNet(in_channels=3) if allChannels else UNet(in_channels=2)
        self.unet2 = UNet(in_channels=6)
        self.false_positive_images = 0
        self.total_negative_images = 0
        self.test_iou_history = []
        self.test_loss_history = []
        self.test_f1_history = []
        self.lr_history = []

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

        # prob_mask = preds.sigmoid()
        # pred_mask = (prob_mask > 0.5).float()
        pred_mask = torch.argmax(preds, dim=1)
        tp2, fp2, fn2, tn2 = smp.metrics.get_stats(pred_mask.long(), masks.long(), mode="multiclass",
                                                   num_classes=classes)
        per_image_iou2 = smp.metrics.iou_score(tp2, fp2, fn2, tn2, reduction="micro-imagewise")
        dataset_iou2 = smp.metrics.iou_score(tp2, fp2, fn2, tn2, reduction="micro")

        self.log("training_per_image_iou", per_image_iou2, prog_bar=True)
        self.log("training_dataset_iou", dataset_iou2, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self.loss_fn(preds, masks)
        self.valid_loss_history.append(loss.item())

        # prob_mask = preds.sigmoid()
        # pred_mask = (prob_mask > 0.5).float()
        pred_mask = torch.argmax(preds, dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), masks.long(), mode="multiclass", num_classes=classes)
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        self.valid_iou_history.append(dataset_iou.item())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", dataset_iou, prog_bar=True)

        rand_idx = random.randint(0, images.shape[0] - 1)

        image_patch = images[rand_idx].cpu()
        mask_patch = masks[rand_idx].cpu()
        pred_patch = preds[rand_idx].cpu()

        # Store 5 random images, masks, and predictions for visualization
        if len(self.val_images) < self.num_val_images:
            self.val_images.append((image_patch, mask_patch, pred_patch))
        elif random.random() < 0.2:  # ~20% chance to replace existing one
            self.val_images[random.randint(0, self.num_val_images - 1)] = (image_patch, mask_patch, pred_patch)

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

            self.lr_history.append(self.trainer.optimizers[0].param_groups[0]['lr'])

        if not self.val_images:
            return

        if len(self.val_images) > 0:
            num_to_plot = min(4, len(self.val_images))  # only plot a few to save memory
            for i in range(num_to_plot):
                input_image, true_mask, pred_mask = self.val_images[i]

                input_image = input_image.cpu().permute(1, 2, 0).numpy()
                true_mask = true_mask.cpu().numpy()
                pred_mask = torch.argmax(pred_mask, dim=0).cpu().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(input_image)
                axs[0].set_title("Originálny obrázok")
                axs[1].imshow(true_mask, cmap='jet', interpolation='nearest', vmin=0, vmax=2)
                axs[1].set_title("Správna maska")
                axs[2].imshow(pred_mask, cmap='jet', interpolation='nearest', vmin=0, vmax=2)
                axs[2].set_title("Predikovaná maska")
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                plt.show()

            self.val_images.clear()  # clear memory

    def test_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self.loss_fn(preds, masks)

        # prob_mask = preds.sigmoid()
        # pred_mask = (prob_mask > 0.5).float()
        pred_mask = torch.argmax(preds, dim=1)
        tp2, fp2, fn2, tn2 = smp.metrics.get_stats(pred_mask.long(), masks.long(), mode="multiclass",
                                                   num_classes=classes)
        per_image_iou2 = smp.metrics.iou_score(tp2, fp2, fn2, tn2, reduction="micro-imagewise")
        # dataset_iou2 = smp.metrics.iou_score(tp2, fp2, fn2, tn2, reduction="micro")
        dataset_iou2 = smp.metrics.iou_score(tp2, fp2, fn2, tn2)
        f1_score = smp.metrics.f1_score(tp2, fp2, fn2, tn2)

        for i in range(masks.shape[0]):  # Iterate over batch
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

        print(f"Final Testing loss: {(sum(self.test_loss_history) / len(self.test_loss_history)):.2f}")
        print(f"Final Testing IoU: {(sum(self.test_iou_history) / len(self.test_iou_history)):.2f}")
        print(f"Final Testing F1: {(sum(self.test_f1_history) / len(self.test_f1_history)):.2f}")
        print(f"Final False Positive Rate: {false_positive_rate:.2f}%")

    def count_frozen_layers(self):
        # Access encoder layers through the model's `encoder`
        total_layers = len(list(self.unet1.model.encoder.parameters()))  # Use `model.encoder`
        frozen_layers = sum(
            1 for param in self.unet1.model.encoder.parameters() if not param.requires_grad)  # Use `model.encoder`
        return total_layers, frozen_layers, total_layers - frozen_layers

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # Start at max lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_MAX, eta_min=eta_min  # Ends at min lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(T_MAX/6), eta_min=eta_min)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        # }
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
    plt.plot(model.avg_val_loss, label='Priemerná validačná strata', color='red')
    plt.plot(model.avg_val_iou, label='Priemerné validačné IoU', color='blue')
    # plt.plot(model.lr_history, label='Priebeh LR počas trénovania', color='green')
    plt.xlabel('Epoch [-]')
    plt.ylabel('Hodnota [-]')
    plt.legend()
    plt.savefig('train_loss_iou.png')
    plt.show()
    # plt.figure()
    # plt.plot(model.avg_val_loss, label='loss počas trénovania')
    # plt.xlabel('Epoch')
    # plt.ylabel('loss')
    # plt.title('loss počas validacie za epochy')
    # plt.legend()
    # plt.savefig(name + '_loss' + '.png')
    # plt.show()

    plt.figure()
    plt.plot(model.lr_history, label='Priebeh LR počas trénovania', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('LR počas validacie za epochy')
    plt.legend()
    plt.savefig('LR' + '.png')
    plt.show()


# Model training
model = Wnet()
checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=0, mode='min')
trainer = pl.Trainer(max_epochs=epochs, enable_progress_bar=True,
                     accelerator='gpu' if torch.cuda.is_available() else 'cpu', callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)

file_name = os.path.basename(models_path)
new_file_name = get_next_file_name(models_path, file_prefix, file_suffix)
new_save_path = models_path + '/' + new_file_name
save_model_state(model, models_path, file_prefix, file_suffix)
plot_accuracy(model, new_file_name)

trainer.test(model, test_loader)
