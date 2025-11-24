import torch
import numpy as np
import os
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    SpatialPadd, ToTensord
)
from text_encoder import ClinicalTextEncoder


class TextPromptComposer:
  """ Compose text prompts for normal and abnormal categories"""

  def __init__(self, modality = 'CT Scan/MRI'):
    self.modality = modality
    self.normal_template = 'A normal {modality} of {organ_name}'
    self.abnormal_template = 'An abnormal {modality} of {disease_name}'
    
    # Organs and diseases
    self.organs = ['pancreas', 'lung', 'liver', 'colon', 'hepatic vessel']
    self.diseases_dict = {'pancreas': 'pancreatic tumor', 'lung': 'lung tumor', 'liver': 'liver tumor', 'colon': 'colon tumor', 'hepatic vessel': 'vascular tumor'}

  def generate_prompts(self, organ_name):
    """Generate both normal and abnormal prompts"""

    prompts = {'normal': self.normal_template.format(modality = self.modality, organ_name = organ_name),
               'abnormal': self.abnormal_template.format(modality = self.modality, disease_name = self.diseases_dict[organ_name])}

    return prompts
  
class FullVolumeTransform:
    """Applies MONAI preprocessing transforms and assigns class label."""

    def __init__(self):
        self.transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-100, a_max=400,
                b_min=0.0, b_max=1.0, clip=True
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 128)),
            ToTensord(keys=["image", "label"])
        ])

    def __call__(self, item):
        item = self.transforms(item)

        # Compute class label based on sum of mask tensor
        label_tensor = item["label"]
        class_label = 0 if torch.sum(label_tensor) == 0 else 1
        item["class_label"] = class_label

        return item


class LoadDataset:
    """Load the dataset containing image path, label path, prompts, embeddings, and class label"""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.prompt_composer = TextPromptComposer()
        self.text_encoder = ClinicalTextEncoder()
        self.transform = FullVolumeTransform()  

    def load_dataset(self):
        dataset_list = []

        for organ in [d for d in os.listdir(self.data_dir) if d != ".ipynb_checkpoints"]:
            organ_dir_path = os.path.join(self.data_dir, organ)
            if os.path.isdir(organ_dir_path):

                images_path = os.path.join(organ_dir_path, "imagesTr")
                labels_path = os.path.join(organ_dir_path, "labelsTr")

                for image_file in os.listdir(images_path):
                    if image_file.startswith(organ) and image_file in os.listdir(labels_path):
                        prompts = self.prompt_composer.generate_prompts(organ)

                        entry = {
                            "image": os.path.join(images_path, image_file),
                            "label": os.path.join(labels_path, image_file),
                            "prompts": prompts,
                            "embeddings": self.text_encoder.encode_text(list(prompts.values()))
                        }

                        # Apply the transformation class here
                        entry = self.transform(entry)

                        dataset_list.append(entry)

        return dataset_list

