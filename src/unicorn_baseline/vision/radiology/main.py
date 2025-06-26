#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from unicorn_baseline.io import resolve_image_path, write_json_file
from unicorn_baseline.vision.radiology.models.ctfm import SegResNetEncoder
from unicorn_baseline.vision.radiology.patch_extraction import extract_patches
from picai_prep.preprocessing import Sample, PreprocessingSettings
from unicorn_baseline.vision.radiology.models.mrsegmentator import MRSegmentator

def extract_features_classification(
    image,
    model,
    domain: str,
    title: str = "image-level-neural-representation",
) -> dict:
    image_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())
    if (image_orientation != 'SPL') and (domain == 'CT'): 
        image = sitk.DICOMOrient(image, desiredCoordinateOrientation='SPL')

    image_array = sitk.GetArrayFromImage(image)
    image_features = model.encode(image_array)

    image_level_neural_representation = make_patch_level_neural_representation(
        image_features=image_features,
        title=title,
    )
    return image_level_neural_representation


def extract_features_segmentation(
    image,
    model,
    domain: str,
    title: str = "patch-level-neural-representation",
    patch_size: list[int] = [16, 64, 64],
    patch_spacing: list[float] | None = None,
) -> list[dict]:
    """
    Generate a list of patch features from a radiology image
    """
    patch_features = []
    
    image_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())
    if (image_orientation != 'SPL') and (domain == 'CT'): 
        image = sitk.DICOMOrient(image, desiredCoordinateOrientation='SPL')
    if (image_orientation != 'LPS') and (domain == 'MR'): 
        image = sitk.DICOMOrient(image, desiredCoordinateOrientation='LPS')
    

    print(f"Extracting patches from image")
    patches, coordinates = extract_patches(
        image=image,
        patch_size=patch_size,
        spacing=patch_spacing,
    )
    if patch_spacing is None:
        patch_spacing = image.GetSpacing()

    MAX_LEN = 1024
    print(f"Extracting features from patches")
    for patch, coords in tqdm(zip(patches, coordinates), total=len(patches), desc="Extracting features"):
        patch_array = sitk.GetArrayFromImage(patch)
        full_feat = model.encode(patch_array)

        n = (len(full_feat) + MAX_LEN - 1) // MAX_LEN

        for i in range(n):
            start = i * MAX_LEN
            end   = min(start + MAX_LEN, len(full_feat))
            chunk = full_feat[start:end]

            # build a new 4-coordinate: [x, y, z, chunk_index]
            coord4 = coords[0] + (i,)
            patch_features.append({
                "coordinates": coord4,
                "features":    chunk
            })

    patch_level_neural_representation = make_patch_level_neural_representation(
        image_features=patch_features,
        patch_size=patch_size,
        patch_spacing=patch_spacing,
        image_size=image.GetSize(),
        image_origin=image.GetOrigin(),
        image_spacing=image.GetSpacing(),
        image_direction=image.GetDirection(),
        title=title,
    )
    return patch_level_neural_representation


def make_patch_level_neural_representation(
    *,
    title: str,
    image_features: Iterable[dict],
    patch_size: Iterable[int] = None,
    patch_spacing: Iterable[float] = None,
    image_size: Iterable[int] = None,
    image_spacing: Iterable[float] = None,
    image_origin: Iterable[float] = None,
    image_direction: Iterable[float] = None,
) -> dict:
    if patch_size is not None:
        if image_origin is None:
            image_origin = [0.0] * len(image_size)
        if image_direction is None:
            image_direction = np.identity(len(image_size)).flatten().tolist()
        return {
            "meta": {
                "patch-size": list(patch_size),
                "patch-spacing": list(patch_spacing),
                "image-size": list(image_size),
                "image-origin": list(image_origin),
                "image-spacing": list(image_spacing),
                "image-direction": list(image_direction),
            },
            "patches": list(image_features),
            "title": title,
        }
    else: 
        return {
            "title": title,
            "features": image_features
        }


def run_radiology_vision_task(
    *,
    task_type: str,
    input_information: dict[str, Any],
    model_dir: Path,
    domain: str,
):
    # Identify image inputs
    image_inputs = []
    for input_socket in input_information:
        if input_socket["interface"]["kind"] == "Image":
            image_inputs.append(input_socket)

    if domain == 'CT':
        model = SegResNetEncoder(Path(model_dir, "ct_fm_feature_extractor")) 
    if domain == 'MR': 
        model = MRSegmentator(Path(model_dir, "mrsegmentator"))

    output_dir = Path("/output")
    neural_representations = []

    if image_inputs[0]['interface']['slug'].endswith('prostate-mri'):
        images_to_preprocess = {}
        for image_input in image_inputs:
            image_path = resolve_image_path(location=image_input["input_location"])
            print(f"Reading image from {image_path}")
            image = sitk.ReadImage(str(image_path))

            if 't2' in str(image_input["input_location"]): 
                images_to_preprocess.update({'t2' : image})
            if 'hbv' in str(image_input["input_location"]): 
                images_to_preprocess.update({'hbv' : image})
            if 'adc' in str(image_input["input_location"]): 
                images_to_preprocess.update({'adc' : image})

        pat_case = Sample(scans=[images_to_preprocess.get('t2'), images_to_preprocess.get('hbv'), images_to_preprocess.get('adc')], settings=PreprocessingSettings(spacing=[3,1.5,1.5], matrix_size=[16,256,256]))
        pat_case.preprocess()
            
        for image in pat_case.scans:
            neural_representation = extract_features_segmentation(
                        image=image,
                        model=model, 
                        domain=domain,
                        title=image_input["interface"]["slug"]
                )
            neural_representations.append(neural_representation)
    else: 
        for image_input in image_inputs:
            image_path = resolve_image_path(location=image_input["input_location"])
            print(f"Reading image from {image_path}")
            image = sitk.ReadImage(str(image_path))

            if task_type == "classification": 
                neural_representation = extract_features_classification(
                        image=image,
                        model=model, 
                        domain=domain,
                        title=image_input["interface"]["slug"]
                    )
            else:
                neural_representation = extract_features_segmentation(
                        image=image,
                        model=model, 
                        domain=domain,
                        title=image_input["interface"]["slug"]
                    )
            neural_representations.append(neural_representation)
    if task_type == "classification": 
        output_path = output_dir / "image-neural-representation.json"
    else:
        output_path = output_dir / "patch-neural-representation.json"
    write_json_file(location=output_path, content=neural_representations)