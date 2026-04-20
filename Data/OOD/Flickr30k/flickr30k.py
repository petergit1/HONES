# coding=utf-8
# Copyright 2022 the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd 
import datasets
import json
from huggingface_hub import hf_hub_url

_INPUT_CSV = "flickr_annotations_30k.csv"
_INPUT_IMAGES = "flickr30k-images"
_REPO_ID = "nlphuji/flickr30k"
_JSON_KEYS = ['raw', 'sentids']

class Dataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="TEST", version=VERSION, description="test"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                 {
                "image": datasets.Image(),
                "caption": [datasets.Value('string')],
                "sentids": [datasets.Value("string")],
                "split": datasets.Value("string"),
                 "img_id": datasets.Value("string"),
                "filename": datasets.Value("string"),
                }
            ),
            task_templates=[],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        repo_id = _REPO_ID
        data_dir = dl_manager.download_and_extract({
            "examples_csv": hf_hub_url(repo_id=repo_id, repo_type='dataset', filename=_INPUT_CSV),
            "images_dir": hf_hub_url(repo_id=repo_id, repo_type='dataset', filename=f"{_INPUT_IMAGES}.zip")
        })

        return [datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=data_dir)]


    def _generate_examples(self, examples_csv, images_dir):
        """Yields examples."""
        df = pd.read_csv(examples_csv)
        for c in _JSON_KEYS:
            df[c] = df[c].apply(json.loads)

        for r_idx, r in df.iterrows():
            r_dict = r.to_dict()
            image_path = os.path.join(images_dir, _INPUT_IMAGES, r_dict['filename'])
            r_dict['image'] = image_path
            r_dict['caption'] = r_dict.pop('raw')
            yield r_idx, r_dict