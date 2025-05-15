# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Hatexplain: A Benchmark Dataset for Explainable Hate Speech Detection"""

import json
import os
import datasets


_CITATION = """\
@misc{mathew2020hatexplain,
      title={HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection},
      author={Binny Mathew and Punyajoy Saha and Seid Muhie Yimam and Chris Biemann and Pawan Goyal and Animesh Mukherjee},
      year={2020},
      eprint={2012.10289},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
Hatexplain is the first benchmark hate speech dataset covering multiple aspects of the issue. \
Each post in the dataset is annotated from three different perspectives: the basic, commonly used 3-class classification \
(i.e., hate, offensive or normal), the target community (i.e., the community that has been the victim of \
hate speech/offensive speech in the post), and the rationales, i.e., the portions of the post on which their labelling \
decision (as hate, offensive or normal) is based.
"""
_LICENSE = "cc-by-4.0"

_URL = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/"
_URLS = {
    "dataset": _URL + "dataset.json",
    "post_id_divisions": _URL + "post_id_divisions.json",
}


class HatexplainConfig(datasets.BuilderConfig):
    """BuilderConfig for Hatexplain."""

    def __init__(self, **kwargs):
        """BuilderConfig for Hatexplain.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(HatexplainConfig, self).__init__(**kwargs)


class Hatexplain(datasets.GeneratorBasedBuilder):
    """Hatexplain: A Benchmark Dataset for Explainable Hate Speech Detection"""

    BUILDER_CONFIGS = [
        HatexplainConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "annotators": datasets.features.Sequence(
                        {
                            "label": datasets.ClassLabel(names=["hatespeech", "normal", "offensive"]),
                            "annotator_id": datasets.Value("int32"),
                            "target": datasets.Sequence(datasets.Value("string")),
                        }
                    ),
                    "rationales": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("int32"))),
                    "post_tokens": datasets.features.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files, "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files, "split": "val"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files, "split": "test"}
            ),
        ]

    def _generate_examples(self, filepath, split):
        """This function returns the examples depending on split"""

        with open(filepath["post_id_divisions"], encoding="utf-8") as f:
            post_id_divisions = json.load(f)
        with open(filepath["dataset"], encoding="utf-8") as f:
            dataset = json.load(f)

        for id_, tweet_id in enumerate(post_id_divisions[split]):
            info = dataset[tweet_id]
            annotators, rationales, post_tokens = info["annotators"], info["rationales"], info["post_tokens"]

            yield id_, {"id": tweet_id, "annotators": annotators, "rationales": rationales, "post_tokens": post_tokens}

# What i wrote
def main():
    data = Hatexplain()
    print(type(data))

    # Set custom download and cache directories
    custom_directory = "Datasets/hatexplain_data"
    # Create the directory if it doesn't exist
    os.makedirs(custom_directory, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = custom_directory

    print(f"Using cache directory: {custom_directory}")

    # Initialize the dataset builder
    dataset_builder = Hatexplain()

    # Print download URLs
    for name, url in _URLS.items():
        print(f"Will download {name} from {url}")

    # Download and prepare the dataset
    dataset_builder.download_and_prepare()

    # Get the dataset
    dataset = dataset_builder.as_dataset()

    # Check where files should be stored
    print(f"Cache directory according to environment variable: {os.environ['HF_DATASETS_CACHE']}")

    # Access specific splits
    train_dataset = dataset["train"]
    validation_dataset = dataset["val"]
    test_dataset = dataset["test"]

    # print the first as example
    print(train_dataset[0])

    # Add this section to convert and save in your desired format
    for split_name, split_dataset in dataset.items():
        output_file = os.path.join(custom_directory, f"{split_name}_formatted.json")

        # Convert the format
        formatted_data = {}
        for item in split_dataset:
            post_id = item["id"]

            # Create the properly structured annotators list
            annotators_list = []
            for i in range(len(item["annotators"]["label"])):
                label_value = item["annotators"]["label"][i]
                # Convert numeric label to string label
                if label_value == 0:
                    label_str = "hatespeech"
                elif label_value == 1:
                    label_str = "normal"
                elif label_value == 2:
                    label_str = "offensive"
                else:
                    label_str = str(label_value)

                annotators_list.append({
                    "label": label_str,
                    "annotator_id": item["annotators"]["annotator_id"][i],
                    "target": item["annotators"]["target"][i]
                })

            # Build the entry in the format you want
            formatted_data[post_id] = {
                "post_id": post_id,
                "annotators": annotators_list,
                "rationales": item["rationales"],
                "post_tokens": item["post_tokens"]
            }

        # Save the formatted data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2)

        print(f"Saved formatted {split_name} split to {output_file}")

    return dataset

if __name__ == "__main__":
    main()