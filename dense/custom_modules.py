#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import os

import faiss
import torch
import numpy as np
from tqdm import tqdm

import gzip

class JsonGzCollectionIterator:
    def __init__(self, collection_paths: str, fields=None, docid_field=None, delimiter="\n"):
        if fields:
            self.fields = fields
        else:
            self.fields = ['text']
        self.docid_field = docid_field
        self.delimiter = delimiter
        self.all_info = self._load(collection_paths)
        self.size = len(self.all_info['id'])
        self.batch_size = 1
        self.shard_id = 0
        self.shard_num = 1

    def __call__(self, batch_size=1, shard_id=0, shard_num=1):
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.shard_num = shard_num
        return self

    def __iter__(self):
        total_len = self.size
        shard_size = int(total_len / self.shard_num)
        start_idx = self.shard_id * shard_size
        end_idx = min(start_idx + shard_size, total_len)
        if self.shard_id == self.shard_num - 1:
            end_idx = total_len
        to_yield = {}
        for idx in tqdm(range(start_idx, end_idx, self.batch_size)):
            for key in self.all_info:
                to_yield[key] = self.all_info[key][idx: min(idx + self.batch_size, end_idx)]
            yield to_yield

    def _parse_fields_from_info(self, info):
        """
        :params info: dict, containing all fields as speicifed in self.fields either under 
        the key of the field name or under the key of 'contents'.  If under `contents`, this 
        function will parse the input contents into each fields based the self.delimiter
        return: List, each corresponds to the value of self.fields
        """
        n_fields = len(self.fields)

        # if all fields are under the key of info, read these rather than 'contents' 
        if all([field in info for field in self.fields]):
            return [info[field].strip() for field in self.fields]

        assert "contents" in info, f"contents not found in info: {info}"
        contents = info['contents']
        # whether to remove the final self.delimiter (especially \n)
        # in CACM, a \n is always there at the end of contents, which we want to remove;
        # but in SciFact, Fiqa, and more, there are documents that only have title but not text (e.g. "This is title\n")
        # where the trailing \n indicates empty fields
        if contents.count(self.delimiter) == n_fields:
            # the user appends one more delimiter to the end, we remove it
            if contents.endswith(self.delimiter):
                # not using .rstrip() as there might be more than one delimiters at the end
                contents = contents[:-len(self.delimiter)]
        return [field.strip(" ") for field in contents.split(self.delimiter)]

    def _load(self, collection_paths):
        filenames = []
        
        if os.path.isfile(collection_paths):
            filenames.append(collection_paths)
        else:
            for filename in os.listdir(collection_paths):
                if filename[-8:] == ".json.gz":
                    filenames.append(os.path.join(collection_paths, filename))


        #for collection_path in collection_paths:
        #    if os.path.isfile(collection_path):
        #        filenames.append(collection_path)
        #    else:
        #        for filename in os.listdir(collection_path):
        #            if filename[-8:] == ".json.gz":
        #                filenames.append(os.path.join(collection_path, filename))
        
        self.num_files = len(filenames)
        print ("Reading "+str(len(filenames))+" files")
        
        all_info = {field: [] for field in self.fields}
        all_info['id'] = []
        for filename in filenames:
            with gzip.open(filename) as f:
                for line_i, line in tqdm(enumerate(f)):
                    info = json.loads(line)
                    if self.docid_field:
                        _id = info.get(self.docid_field, None)
                    else:
                        _id = info.get('id', info.get('_id', info.get('docid', None)))
                    if _id is None:
                        raise ValueError(f"Cannot find f'`{self.docid_field if self.docid_field else '`id` or `_id` or `docid'}`' from {filename}.")
                    all_info['id'].append(str(_id))
                    fields_info = self._parse_fields_from_info(info)
                    if len(fields_info) != len(self.fields):
                        raise ValueError(
                            f"{len(fields_info)} fields are found at Line#{line_i} in file {filename}." \
                            f"{len(self.fields)} fields expected." \
                            f"Line content: {info['contents']}"
                        )

                    for i in range(len(fields_info)):
                        all_info[self.fields[i]].append(fields_info[i])
        return all_info
