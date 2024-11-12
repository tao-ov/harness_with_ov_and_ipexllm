# openvino.py

# Copyright 2016 The BigDL Authors.
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
#

import inspect
from functools import partial
from lm_eval.models.huggingface import AutoCausalLM
from lm_eval import utils
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoConfig

# Ensure Reorderer is in decreasing order to minimize memory allocation frequency
def force_decrease_order(Reorderer):
    def DecreaseReorderer(arr, fn):
        def _collate(x):
            length, tokens = fn(x)
            length = -abs(length)
            return length, tokens
        return Reorderer(arr, _collate)
    return DecreaseReorderer
utils.Reorderer = force_decrease_order(utils.Reorderer)

class OpenVINOLLM(AutoCausalLM):
    AUTO_MODEL_CLASS = OVModelForCausalLM
    AutoCausalLM_ARGS = inspect.getfullargspec(AutoCausalLM.__init__).args

    def __init__(self, *args, **kwargs):
        self.ov_model_kwargs = {}
        keys = list(kwargs.keys())
        
        # Extract specialized parameters for OV models
        for k in keys:
            if k not in self.AutoCausalLM_ARGS:
                self.ov_model_kwargs[k] = kwargs.pop(k)
        
        # Set default configuration
        self.ov_model_kwargs['use_cache'] = self.ov_model_kwargs.get('use_cache', True)
        
        # Attempt to load model
        try:
            OVModelForCausalLM.from_pretrained = partial(OVModelForCausalLM.from_pretrained, **self.ov_model_kwargs, trust_remote_code=True)
        except:
            from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
            from optimum.exporters import TasksManager
            
            # Register new model configuration
            TasksManager._SUPPORTED_MODEL_TYPE["stablelm-epoch"] = TasksManager._SUPPORTED_MODEL_TYPE["llama"]
            NormalizedConfigManager._conf["stablelm-epoch"] = NormalizedTextConfig.with_args(
                num_layers="num_hidden_layers",
                num_attention_heads="num_attention_heads"
            )
            config = AutoConfig.from_pretrained(kwargs.get("model_name_or_path"), trust_remote_code=True)
            OVModelForCausalLM.from_pretrained = partial(
                OVModelForCausalLM.from_pretrained, 
                config=config, 
                trust_remote_code=True, 
                use_cache=True
            )
        
        # Set model trust_remote_code and initialize
        kwargs['trust_remote_code'] = kwargs.get('trust_remote_code', True)
        super().__init__(*args, **kwargs)

    @property
    def add_special_tokens(self) -> bool:
        return False
