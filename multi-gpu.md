# Multi-GPU  one machine with multi gpu cards
```Python
import os
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 3, 0'  ### set the device id of using
local_rank = [0, 1, 2] ### range from 0 to the number of using gpu cards
device = torch.device(f'cuda:{torch.cuda.current_device()}')
torch.distributed.init_process_group(init_method='tcp://localhost:36699', rank=0, world_size=1, backend='nccl')  # set multi-gpu environment

### tips for writing model code ###
### It is recommanded to put your core code into the Model class, e.g.,
'''
Model(nn.Module):
    def __init__(self,...):
        ...
    def forward(self, input):
        Code that requires gpu computation should be placed here
        the Input should be tensor type data, I'm not sure if the other types of data are supported by multiple GPUs
        batch size here typically is Batch / number of gpus
'''

...
model = Model()  
model.to(device)  ## put model to the main gpu device
model = nn.parallel.DistributedDataParallel(
 model,
 find_unused_parameters=True,
 device_ids=local_rank, output_device=local_rank[0],
 broadcast_buffers=False
 )

batch = tuple(t.to(device) for t in batch)  ## put data to gpu 
input_id, input_mask, lable, ... = batch
### input_id size is [Batch, Seq_len]
loss = model(input_id, input_mask, lable, ...) 
'''
loss size is [Batch], i.e., pytorch would automatically assign Batch size data into multi gpu cards, 
then gether the processd data to one gpu card and return it
'''
loss = loss.mean() ### mean() to average on multi-gpu
loss.backward()
...
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.bin"))
