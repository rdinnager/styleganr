import torch
import pickle
import json
import zipfile

model_name = 'stylegan3-r-afhqv2-512x512'
model_pickle = model_name + '.pkl'
out_file = model_name + '-R.pt'
out_json = model_name + '-R.json'
out_zip = model_name + '-R.zip'

with open(model_pickle, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    
state_dict = dict(G.state_dict())
init_kwargs = dict(G.init_kwargs)

torch.save(state_dict, out_file, _use_new_zipfile_serialization=True)
with open(out_json, 'w') as fp:
    json.dump(init_kwargs, fp)

zip_file = zipfile.ZipFile(out_zip, "w")
zip_file.write(out_file)
zip_file.write(out_json)
zip_file.close()
