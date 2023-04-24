# export PYTHONPATH=$PYTHONPATH:../
from autograd.builtins import tuple as ag_tuple
from torch.utils.data import DataLoader

import craystack as cs
import numpy as np
import time
import torch
import os
import pickle

from config import Config_hilloc
from codec.bbc_scheme import ResNetVAE, custom_ResNetVAE
from utils.common import load_data, load_model, same_seed


cf = Config_hilloc()
cf_compress = cf.compress_hparam
cf_model = cf.model_hparam

same_seed(cf.seed)

print(f"Model:{cf.model_name}; Dataset:{cf.dataset}")

# ******* data ********
_, test_set = load_data(cf.dataset, cf.model_name, load_train=False)
cf_model.xdim = test_set[0][0].shape
# export PYTHONPATH=$PYTHONPATH:../
num_images = 1000#len(test_set)
batch_size = cf_compress.batch_size
n_batches = num_images // batch_size

# ******* model *******
model, _, _ = load_model(cf.model_name, cf.model_pt, 
                         cf_model, cf.lr, cf.decay)
model.eval().to(cf.device)
# *********************
# **********************
test_loader = DataLoader(
    dataset=test_set, sampler=None, 
    batch_size=batch_size, shuffle=True)

images = []
for i, x in enumerate(test_loader):
    if i == n_batches:
        break
    # print(i)
    # c = x[0]
    # elbo, _ = model.loss(c.to(cf.device), 'test')
    # c = x[0].numpy().astype(np.uint64)
    # c = torch.from_numpy(c.astype(np.float32))
    # elbo, _ = model.loss(c.to(cf.device), 'test')
    images.append(x[0].numpy().astype(np.uint64))
# **********************
# images = [np.transpose(np.array([image]).astype('uint64'), (0, 3, 1, 2))
#           for image in test_set.data]
# images = images[:num_images]
# **********************

num_dims = num_images * np.prod(cf_model.xdim)# np.sum([img.size for img in images])
print(f'num data: {len(images)} x {images[0].shape}')
# *********************



# ******* codec *******
scheme = ResNetVAE(cf_compress, model)#custom_ResNetVAE

codec_shape = (batch_size, *cf_model.xdim)
print(f"Creating codec for shape {codec_shape}")
latent_shape = (codec_shape[0], cf_model.z_size, 
                codec_shape[2] // 2, codec_shape[3] // 2)
latent_dims = np.prod(latent_shape)

def vae_view(head):
    return ag_tuple((np.reshape(head[:latent_dims], latent_shape),
                     np.reshape(head[latent_dims:],  codec_shape)))

codec = lambda: cs.repeat(cs.substack(scheme, vae_view), len(images))
vae_push, vae_pop = codec()
# *********************

# ******* state *******
init_t0 = time.time()
state_path = f"bitstreams/initial_bit_{cf_compress.initial_bits}.pkl"
if os.path.exists(state_path):
    print('load init state')
    with open(state_path, 'rb') as f:
        state = pickle.load(f)
else:
    print('create init state')
    state = cs.random_message(cf_compress.initial_bits, (1,))
    with open(state_path, 'wb') as f:
        pickle.dump(state, f)
init_head_shape = (np.prod(codec_shape) + np.prod(latent_shape),)
state = cs.reshape_head(state, init_head_shape)
init_t = time.time() - init_t0
print("Initialization time: {:.2f}s".format(init_t))
# *********************

# **** compression ****
print("Start Encoding.")
encode_t0 = time.time()
state, = vae_push(state, images)
encode_t = time.time() - encode_t0
print("All encoded in {:.2f}s".format(encode_t))
print("Average singe image encoding time: {:.2f}s.".format(encode_t / num_images))

flat_t0 = time.time()
flat_state = cs.flatten(state)
print("flatten state cost {:.2f}s.".format(time.time() - flat_t0))

state_len = 32 * len(flat_state)
print("Used {} bits.".format(state_len))
print("This is {:.2f} bits per dim.".format(state_len / num_dims))

extra_bits = state_len - 32 * cf_compress.initial_bits
print('Exclude initial state length, Used {} bits.'.format(extra_bits))
print('Net bit rate: {:.2f} bits per dim.'.format(extra_bits / num_dims))
# *********************

# *** decompression ***
print("Start Decoding.")
decode_t0 = time.time()
state = cs.unflatten(flat_state, init_head_shape)
state, decoded_images = vae_pop(state)
state = cs.reshape_head(state, (1,))

decode_t = time.time() - decode_t0
print('All decoded in {:.2f}s'.format(decode_t))
print("Average singe image decoding time: {:.2f}s.".format(decode_t / num_images))

# check if decompressed_data == original
# print('decoded_images')
# print(decoded_images)
# print('images')
# print(images)
assert len(images) == len(decoded_images), (len(images), len(decoded_images))
for test_image, decoded_image in zip(images, decoded_images):
    np.testing.assert_equal(test_image, decoded_image)
# *********************