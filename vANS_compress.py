# export PYTHONPATH=$PYTHONPATH:../
from autograd.builtins import tuple as ag_tuple
from torch.utils.data import DataLoader

import craystack as cs
import numpy as np
import time
import os
import pickle

from config import *
from utils.common import load_data, load_model, load_scheme, same_seed


cf = Config_hilloc() # Config_bbans # Config_hilloc # Config_shvc
cf_compress = cf.compress_hparam
cf_model = cf.model_hparam

same_seed(cf.seed)

print(f"Model:{cf.model_name}; Dataset:{cf.dataset}")

# ******* data ********
_, test_set = load_data(cf.dataset, cf.model_name, load_train=False)
cf_model.xdim = test_set[0][0].shape

num_images = len(test_set)
batch_size = cf_compress.batch_size
n_batches = num_images // batch_size

# **********************
test_loader = DataLoader(
    dataset=test_set, sampler=None, 
    batch_size=batch_size, shuffle=True)

images = []
for i, x in enumerate(test_loader):
    if i == n_batches:
        break
    images.append(x[0].numpy().astype(np.uint64))

num_dims = num_images * np.prod(cf_model.xdim)
print(f'num data: {len(images)} x {images[0].shape}')
# *********************

# ******* model *******
model, _, _ = load_model(cf.model_name, cf.model_pt, 
                         cf_model, cf.lr, cf.decay)
model.eval().to(cf.device)
# *********************

# ******* codec *******
scheme = load_scheme(cf.model_name, cf_compress, model)

codec_shape = (batch_size, *model.xdim)
latent_shape = (batch_size, *model.zdim)
print(f"Creating codec for shape x:{codec_shape} z:{latent_shape}")
latent_dims = np.prod(latent_shape)

def vae_view(head):
    return ag_tuple((np.reshape(head[:latent_dims], latent_shape),
                     np.reshape(head[latent_dims:],  codec_shape)))

codec = lambda: cs.repeat(cs.substack(scheme, vae_view), n_batches)
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

init_state = cs.flatten(state)

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

flat_state = cs.flatten(state)
state_len = 32 * len(flat_state)

count = 0
for i in range(1, len(init_state)):
    if init_state[-i] == flat_state[-i]:
        count += 1
    else:
        break
init_cost = 32 * (len(init_state) - count)
print(f'Initial cost: {init_cost} bits.')

single_image_init_cost = model.init_cost_record / num_images
print(f'Average initial cost/image: {single_image_init_cost}')
print(f'Average initial cost/z dim: {single_image_init_cost / (np.prod(model.zdim) * model.n_blocks)}')
print(f'Average initial cost/x dim: {single_image_init_cost / np.prod(model.xdim)}')

extra_bits = state_len - 32 * cf_compress.initial_bits
print('Exclude initial cost, Used {} bits.'.format(extra_bits))
print('Net bit rate: {:.2f} bits per dim.'.format(extra_bits / num_dims))

total_bits = extra_bits + init_cost
print("Total used {} bits.".format(total_bits))
print("Bit rate: {:.2f} bits per dim.".format(total_bits / num_dims))
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
assert len(images) == len(decoded_images), (len(images), len(decoded_images))
for test_image, decoded_image in zip(images, decoded_images):
    np.testing.assert_equal(test_image, decoded_image)
# *********************