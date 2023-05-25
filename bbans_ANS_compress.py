import os
import time
import torch
import numpy as np
import utils.bbans_util as util
import utils.bbans_rans as rans

from config import Config_bbans
from torch.utils.data import DataLoader
from utils.common import same_seed, load_model, load_data


cf = Config_bbans()
cf_compress = cf.compress_hparam
cf_model = cf.model_hparam

same_seed(cf.seed)
torch.backends.cudnn.benchmark = True

rng = np.random.RandomState(0)
np.seterr(over='raise')

prior_precision = 8
obs_precision = 14
q_precision = 14

print(f"Model:{cf.model_name}; Dataset:{cf.dataset}")

## Load mnist images
_, test_set = load_data(cf.dataset, cf.model_name, load_train=False)
cf.model_hparam.xdim = test_set[0][0].shape

batch_size = cf_compress.batch_size
num_images = 1000#len(test_set) - len(test_set) % batch_size
n_batches = num_images // batch_size

test_loader = DataLoader(
    dataset=test_set, sampler=None, 
    batch_size=batch_size, shuffle=False)
images = []
for i, x in enumerate(test_loader):
    if i == n_batches:
        break
    images.append(torch.flatten(x[0], start_dim=-3).numpy().astype(np.uint8))
print(f'num data: {len(images)} x {images[0].shape}')


model, _, _ = load_model(cf.model_name, cf.model_pt, 
                         cf.model_hparam, cf.lr, cf.decay)
model.eval()

obs_append = util.beta_binomial_obs_append(255, obs_precision)
obs_pop = util.beta_binomial_obs_pop(255, obs_precision)

latent_dim = model.zdim[0]
latent_shape = (batch_size, latent_dim)

vae_append = util.vae_append(latent_shape, model, obs_append,
                             prior_precision, q_precision)
vae_pop = util.vae_pop(latent_shape, model, obs_pop,
                       prior_precision, q_precision)


# randomly generate some 'other' bits
other_bits = rng.randint(low=1 << 16, high=1 << 31, size=100, dtype=np.uint32)
state = rans.unflatten(other_bits)

print_interval = 50
encode_start_time = time.time()
for i, image in enumerate(images):
    state = vae_append(state, image)

    if not i % print_interval:
        print('Encoded {}'.format(i))

encode_t = time.time() - encode_start_time
print('\nAll encoded in {:.2f}s'.format(encode_t))
print("Average singe image encoding time: {:.2f}s.".format(encode_t / num_images))


compressed_message = rans.flatten(state)

count = 0
for i in range(1, len(other_bits)):
    if other_bits[-i] == compressed_message[-i]:
        count += 1
    else:
        break

init_cost = 32 * (len(other_bits) - count)
print(f'Initial cost: {init_cost} bits.')

single_image_init_cost = model.init_cost_record / num_images
print('Average initial cost/image: {:.4f}'.format(single_image_init_cost))
print('Average initial cost/z dim: {:.4f}'.format(single_image_init_cost / (np.prod(model.zdim) * model.nz)))
print('Average initial cost/x dim: {:.4f}'.format(single_image_init_cost / np.prod(model.xdim)))

extra_bits = 32 * (len(compressed_message) - len(other_bits))
print('Exclude initial cost, Used {} bits.'.format(extra_bits))
print('Net bit rate: {:.4f} bits per dim.'.format(extra_bits / (num_images * np.prod(cf.model_hparam.xdim))))

total_bits = extra_bits + init_cost
print("Total used {} bits.".format(total_bits))
print("Bit rate: {:.4f} bits per dim.".format(total_bits / (num_images * np.prod(cf.model_hparam.xdim))))


state = rans.unflatten(compressed_message)
decode_start_time = time.time()

for n in range(len(images)):
    state, image_ = vae_pop(state)
    original_image = images[len(images)-n-1]
    # np.testing.assert_allclose(original_image, image_)

    if not n % print_interval:
        print('Decoded {}'.format(n))

decode_t = time.time() - encode_start_time
print('\nAll decoded in {:.2f}s'.format(decode_t))
print("Average singe image decoding time: {:.2f}s.".format(decode_t / num_images))

recovered_bits = rans.flatten(state)
assert all(other_bits == recovered_bits)
