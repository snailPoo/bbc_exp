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
_, test_set = load_data(cf.dataset, cf.model_name)
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
print(f'num data to be processed: {len(images)} x {images[0].shape}')


latent_dim = cf_model.z_size
latent_shape = (batch_size, latent_dim)

model, _, _ = load_model(cf.model_name, cf.model_pt, 
                         cf.model_hparam, cf.lr, cf.decay)
model.eval()

rec_net = util.torch_fun_to_numpy_fun(model.encode)
gen_net = util.torch_fun_to_numpy_fun(model.decode)

obs_append = util.beta_binomial_obs_append(255, obs_precision)
obs_pop = util.beta_binomial_obs_pop(255, obs_precision)

vae_append = util.vae_append(latent_shape, gen_net, rec_net, obs_append,
                             prior_precision, q_precision)
vae_pop = util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
                       prior_precision, q_precision)


# randomly generate some 'other' bits
other_bits = rng.randint(low=1 << 16, high=1 << 31, size=50, dtype=np.uint32)
state = rans.unflatten(other_bits)

compress_lengths = []
print_interval = 10
encode_start_time = time.time()
for i, image in enumerate(images):
    state = vae_append(state, image)

    if not i % print_interval:
        print('Encoded {}'.format(i))

    compressed_length = 32 * (len(rans.flatten(state)) - len(other_bits)) / (i+1)
    compress_lengths.append(compressed_length)

encode_t = time.time() - encode_start_time
print('\nAll encoded in {:.2f}s'.format(encode_t))
print("Average singe image encoding time: {:.2f}s.".format(encode_t / num_images))


compressed_message = rans.flatten(state)

compressed_bits = 32 * len(compressed_message)
print("Used " + str(compressed_bits) + " bits.")
print('This is {:.2f} bits per pixel'.format(compressed_bits / (num_images * np.prod(cf.model_hparam.xdim))))

compressed_bits = 32 * (len(compressed_message) - len(other_bits))
print("Exclude initial message length, Used " + str(compressed_bits) + " bits.")
print('Net bit rate: {:.4f} bits per pixel'.format(compressed_bits / (num_images * np.prod(cf.model_hparam.xdim))))


if not os.path.exists('results'):
    os.mkdir('results')
np.savetxt('compressed_lengths_cts', np.array(compress_lengths))

state = rans.unflatten(compressed_message)
decode_start_time = time.time()

for n in range(len(images)):
    state, image_ = vae_pop(state)
    original_image = images[len(images)-n-1]
    np.testing.assert_allclose(original_image, image_)

    if not n % print_interval:
        print('Decoded {}'.format(n))

decode_t = time.time() - encode_start_time
print('\nAll decoded in {:.2f}s'.format(decode_t))
print("Average singe image decoding time: {:.2f}s.".format(decode_t / num_images))

recovered_bits = rans.flatten(state)
assert all(other_bits == recovered_bits)
