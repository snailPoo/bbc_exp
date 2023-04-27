import time
import torch
import numpy as np
import craystack as cs
from autograd.builtins import tuple as ag_tuple
from torch.utils.data import DataLoader

from config import Config_bbans
from utils.common import same_seed, load_model, load_data
from utils.distributions import generate_beta_binomial_probs
from codec.bbc_scheme import VAE


cf = Config_bbans()
cf_compress = cf.compress_hparam
cf_model = cf.model_hparam

same_seed(cf.seed)

rng = np.random.RandomState(0)

prior_precision = cf_compress.prior_precision
obs_precision = cf_compress.obs_precision
q_precision = cf_compress.q_precision
n = torch.tensor(255)

print(f"Model:{cf.model_name}; Dataset:{cf.dataset}")

## Load mnist images
_, test_set = load_data(cf.dataset, cf.model_name)
cf.model_hparam.xdim = test_set[0][0].shape

batch_size = cf_compress.batch_size
num_images = len(test_set) - len(test_set) % batch_size
n_batches = num_images // batch_size

test_loader = DataLoader(
    dataset=test_set, sampler=None, 
    batch_size=batch_size, shuffle=False)
images = []
for i, x in enumerate(test_loader):
    if i == n_batches:
        break
    images.append(torch.flatten(x[0], start_dim=-3).numpy().astype(np.uint64))
print(f'num data: {len(images)} x {images[0].shape}')


## Setup codecs
# VAE codec
model, _, _ = load_model(cf.model_name, cf.model_pt, 
                         cf.model_hparam, cf.lr, cf.decay)
model.to(cf.device)
model.eval()
rec_net = model.encode # torch_fun_to_numpy_fun(model.encode)
gen_net = model.decode # torch_fun_to_numpy_fun(model.decode)

obs_codec = lambda x: cs.Categorical(generate_beta_binomial_probs(*x, n), obs_precision)

x_flat = int(np.prod(cf.model_hparam.xdim))
latent_shape = (batch_size, cf.model_hparam.z_size)
latent_size = np.prod(latent_shape)
obs_shape = (batch_size, x_flat)
obs_size = np.prod(obs_shape)

def vae_view(head):
    return ag_tuple((np.reshape(head[:latent_size], latent_shape),
                     np.reshape(head[latent_size:], obs_shape)))

vae_append, vae_pop = cs.repeat(cs.substack(
    VAE(gen_net, rec_net, obs_codec, prior_precision, q_precision, cf.device),
    vae_view), n_batches)

## Encode
# Initialize message with some 'extra' bits
encode_t0 = time.time()
init_message = cs.base_message(obs_size + latent_size)

# Encode the mnist images
message, = vae_append(init_message, images)

flat_message = cs.flatten(message)
encode_t = time.time() - encode_t0

print("All encoded in {:.2f}s.".format(encode_t))
print("Average singe image encoding time: {:.2f}s.".format(encode_t / num_images))

message_len = 32 * len(flat_message)
print("Used {} bits.".format(message_len))
print("This is {:.4f} bits per pixel.".format(message_len / (num_images * x_flat)))

flat_init_message = cs.flatten(init_message)
init_message_len = 32 * len(flat_init_message)
print("Exclude initial message length, Used {} bits.".format(message_len-init_message_len))
print("Net bit rate: {:.4f} bits per pixel.".format((message_len-init_message_len) / (num_images * x_flat)))

## Decode
decode_t0 = time.time()
message = cs.unflatten(flat_message, obs_size + latent_size)

message, images_ = vae_pop(message)
decode_t = time.time() - decode_t0

print('All decoded in {:.2f}s.'.format(decode_t))
print("Average singe image decoding time: {:.2f}s.".format(decode_t / num_images))

np.testing.assert_equal(images, images_)
