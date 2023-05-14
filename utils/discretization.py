from utils.torch.rand import *
from tqdm import tqdm
import os
from torch.utils.data import *
from sklearn.preprocessing import KBinsDiscretizer

# returns discretization bin endpoints and centres
# This step is only done once after the model training is completed
def posterior_sampling(cf, model, train_loader):
    
    nz = model.nz
    zdim = model.zdim
    quantbits = cf.z_quantbits
    num_sample_per_bin = 30
    batch_size = cf.discretization_batch_size
    num_bin = (1 << quantbits)
    # total number of samples (num_sample_per_bin * number of bins)
    total_samples = num_sample_per_bin * num_bin
    total_zdim = np.prod(zdim)

    with torch.no_grad():
        # check if there does already exists a file with the discretization bins saved into it
        if not os.path.exists(os.path.join(cf.discretization_dir, f'{cf.model_name}.pt')):
            # set up an empty tensor for all the bins (number of latent variables, total dimension of latent, number of bins)
            # note that we do not include the first and last endpoints, because those will always be -inf and inf
            z_bin_ends = np.zeros((nz, total_zdim, num_bin - 1))
            z_bin_centres   = np.zeros((nz, total_zdim, num_bin))

            # --------------------------- original ---------------------------
            # top latent is fixed, so we can calculate the discretization bins without samples
            if cf.model_name == 'bitswap':
                zbins = Bins(mu = torch.zeros((1, 1, total_zdim)), 
                            scale = torch.ones((1, 1, total_zdim)), 
                            precision = quantbits)
                z_bin_ends[nz - 1] = zbins.endpoints().numpy()
                z_bin_centres[nz - 1] = zbins.centres().numpy()
            # ----------------------------------------------------------------

            num_batch = total_samples // batch_size # number of num_batch

            # set-up a batch-loader for the datasetc
            datapoints = list(train_loader)

            # concatenate the dataset with itself if the length is not sufficient
            while len(datapoints) < total_samples:
                datapoints += datapoints

            # use 16-bit values to reduce memory usage
            gen_samples = np.zeros((nz, total_samples) + zdim, dtype=np.float16)
            inf_samples = np.zeros((nz, total_samples) + zdim, dtype=np.float16)

            # --------------------------- original ---------------------------
            if cf.model_name == 'bitswap':
                gen_samples[-1] = sample_from_logistic(0, 1, (total_samples,) + zdim, device="cpu", bound=1e-30).numpy()
            # ------------------------ autoregressive ------------------------
            elif cf.model_name == 'shvc':
                iterator = tqdm(range(num_batch), desc=f"sampling z{nz} from generator")
                for bi in iterator:
                    gen_from = bi * batch_size
                    to = gen_from + batch_size
                    hidden = None
                    lstm_input = torch.zeros((batch_size, 1, 1, model.zdim[-2], model.zdim[-1])).to(cf.device) # z3: (B, c=1, 1, h, w)
                    for i in range(0, model.zdim[0]):
                        param, hidden = model.p_z.ar_model(lstm_input, hidden=hidden) # z3: (B, c=1, 2, h, w)
                        mu, logsd = torch.split(param.squeeze(1), 1, dim=1) # (B, 1, h, w)
                        mu = mu.squeeze(1)
                        scale = torch.exp(logsd.squeeze(1))
                        lstm_input = sample_from_logistic(mu, scale, mu.shape, device=cf.device, bound=1e-30)
                        gen_samples[-1][gen_from: to, i,] = lstm_input.cpu().numpy()
                        lstm_input = lstm_input.unsqueeze(1).unsqueeze(1)
            # ----------------------------------------------------------------

            # iterate over the latent variables
            # gen z7 given z8 ~ gen z1 given z2
            for zi in reversed(range(1, nz)):
                # obtain samples from the generative model
                iterator = tqdm(range(num_batch), desc=f"sampling z{zi} from generator")
                for bi in iterator:
                    gen_from = bi * batch_size
                    to = gen_from + batch_size

                    mu, scale = model.generate(zi)(given=torch.from_numpy(gen_samples[zi][gen_from: to]).to(cf.device).float())
                    gen_samples[zi - 1][gen_from: to] = sample_from_logistic(mu, scale, mu.shape, device=cf.device, bound=1e-30).cpu()

            # inf z1 given x ~ inf z7 given z6
            rng = nz - 1 if cf.model_name == 'bitswap' else nz
            for zi in range(0, rng):
                # obtain samples from the inference model (using the dataset)
                iterator = tqdm(range(num_batch), desc=f"sampling z{zi + 1} from inference model")
                for bi in iterator:
                    gen_from = bi * batch_size
                    to = gen_from + batch_size

                    if zi == 0: # z1
                        given = datapoints[bi][0]
                        # ------------ autoregressive ------------
                        if cf.model_name == 'shvc':
                            given = modules.lossless_downsample(given)
                            given[:, int(model.s)+1:,] = 0
                        # ----------------------------------------
                    else:
                        given = torch.from_numpy(inf_samples[zi - 1][gen_from: to])
                    
                    mu, scale = model.infer(zi)(given=given.to(cf.device).float())
                    inf_samples[zi][gen_from: to] = sample_from_logistic(mu, scale, mu.shape, device=cf.device, bound=1e-30).cpu().numpy()

            # get the discretization bins
            for zi in range(rng):
                samples = np.concatenate([gen_samples[zi], inf_samples[zi]], axis=0).reshape(-1, total_zdim)
                z_bin_ends[zi], z_bin_centres[zi] = discretize_kbins(samples, quantbits, strategy='uniform')

            # move the discretization bins to the GPU
            z_bin_ends = torch.from_numpy(z_bin_ends)
            z_bin_centres = torch.from_numpy(z_bin_centres)

            # save the bins for reproducibility and speed purposes
            if not os.path.exists(cf.discretization_dir):
                os.makedirs(cf.discretization_dir)
            result = {
                "endpoints" : z_bin_ends, 
                "centres" : z_bin_centres
            }
            torch.save(result, os.path.join(cf.discretization_dir, f'{cf.model_name}.pt'))
            
        else:
            result = torch.load(os.path.join(cf.discretization_dir, f'{cf.model_name}.pt'))
            z_bin_ends = result['endpoints']
            z_bin_centres = result['centres']

    # cast the bins to the appropriate type (in our experiments always float64)
    return z_bin_ends.type(cf.type).to(cf.device), z_bin_centres.type(cf.type).to(cf.device)

# function that exploits the KBinsDiscretizer from scikit-learn
# two strategy are available
# 1. uniform: bins with equal width
# 2. quantile: bins with equal frequency
def discretize_kbins(samples, quantbits, strategy):

    # apply discretization
    est = KBinsDiscretizer(n_bins=1 << quantbits, strategy=strategy)
    est.fit(samples)

    # obtain the discretization bins endpoints
    endpoints = np.array([np.array(ar) for ar in est.bin_edges_]).transpose()
    centres = (endpoints[:-1,:] + endpoints[1:,:]) / 2
    endpoints = endpoints[1:-1]

    return endpoints.transpose(), centres.transpose()