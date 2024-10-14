import numpy as np
from sklearn.preprocessing import QuantileTransformer
from numpy.testing import assert_allclose
import torch

# Assuming GPUQuantileTransformer is your custom class
from gamadhani.utils.pitch_to_audio_utils import GPUQuantileTransformer 

def test_inverse_transform():
    # Generate some random data
    X = np.random.rand(32, 1, 1000) # shape (bs, 1, seq_len)

    # Initialize CPU version of QuantileTransformer
    qt_cpu = QuantileTransformer(n_quantiles=100, random_state=42, output_distribution='normal')

    # Fit cpu transformers on the data
    qt_cpu.fit(X.reshape(-1, 1))    # reshape data to (bs*seq_len, 1)

    # Initialize your custom GPU version of QuantileTransformer
    qt_gpu = GPUQuantileTransformer(qt_cpu, device='cuda')

    # Transform the data 
    X_transformed_cpu = qt_cpu.transform(X.reshape(-1, 1)).reshape(X.shape) # reshape data back to (bs, 1, seq_len)
    X_transformed_gpu = torch.tensor(X_transformed_cpu).to('cuda')

    # Now apply inverse_transform to the transformed data
    X_inv_transformed_cpu = np.array([qt_cpu.inverse_transform(x.reshape(-1, 1)).reshape(1, -1) for x in X_transformed_cpu])
    X_inv_transformed_gpu = qt_gpu.inverse_transform(X_transformed_gpu).detach().cpu().numpy()

    # Assert that the results of inverse_transform are the same (within a tolerance)
    assert_allclose(X_inv_transformed_cpu, X_inv_transformed_gpu, rtol=1e-6, atol=1e-6)
