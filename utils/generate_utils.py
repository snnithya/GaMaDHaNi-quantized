# 1. invert the task function(s) - maintain the same name with just "inert_" at the beginning

# 2. add other util functions: plot pitch

# 3. remove outliers

# 4. loading the data
# use the load pitch function? or the load data is different from the transformers repo

# 5. loading pitch model diffusion or transformer , model as an argument
def load_pitch_model(config, ckpt, qt = None, prime_file=None, model_type = None):
    gin.parse_config_file(config)
    assert model_type is not None, 'model_type argument is not passed for the pitch generator model, choose either diffusion or transformer'
    if model_type=="diffusion":
        model = UNet()
    elif model_type=="transformer":
        model = XTransformerPrior()
        
    model.load_state_dict(torch.load(ckpt)['state_dict'])  
    model.to('cuda')
    if qt is not None:
        qt = joblib.load(qt)
    if prime_file is not None:
        with gin.config_scope('val'): # probably have to change this
            with gin.unlock_config():
                gin.bind_parameter('dataset.pitch_read_w_downsample.qt_transform', qt)
        primes = np.load(prime_file, allow_pickle=True)['concatenated_array'][:, 0]
    else:
        primes = None
        task_fn = None
    task_fn = partial(pitch_read_w_downsample,
    seq_len=None)
    return model, qt, primes, task_fn

# 6. loading the audio model
def load_audio_model(config, ckpt, qt = None):
    gin.parse_config_file(config)
    model = UNetPitchConditioned() # there are no gin parameters for some reason
    model.load_state_dict(torch.load(ckpt)['state_dict'])  
    model.to('cuda')
    if qt is not None:
        qt = joblib.load(qt)

    return model, qt
