import numpy as np
import torch
import os
from functools import partial
import gin
from absl import flags, app
from GaMaDHaNi.src.dataset import Task
from GaMaDHaNi.utils.generate_utils import load_pitch_model, load_audio_model
import GaMaDHaNi.utils.pitch_to_audio_utils as p2a
from GaMaDHaNi.utils.utils import get_device, plot, save_figure, save_csv, save_audio
from absl import app
import torch.nn.functional as F
import logging


FLAGS = flags.FLAGS

flags.DEFINE_string('pitch_run', default=None, required=True, help='Path to the pitch generation model folder')
flags.DEFINE_string('audio_run', default=None, required=True, help='Path to the pitch to audio generation model folder')
flags.DEFINE_string('db_path_audio', default=None, required=True, help='Path to the audio DB')
flags.DEFINE_bool('prime', default=False, help='Boolean value indicating whether to perform Primed Generation')
flags.DEFINE_string('pitch_model_type', default=None, help='Whether to use ')
flags.DEFINE_string('prime_file', default=None, help='numpy file containing the primes')
flags.DEFINE_string('dataset_split_file', default=None, help='JSON file containing the dataset split of files used in train and validation sets respectively')
flags.DEFINE_integer('number_of_samples', default=16, help='number of samples to generate')
flags.DEFINE_string('outfolder', default=os.getcwd(),help='path where the generated pitch contour plots, csv files and audio files are to be saved' )
flags.DEFINE_integer('seq_len', default=1200, help='Used when pitch_model_type==transformer, total length of the sequence to be generated, when running primed generation, seq_len includes the prime too')
flags.DEFINE_float('temperature', default=1.0, help='Used when pitch_model_type==transformer, controls randomness in sampling; lower values (< 1.0) produce more deterministic results, while higher values (> 1.0) increase diversity. ')
flags.DEFINE_integer('num_steps', default=100, help='Used when pitch_model_type==diffusion, the number of diffusion steps for the model; more steps can improve quality but increase computation time.')
flags.DEFINE_list('singers', default=[3], help='Used by the pitch to audio model, singer IDs for the singer conditioning')
flags.DEFINE_string('pitch_config_path', default=None, help='config file path for the pitch generation model')
flags.DEFINE_string('audio_config_path', default=None, help='config file path for the pitch to audio generation model')

def load_pitch_fns(pitch_path: str, model_type: str, prime: bool = False, prime_file: Optional[str] = None, number_of_samples: int = 16, config_path: Optional[str] = None) -> Tuple[Any, Optional[Any], Callable, Callable, Optional[torch.Tensor]]:
    config_path = os.path.join(pitch_path, 'config.gin') if not config_path else config_path

    if prime:
        assert prime_file, \
            "Error: If 'prime' is True, either 'prime_file' must be provided."
    gin.parse_config_file(config_path)  
    if model_type=="diffusion":
        qt_path = os.path.join(pitch_path, 'qt.joblib')
    else:
        qt_path = None

    seq_len = gin.query_parameter("%SEQ_LEN")
    time_downsample = gin.query_parameter('src.dataset.Task.kwargs').get("time_downsample")
    seq_len_cache = int((1/3) * seq_len)
    
    pitch_model, pitch_qt, primes = load_pitch_model(
        config = config_path, 
        ckpt = os.path.join(pitch_path, 'models', 'last.ckpt'), 
        qt = qt_path,
        prime_file = prime_file,
        model_type = model_type,
        number_of_samples=number_of_samples
        )
    
    if not prime:
        #unprimed generation
        if model_type=="transformer":
            #get start token (sampling a start token from a popular range in validation set) -> primes
            primes = np.random.randint(low=154,high=222, size=(number_of_samples,1)).astype(np.float64)
        else:
            primes = None

    task_obj = Task()
    pitch_task_fn = partial(task_obj.read_) # , qt_transform = pitch_qt, add_noise_to_silence = True)
    invert_pitch_task_fn = partial(task_obj.invert_) #, qt_transform = pitch_qt)
    processed_primes = None
    if prime:
        if model_type=="diffusion":
            pitch_task_fn = partial(pitch_task_fn, qt_transform = pitch_qt, add_noise_to_silence = True)
            invert_pitch_task_fn = partial(invert_pitch_task_fn, qt_transform = pitch_qt)
        
        processed_primes = [torch.tensor(pitch_task_fn(
                    **{"inputs": {"pitch": {"data": torch.tensor(p[:seq_len_cache*time_downsample:])}}})["sampled_sequence"]) for p in primes]    
        primes = torch.stack(processed_primes)

    else:
        if model_type=="transformer":
            processed_primes = [torch.tensor(pitch_task_fn(
                **{"inputs": {"pitch": {"data": torch.tensor(p)}}})["sampled_sequence"]) for p in primes]
            primes = torch.stack(processed_primes)
        else:
            invert_pitch_task_fn = partial(invert_pitch_task_fn, qt_transform = pitch_qt)
            primes = None
    return pitch_model, pitch_qt, pitch_task_fn, invert_pitch_task_fn, primes

def load_audio_fns(audio_path, db_path_audio, config_path=None):
    ckpt = os.path.join(audio_path, 'models', 'last.ckpt')
    config = os.path.join(audio_path, 'config.gin') if not config_path else config_path
    qt = os.path.join(db_path_audio, 'qt.joblib')

    audio_model, audio_qt = load_audio_model(config, ckpt, qt)
    audio_seq_len = gin.query_parameter('%AUDIO_SEQ_LEN')

    invert_audio_fn = partial(
        p2a.normalized_mels_to_audio,
        qt=audio_qt,
        n_iter=200
    )

    return audio_model, audio_qt, audio_seq_len, invert_audio_fn


def generate_pitch(pitch_model,
                   pitch_model_type,
                   invert_pitch_fn, 
                   num_samples, 
                   seq_len=1200,
                   temperature=1.0,
                   num_steps=None, 
                   outfolder=None, 
                   processed_primes=None,
                   pitch_sample_rate=200,
                   prime=False):
    if pitch_model_type=="diffusion":
        samples = pitch_model.sample_fn(batch_size=num_samples, num_steps=num_steps, prime=processed_primes) # keep only 4s
    else:
        samples = pitch_model.sample_fn(batch_size=num_samples, seq_len=seq_len, prime=processed_primes)

    inverted_pitches = [invert_pitch_fn(**{"f0": sample}) for sample in samples.detach().cpu().numpy()]

    if outfolder is not None:
        for i, pitch in enumerate(inverted_pitches):
            fig = plot(f0_array=pitch, time_array=np.arange(0, len(pitch) / pitch_sample_rate, 1/pitch_sample_rate), prime=prime)
            save_figure(fig, dest_path=f"{outfolder}/pitch/{i}.png")
    return samples, inverted_pitches

def generate_audio(audio_model, f0s, invert_audio_fn, outfolder, singers=[3], num_steps=100):
    singer_tensor = torch.tensor(np.repeat(singers, repeats=f0s.shape[0])).to(audio_model.device)
    samples, _, singers = audio_model.sample_cfg(f0s.shape[0], f0=f0s, num_steps=num_steps, singer=singer_tensor, strength=3)
    audio = invert_audio_fn(samples)
    
    if outfolder is not None:
        os.makedirs(outfolder, exist_ok=True)
        for i, a in enumerate(audio):
            logging.log(logging.INFO, f"Saving audio {i}")
            save_audio(audio_array=a.clone().detach().unsqueeze(0).cpu(), dest_path=f"{outfolder}/audio/{i}.wav", sample_rate=16000)
            # save_csv()
    return audio

def generate(audio_model=None, 
            pitch_model=None, 
            pitch_model_type=None,
            invert_pitch_fn=None,
            invert_audio_fn=None,
            num_samples=2, 
            singers=[3],
            num_steps=100, 
            seq_len=1200,
            temperature=1., 
            outfolder='temp', 
            audio_seq_len=750, 
            pitch_qt=None, 
            prime=False,
            processed_primes=None, 
            device=None):
    
    logging.log(logging.INFO, 'Generate function')

    pitch_model = pitch_model.to(device) 
    processed_primes = processed_primes.to(device) if prime else None

    pitch, inverted_pitch = generate_pitch(pitch_model=pitch_model,
                                           pitch_model_type=pitch_model_type,
                                           invert_pitch_fn=invert_pitch_fn, 
                                           num_samples=num_samples, 
                                           num_steps=num_steps, 
                                           outfolder=outfolder, 
                                           seq_len=seq_len,
                                           temperature=temperature,
                                           processed_primes=processed_primes,
                                           pitch_sample_rate=100,
                                           prime=prime)
    
    pitch = pitch.unsqueeze(1) if pitch.dim()==2 else pitch
    interpolated_pitch = p2a.interpolate_pitch(pitch=pitch, audio_seq_len=audio_seq_len)
    interpolated_pitch = torch.nan_to_num(interpolated_pitch, nan=196)
    interpolated_pitch = interpolated_pitch.squeeze(1).float() # to match input size by removing the extra dimension
    audio_model = audio_model.to(device)
    audio = generate_audio(audio_model, interpolated_pitch, invert_audio_fn, singers=singers, num_steps=100, outfolder=outfolder)[:, :16000*4]
    return pitch, audio
    
def main(argv):
    pitch_path = FLAGS.pitch_run
    audio_path = FLAGS.audio_run
    db_path_audio = FLAGS.db_path_audio
    prime = FLAGS.prime
    pitch_model_type = FLAGS.pitch_model_type
    prime_file = FLAGS.prime_file
    number_of_samples = FLAGS.number_of_samples
    outfolder = FLAGS.outfolder
    seq_len = FLAGS.seq_len
    temperature = FLAGS.temperature
    num_steps = FLAGS.num_steps
    singers = FLAGS.singers
    device = get_device() 
    pitch_config_path = FLAGS.pitch_config_path  
    audio_config_path = FLAGS.audio_config_path 

    if seq_len:
        seq_len_cache = int((1/3) * seq_len)
        seq_len_gen = int((2/3) * seq_len)
        
    #1. loading pitch and audio model and supporting functions
    pitch_model, pitch_qt, pitch_task_fn, invert_pitch_fn, primes = load_pitch_fns(pitch_path=pitch_path, 
                                                                                   model_type = pitch_model_type, 
                                                                                   prime=prime, 
                                                                                   prime_file=prime_file, 
                                                                                   number_of_samples=number_of_samples,
                                                                                   config_path=pitch_config_path)  

    audio_model, audio_qt, audio_seq_len, invert_audio_fn = load_audio_fns(audio_path=audio_path, 
                                                                           db_path_audio=db_path_audio,
                                                                           config_path=audio_config_path)
  
    # 3. generate (I) pitch and (II) convert pitch to audio (generate audio conditioned on pitch)
    pitch, audio = generate(audio_model=audio_model,
                            pitch_model=pitch_model,
                            pitch_model_type=pitch_model_type,
                            invert_pitch_fn=invert_pitch_fn, 
                            invert_audio_fn=invert_audio_fn,
                            num_samples=number_of_samples, 
                            singers=singers,
                            num_steps=num_steps, 
                            outfolder=outfolder, 
                            seq_len=seq_len_gen,
                            temperature=temperature,
                            prime=prime,
                            processed_primes=primes,
                            device=device)



if __name__ == '__main__':
    app.run(main)       