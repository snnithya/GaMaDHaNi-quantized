# GaMaDHaNi: Hierarchical Generative Modeling of Melodic Vocal Contours in Hindustani Classical Music
GaMaDHaNi is a modular two-level hierarchy, consisting of a generative model on pitch contours, and a pitch contour to audio synthesis model.\
Using pitch contours as an intermediate representation, we show that our model may be better equipped to listen and respond to musicians in a human-AI collaborative setting by highlighting two potential interaction use cases (1) primed generation (as shown in the diagram below), and (2) coarse pitch conditioning.\

![GaMaDHaNi](hero_diag.jpg)

Read the ISMIR 2024 Paper [here](https://arxiv.org/abs/2408.12658) and 
Listen to Audio Samples [here](https://snnithya.github.io/gamadhani-samples/).
## Installation

   ```bash
   git clone https://github.com/snnithya/GaMaDHaNi.git
   pip install -r requirements.txt
   ```

## How to use

For generating without any melodic prompt(no pitch prime)

```bash
cd GaMaDHaNi
python scripts/generate.py --pitch_model_type=diffusion --prime=False --number_of_samples=1 --download_model_from_hf=True
```

For generating with a melodic prompt(pitch prime)

```bash
cd GaMaDHaNi
python scripts/generate.py --pitch_model_type=diffusion --prime=True --number_of_samples=1 --download_model_from_hf=True 
```
Note: Currently the only pitch_model_type allowed is "diffusion", "transformer" model is soon to be released.

## BibTex
```
@misc{shikarpur2024hierarchicalgenerativemodelingmelodic,
      title={Hierarchical Generative Modeling of Melodic Vocal Contours in Hindustani Classical Music}, 
      author={Nithya Shikarpur and Krishna Maneesha Dendukuri and Yusong Wu and Antoine Caillon and Cheng-Zhi Anna Huang},
      year={2024},
      eprint={2408.12658},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2408.12658}, 
}
```
