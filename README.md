# NIGHTSHADE

This repo contains the research code release for Nightshade.

:warning: If you plan to use Nightshade to protect your own copyrighted content, please use our MacOS/Windows prototype on our [webpage](https://nightshade.cs.uchicago.edu/downloads.html). For details on the differences, please checkout the FAQ below. 

### OVERVIEW

The repo contains code to generate Nightshade data for a source concept (e.g. "dog"). Specifically,

1) The code first identify an optimal set of clean image/text pairs from a pool of clean data from the source concept. This step is designed to find a good set of text prompts to use for the poison. We select data from LAION dataset in our experiments (details in Section 5.2 of the paper).
2) We optimize a perturbation on each of the selected candidate images.

### HOW TO

#### Step 1: Candidate Data Selection

We first extract a desired set of clean image/text pairs as the starting point of poison generation. Given a source concept (e.g. "dog"), you need to collect a large set (> 500) image/text pairs that contain the source concept (e.g. dog images with their corresponding text prompts). In the paper, we used images from LAION and ConceptualCaption dataset. If you do not have text prompts, you can use BLIP or similar techniques to generate the prompts or simply use "a photo of X" for the prompts.

**Data format:** To better store longer prompts, we use pickle files for image/text pairs. Each pickle file contains a numpy image (key "img") and its corresponding text prompt (key "text"). You can download some example data from [here](https://mirror.cs.uchicago.edu/fawkes/files/resources/example-data.zip).

Next, run `data_extraction.py` to select a set of 100 poison candidates. `python3 data_extraction.py --directory data/ --concept dog --num 100`

#### Step 2: Poison Generation

Next, we add perturbation to the images given a target concept (e.g. "cat"). `python3 data_extraction.py --directory selected_data/ --target_name cat --outdir OUTPUTDIR`. The code will output the perturbed images to the output folder.

#### Requirements
`torch=>2.0.0`, `diffusers>=0.20.0`

### FAQ

#### How does this code differ from the APP on the Nightshade website?
The goal of the code release is different from the Nightshade APP. This code base seeks to provide a reference, basic implementation for research experimentation whereas the APP is designed for people to nightshade their own images. As a result, there are two main differences. First, the Nightshade APP automatically extracts the source concept from a given image and selects a target concept for the image. This code base gives researchers the flexibility to select different poison target. Second, this code base uses Linf perturbation compared to LPIPS perturbation in the APP. Linf perturbation leads to more stable results but more visible perturbations.

#### How do I test Nightshade?
The easiest way to train the model is to use latent diffusion [source code](https://github.com/CompVis/stable-diffusion). We do not recommend using the Dreambooth/LORA finetuning code as it is designed for small-scale finetuning rather than full model training.

### Citation

```
@inproceedings{shan2024nightshade,
  title={Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models},
  author={Shan, Shawn and Ding, Wenxin and Passananti, Josephine and Wu, Stanley and Zheng, Haitao and Zhao, Ben Y.},
  booktitle={IEEE Symposium on Security and Privacy},
  year={2024},
}
```

For any questions with the code, please email shawnshan@cs.uchicago.edu. 
