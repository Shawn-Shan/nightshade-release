# NIGHTSHADE

This repo contains the research code release for Nightshade. 

:warning: If you plan to use Nightshade to protect your own copyrighted content, please uses our MacOS/Windows prototype on our [webpage](https://nightshade.cs.uchicago.edu/downloads.html). 

### OVERVIEW

The repo contains code to generate shaded data using a large candidate pool of clean image/text pairs from a given source concept (e.g. "dog"). Specifically, 
1) We first search and extract a small set of candidate image/text pairs that will be used for poison generation. (details in Section 5.2 of the paper).
2) We optimize a perturbation on each of the selected candidate images.

### HOW TO

#### Step 1: Candidate Data Selection

We first extract a desired set of clean image/text pairs as the starting point of poison generation. Given a source concept (e.g. "dog"), you need to collect a large set (> 500) image/text pairs that contain the source concept (e.g. dog images with their corresponded text prompts). If you do not have text prompt, you can use BLIP or similar technique to generate the prompts or simply use "a photo of X" for the prompts. 

**Data format: ** To better store longer prompts, we use pickle files for image/text pairs. Each pickle file contains an numpy image (key "img") and its corresponded text prompt (key "text"). You can download some example data from [here](https://mirror.cs.uchicago.edu/fawkes/files/resources/example-data.zip). 
