# Notes for Preprint

I'm writing a submission to ICML. The premise is that we apply sparse autoencoders to vision models like DINOv2 and CLIP to interpret and control their internal representations. 

## Outline

We're trying to (informally) explain our position with the following metaphor:

Scientific method: observation -> hypothesis -> experiment
Interpretability methods: model behavior -> proposed explanation → ?
SAEs complete the cycle: model behavior -> proposed explanation -> feature intervention

1. Introduction
    1.1. Understanding requires intervention - we must test hypotheses through controlled experiments (scientific method)
    1.2. Current methods provide only understanding or only control, never both
    1.3 Understanding and controlling vision models requires three key capabilities: (1) the ability to identify human-interpretable features (like 'fur' or 'wheels'), (2) reliable ways to manipulate these features, and (3) compatibility with existing models. 
    1.4 Current methods fail to meet these requirements; they either discover features that can't be manipulated, enable manipulations that aren't interpretable, or require expensive model retraining.
    1.5. SAEs from NLP provide unified solution: interpretable features that can be precisely controlled to validate hypotheses.
    1.6. Contributions: SAE for vision model, new understanding of differences in vision models, multiple control examples across tasks

2. Background & Related Work
    2.1. Vision model interpretability
    2.2. Model editing
    2.4. SAEs in language models

3. Method
    3.1. SAE architecture and training
    3.2. Feature intervention framework
        3.2.1. Train a (or use an existing pre-trained) task-specific head: make predictions based on [CLS] activations, use an LLM to generate captions based on an image and a prompt, etc.
        3.2.2. Manipulate the vision transformer activations using the SAE-proposed features and compare outputs before/after intervention.
    3.4. Evaluation metrics

4. Understanding Results
    4.1. Pre-Training Modality Affects Learned Features - DINOv2 vs CLIP
    4.2. Pre-Training Distritbuion Affects Learned Features - CLIP vs BioCLIP

5. Control Results - Task & Model Agnostic
    5.1. Semantic Segmentation Control
        * Intro explaining?
        * Technical description of training linear semseg head on DINOv2 features.
        * Qualitative results (cherry picked examples, full-width figure)
        * [MAYBE] Description of how we automatically find ADE20K class features in SAE latent space
        * [MAYBE] Quantitative results (single-column table)
    5.2. Image Classification Control
        * Birds with interpretable traits
    5.3. Vision-Language Control
        * Counting + removing objects
        * Colors + changing colors
        * Captioning (classification

6. Discussion
    6.1. Limitations
    6.2. Future work


## List of Figures

1. Hook figure: Full width explanatory figure that shows an overview of how we can use SAEs to interpret vision models and then intervene on that explanation and see how model predictions change. Status: visual outline
2. CLIP vs DINOv2: Full width figure demonstrating that CLIP learns semntically abstract visual features like "human teeth" across different visual styles, while DINOv2 does not. Status: visual outline
3. CLIP vs BioCLIP: Full width figure demonstrating some difference in CLIP and BioCLIP's learned features. Status: untouched.
4. Semantic segmentation: Full width figure demonstrating that we can validate patch-level hypotheses. Status: drafted
5. Image-classification: Full width figure demonstrating how you can manipulate fine-grained classification with SAEs. Status: untouched
6. Image captioning: Full width figure. Status: untouched

I also want to build some interactive dashboards and tools to demonstrate that this stuff works.

1. I want my current PCA dashboard with UMAP instead
2. Given a linear classifier of semantic segmentation features, I want to manipulate the features in a given patch, and apply the suppression to all patches to see the live changes on the segmentation mask.
3. After training an SAE on a CLS token, I can then train a linear classifier on the CLS token with ImageNet-1K, and manipulate the features directly.
4. Given a small vision-language model like Phi-3.5 or Moondream, I want to manipulate the vision embeddings (suppressing or adding one or more features) and then see how the top 5 responses change in response to the user input (non-zero temperature).
5. Given a zero-shot CLIP or SigLIP classifier, you can add subtract features from all patches, then see how the classification changes

---

With respect to writing, we want to frame everything as Goal->Problem->Solution. In general, I want you to be skeptical and challenging of arguments that are not supported by evidence.

Some questions that come up that are not in the outline yet:

Q: Am you using standard SAEs or have you adopted the architecture?

A: I am using ReLU SAEs with an L1 sparsity term and I have constrained the columns of W_dec to be unit norm to prevent shrinkage. We are not using sigmoid or tanh activations because of prior work from Anthropic exploring the use of these activation functions, finding them to produce worse features than ReLU.

Q: What datasets are you using?

A: I am using ImageNet-1K for training and testing. I am extending it to iNat2021 (train-mini, 500K images) to demonstrate that results hold beyond ImageNet. 

We're going to work together on writing this paper, so I want to give you an opportunity to ask any questions you might have.

It can be helpful to think about this project from the perspective of a top machine learning researcher, like Lucas Beyer, Yann LeCun, or Francois Chollet. What would they think about this project? What criticisms would they have? What parts would be novel or exciting to them?



