# Related Work

Various papers and internet posts on training SAEs for vision.

## Preprints

[An X-Ray Is Worth 15 Features: Sparse Autoencoders for Interpretable Radiology Report Generation](https://arxiv.org/pdf/2410.03334)
* Haven't read this yet, but Hugo Fry is an author.

## LessWrong

[Towards Multimodal Interpretability: Learning Sparse Interpretable Features in Vision Transformers](https://www.lesswrong.com/posts/bCtbuWraqYTDtuARg/towards-multimodal-interpretability-learning-sparse-2)
* Trains a sparse autoencoder on the 22nd layer of a CLIP ViT-L/14. First public work training an SAE on a ViT. Finds interesting features, demonstrating that SAEs work with ViTs.

[Interpreting and Steering Features in Images](https://www.lesswrong.com/posts/Quqekpvx8BGMMcaem/interpreting-and-steering-features-in-images)
* Havne't read it yet.

[Case Study: Interpreting, Manipulating, and Controlling CLIP With Sparse Autoencoders](https://www.lesswrong.com/posts/iYFuZo9BMvr6GgMs5/case-study-interpreting-manipulating-and-controlling-clip)
* Followup to the above work; haven't read it yet.

[A Suite of Vision Sparse Autoencoders](https://www.lesswrong.com/posts/wrznNDMRmbQABAEMH/a-suite-of-vision-sparse-autoencoders)
* Train a sparse autoencoder on various layers using the TopK with k=32 on a CLIP ViT-L/14 trained on LAION-2B. The SAE is trained on 1.2B tokens including patch (not just [CLS]). Limited evaluation.
