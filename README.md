# Spuriosity Rankings

This is the official code repository for the NeurIPS 2023 Spotlight work [**Spuriosity Rankings: Sorting Data to Measure and Mitigate Biases**](https://arxiv.org/abs/2212.02648) by Mazda Moayeri, Wenxiao Wang, Sahil Singla, and Soheil Feizi. 

For a quick overview, checkout `demo.ipynb`, which walks you through the code and the method.

# Setup and Quick Use

You'll need the adversarially trained feature encoder, which can be downloaded from [here](https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0). Update the path for `_ROBUST_MODELS_ROOT` (line 25 in `spuriosity_rankings.py`) to point to wherever you save that model. 

All you need after that is to set up the dataset you wish to explore! The requirements for the dataset object are that they (i) return (sample, int) pairs, where the sample is an image or tensor and the int is the class index for that sample, and (ii) have a field `dset.classes` with names for each class.

Then, after setting `_RESULTS_ROOT` to wherever you want results to be saved, you can run `feature_discovery()` to discover important features (and hidden minority subpopulations) in your dataset. Then, if you find per-class biases that you don't want your model to have, you can create a dictionary called `spurious_ftrs_by_class`, and feed that (along with a classification head you wish to de-bias) to `mitigate_bias()`.

# Citation

If this work is of use to you, we would be grateful if you could include the following citation in your work:

```
@inproceedings{
  moayeri2023spuriosity,
  title={Spuriosity Rankings: Sorting Data to Measure and Mitigate Biases},
  author={Mazda Moayeri and Wenxiao Wang and Sahil Singla and Soheil Feizi},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=jSuhnO9QJv}
}
```
