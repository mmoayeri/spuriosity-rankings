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

# License

COPYRIGHT AND PERMISSION NOTICE
UMD Software Spuriosity Rankings Copyright (C) 2024 University of Maryland
All rights reserved.
The University of Maryland (“UMD”) and the developers of Salient ImageNet software (“Software”) give recipient (“Recipient”) permission to download a single copy of the Software in source code form and use by university, non-profit, or research institution users only, provided that the following conditions are met:
1)  Recipient may use the Software for any purpose, EXCEPT for commercial benefit.
2)  Recipient will not copy the Software.
3)  Recipient will not sell the Software.
4)  Recipient will not give the Software to any third party.
5)  Any party desiring a license to use the Software for commercial purposes shall contact:
UM Ventures, College Park at UMD at otc@umd.edu.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS, CONTRIBUTORS, AND THE UNIVERSITY OF MARYLAND "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO  EVENT SHALL THE COPYRIGHT OWNER, CONTRIBUTORS OR THE UNIVERSITY OF MARYLAND BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
