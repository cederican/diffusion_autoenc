# Official implementation of Diffusion Autoencoders

A CVPR 2022 (ORAL) paper ([paper](https://openaccess.thecvf.com/content/CVPR2022/html/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.html), [site](https://diff-ae.github.io/), [5-min video](https://youtu.be/i3rjEsiHoUU)):

```
@inproceedings{preechakul2021diffusion,
      title={Diffusion Autoencoders: Toward a Meaningful and Decodable Representation}, 
      author={Preechakul, Konpat and Chatthee, Nattanat and Wizadwongsa, Suttisak and Suwajanakorn, Supasorn},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
      year={2022},
}
```

### Prerequisites

See `requirements.txt`

```
pip install -r requirements.txt
```

### LMDB Datasets

We do not own any of the following datasets. We provide the LMDB ready-to-use dataset for the sake of convenience.

- [MRI_COR]
- [MRI_COR2]

The directory tree should be:

```
datasets/
- FastMRI_cor2.lmdb
- FastMRI_cor.lmdb
```


## Training

We provide scripts for training & evaluate DiffAE (including latent DPM) on the following datasets: MRI Knee COR.
Usually, the evaluation results (FID's) will be available in `eval` directory.

Note: Most experiment requires at least 4x V100s during training the DPM models while requiring 1x 2080Ti during training the accompanying latent DPM. Or reduce the batch size.



**MRI Knee COR**
```
# diffae
python run_mri.py
```

A classifier (for manipulation) can be trained using:
```
python run_mri_cls.py
```

## Evaluation

```
# evaluate diffusion autoencoder
python test_autoencoding.py

# evaluate latent sampling and interpolation
python test_interpolate.py
python test_sample.py

# evaluate sequence conversion
python test_manipulate.py

# cool visualizations
python test_giftorow.py
python test_diffaenoise.py

```