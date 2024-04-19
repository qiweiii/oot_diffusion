# OOTDiffusion

A packaged version of [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) that works with Pip.

No need to manually download models, checkpoints, weights etc. Should work out if the box.

Needs CUDA and GPU.

OOTDiffusion 的打包版本.

[Try on Replicate: dc (full body)](https://replicate.com/qiweiii/oot_diffusion_dc)

<!-- [Try on Replicate: hd (half body)](https://replicate.com/qiweiii/oot_diffusion_hd) -->

## Qiwei Notes

- **This private repo is created for deploying the full-body version api, so I can use them in my project**
- Decision
  - [cog-hd] use the viktorfa one for half-body api (later I can deploy one as well if needed)
  - [cog-dc] use my deployed one for full-body api


## Instructions

`pip install git+https://github.com/viktorfa/oot_diffusion.git`

Examples for Colab. But can be used anywhere.

If you don't set hg_root, a folder called ootd_models will be created in your working dir.

Load model
```python
from oot_diffusion import OOTDiffusionModel
from PIL import Image
from pathlib import Path


def get_ootd_model():
  model = OOTDiffusionModel(
    hg_root="/content/models",
    cache_dir="/content/drive/MyDrive/hf_cache",
  )
  return model
```

Generate image
```python
def generate_image():
  model = get_ootd_model()
  generated_images, mask_image = model.generate(
      model_path="/YOUR_MODEL.jpg",
      cloth_path="/YOUR_GARMENT.jpg",
      steps=10,
      cfg=2.0,
      num_samples=2,
    )

  return generated_images, mask_image


generated_images, mask_image = generate_image()
```

Display images

```python
from IPython.display import display

for image in generated_images:
  display(image)

display(mask_image)
```


## Credits

The original author of [packaged ootd](https://github.com/viktorfa/oot_diffusion)

The original author of [oot_cog_samples](https://github.com/viktorfa/oot_cog_samples)

The original authors of [OOTDiffusion](https://github.com/levihsu/OOTDiffusion)

The authors of [ComfyUI-OOTDiffusion](https://github.com/AuroBit/ComfyUI-OOTDiffusion), who made it easier to package the code.

## Official implementation

See [oms-Diffusion](https://github.com/ShineChen1024/oms-Diffusion) for the official implementation of OOTDiffusion.
