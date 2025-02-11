# vs-denoise

### VapourSynth denoising, regression, and motion-compensation functions

<br>

Wrappers for denoising, regression, and motion-compensation-related plugins and functions.

For support you can check out the [JET Discord server](https://discord.gg/XTpc6Fa9eB). <br><br>

## How to install

Install `vsdenoise` with the following command:

```sh
pip install vsdenoise
```

Or if you want the latest git version, install it with this command:

```sh
pip install git+https://github.com/Jaded-Encoding-Thaumaturgy/vs-denoise.git
```

<br><br>

## Example usage

```py
from vsdenoise import MVToolsPresets, Prefilter, mc_degrain, BM3DCuda, Profile, nl_means

clip = ...

ref = mc_degrain(
    clip, prefilter=Prefilter.DFTTEST, preset=MVToolsPresets.HQ_SAD, thsad=100
)

denoise = BM3DCuda.denoise(
    clip, sigma=0.8, tr=2, profile=Profile.NORMAL, ref=ref, planes=0
)

denoise = nl_means(denoise, tr=2, strength=0.2, ref=ref, planes=[1, 2])
```
