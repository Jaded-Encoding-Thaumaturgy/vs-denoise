# vs-denoise

### VapourSynth denoising, regression, and motion-compensation functions
<br>

Wrappers for denoising, regression, and motion-compensation-related plugins and functions.

You can find me in the [IEW Discord server](https://discord.gg/qxTxVJGtst), @Setsugennoao#6969.
<br><br>
## How to install

Install `vsdenoise` with the following command:

```sh
$ pip install vsdenoise
```

Or if you want the latest git version, install it with this command:

```sh
$ pip install git+https://github.com/Irrational-Encoding-Wizardry/vs-denoise.git
```
<br><br>

## Example usage
```py
from vsdenoise import MVTools, SADMode, MotionMode, Prefilter, BM3DCuda, Profile, nl_means

clip = ...

ref = MVTools.denoise(
    clip, thSAD=100, block_size=32, overlap=8,
    motion=MotionMode.HIGH_SAD,
    prefilter=Prefilter.DFTTEST,
    sad_mode=(
        SADMode.ADAPTIVE_SPATIAL_MIXED,
        SADMode.ADAPTIVE_SATD_MIXED,
    )
)

denoise_luma = BM3DCuda.denoise(
    clip, sigma=[0.8, 0], radius=2, profile=Profile.NORMAL, ref=ref
    )

denoise_chroma = nl_means(
    denoise_luma, tr=2, rclip=ref, strength=0.6, wmode=3, planes=[1, 2],
)
```
