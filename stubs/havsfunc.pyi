import vapoursynth as vs

from typing import Optional, Union, Sequence

def MinBlur(clp: vs.VideoNode, r: int = 1, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode: ...