import vapoursynth as vs


def merge_chroma(luma: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
    return vs.core.std.ShufflePlanes([luma, ref], [0, 1, 2], vs.YUV)
