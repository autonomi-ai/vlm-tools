import requests
from loguru import logger
from PIL import Image
from tqdm import tqdm

from vlm_tools.constants import VLM_TOOLS_CACHE_DIR
from vlm_tools.video import VideoItertools, VideoReader

URL = "https://zackakil.github.io/video-intelligence-api-visualiser/assets/test_video.mp4"


def test_content_based_frame_sampler():
    path = VLM_TOOLS_CACHE_DIR / "test_data/test_video.mp4"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(requests.get(URL).content)

    video = VideoReader(str(path))
    logger.info(f"video={video}")
    itertools = VideoItertools()

    n = len(video)
    for _idx, img in tqdm(enumerate(itertools.islice(video, similarity_threshold=0.9)), desc="Processing frames"):
        img = Image.fromarray(img).convert("RGB")
    logger.info(f"Processed {_idx + 1}/{n} frames, sampling rate={(_idx + 1) * 100. / n:.2f}%")
