# Tools for VLM-1

<p align="center">
    <a href="https://pypi.org/project/vlm-tools/">
        <img alt="PyPi Version" src="https://badge.fury.io/py/vlm-tools.svg">
    </a>
    <a href="https://pypi.org/project/vlm-tools/">
        <img alt="PyPi Version" src="https://img.shields.io/pypi/pyversions/vlm-tools">
    </a>
</p>


`vlm-tools` provides a convenient set of tools to interact with the [VLM-1](https://autonomi.ai/vlm-1) API.


## Installation

The `vlm-tools` python package is available on PyPI. You can install it using pip:

```sh
pip install vlm-tools
```

Optionally, if you want to install the `torch` extra dependencies, you can do so by installing the package with the `torch` extra:

```sh
pip install vlm-tools[torch]
```


## Authentication

The VLM-1 API requires an API key to authenticate requests. You can obtain an API key by signing up on the [waitlist](https://airtable.com/appjX6543bChjNaEN/pagnciKtynSt4rOT9/form). Once you have an API key, you can set it in the environment variable `VLM_API_KEY`:

```bash
export VLM_API_KEY='...'
```

## Usage

### Image -> JSON

The python client needs to be configured with your personal API key before you can use it. You can set the API key in the environment variable `VLM_API_KEY` as described above or pass it to the client constructor.

```python
from vlm_tools.api import vlm

>>> image = Image.open(...)
>>> response_json = vlm(image, domain="document.presentation")
>>> response_json
{
  "description": "...",
  "title": "Differentiated Operating Model",
  "page_number": 7,
  "plots": [
    ...
  ],
  "tables": null,
  "others": [
    ...
  ]
}
```

### Streaming Image -> JSON

You can also stream the image to the API using the `stream` method. This is useful when you want to stream images to our API sequentially, and one-by-one. We provide some helper functions to help you with sampling unique-images (content-based sampler) from a video stream.

```python
from vlm_tools.video import VideoReader, VideoItertools

>>> itertools = VideoItertools()
>>> video = VideoReader("path/to/video.mp4")
>>> for img in itertools.islice(video, similarity_threshold=0.9)
...     response_json = vlm(img, domain="document.presentation")
```

## Requirements

 - Python 3.10+

 We currently support Python 3.10+ on Linux and macOS. If you have any questions or issues, please [open an issue](https://github.com/autonomi-ai/vlm-cookbook/issues).
