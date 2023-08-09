import bisect
import re
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

from elias.util import ensure_directory_exists_for_file


def make_wandb_video(wandb_project: str,
                     run_id: str,
                     image_key: str,
                     output_folder: str,
                     /,
                     max_n_frames: int = 30,
                     fps: int = 10):
    """
    Collects all images that were logged with a given key to a wandb run.
    Images are packed into a .mp4 video for playback.

    Parameters
    ----------
        wandb_project:
            name of the wandb project
        run_id:
            Id of the wandb run. Can be retrieved from the URL: https://wandb.ai/$USER/diff-vp/runs/plh665gq <- run ID
        image_key:
            Which subset of images that are logged to the run should be downloaded. The image key is usually written
            in bold font above the logged image in the dashboard
        output_folder:
            Path to a folder in which the .mp4 will be stored
        max_n_frames:
            Sometimes, many images are logged to a run.
            To avoid excessive downloads, a maximum number of frames is specified
        fps:
            Playback speed of the generated video
    """

    # TODO: Currently, models are imported on-demand
    #   Wanted to avoid adding all these dependencies just for one method
    #   Maybe need yet another different repository for these kind of training/logging helpers?
    import wandb
    from tqdm import tqdm
    import requests
    import mediapy

    entity = wandb.setup()._get_username()
    api = wandb.Api()
    run = api.run(f"{entity}/{wandb_project}/{run_id}")

    output_path = f"{output_folder}/{run.name}_{image_key.replace('/', '-')}.mp4"
    ensure_directory_exists_for_file(output_path)

    # Collect all image urls from the wandb run that match the specified image key
    extract_step_regex = re.compile(f"media/images/{re.escape(image_key)}_(\d+).*")
    image_files = OrderedDict()
    for file in tqdm(run.files(), desc="Collect images"):
        matched = extract_step_regex.match(file.name)
        if matched:
            step = int(matched.group(1))
            image_files[step] = file

    image_files = OrderedDict(sorted(image_files.items()))
    image_files_values = list(image_files.values())

    # Select at most max_n_frames images
    logged_steps = list(image_files.keys())
    min_step = min(logged_steps)
    max_step = max(logged_steps)
    n_steps = len(logged_steps)
    if n_steps < max_n_frames:
        # Use all logged images
        steps = logged_steps
    else:
        # Use an equally spaced selection of images
        steps = np.linspace(min_step, max_step, max_n_frames)

    writer = None
    for step in tqdm(steps, desc="Downloading images"):
        # Find image file with step closest to requested step
        closest_step_idx = bisect.bisect_left(logged_steps, step)
        file = image_files_values[closest_step_idx]
        closest_step = logged_steps[closest_step_idx]

        # Actually download artifact and parse into image
        response = requests.get(file.url, auth=("api", api.api_key), stream=True, timeout=5)
        image = Image.open(response.raw)
        image_numpy = np.asarray(image)

        # Font-face:
        # 1 -> 20px
        # 2 -> 40px
        # Good height is ~10% of image width
        font_face = image_numpy.shape[1] / 10 / 20
        text = f"{closest_step}"
        cv2.putText(image_numpy, text, (10, image_numpy.shape[0] - 10), 0, font_face, (0, 255, 0))

        # Fuse image into video and stream to disk
        if writer is None:
            render_width = int(image_numpy.shape[1])
            render_height = int(image_numpy.shape[0])
            writer = mediapy.VideoWriter(
                path=output_path,
                shape=(render_height, render_width),
                fps=fps,
            )
            writer.__enter__()
        writer.add_image(image_numpy)

    writer.__exit__()
