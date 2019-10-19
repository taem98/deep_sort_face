import argparse


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT with detection")
    parser.add_argument(
        "--sequence_dir", help="Path to sequence directory, either image file or whole video",
        default=r"/media/msis_dasol/1TB/dataset/test/MOT16-06", required=True)
    parser.add_argument(
        "--running_name", help="Name of this running session to custom detections.",
        default="default", required=False)
    parser.add_argument(
        "--running_cfg", help="Running configuration: from_detect, from_encoded and track",
        default="track")
    parser.add_argument(
        "--gpu", help="GPU id to running the detection and encoder",
        default=0, type=int
    )
    parser.add_argument(
        "--sequence", help="specify the sequence or the video file name to run or leave empty to run all",
        default=""
    )
    parser.add_argument(
        "--bind_port", help="address to bind Embedding server",
        default=0, type=int
    )
    parser.add_argument(
        "--server_addr", help="address to connect to Embedding server",
        default=""
    )

    parser.add_argument(
        "--mc_mode", help="multiple camera tracker mode: 0: async, 1: sync master, 2: sync slave",
        default=2, type=int
    )

    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.3, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=25, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.1)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=200)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--save_video", help="Save the result videos",
        default=False, type=bool_string)
    return parser