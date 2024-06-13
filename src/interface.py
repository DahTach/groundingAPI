import gradio as gr
from dino import Dino
from autodistill.detection import CaptionOntology
from dataset import Dataset, FILE_PATH
import benchmark as bench
from typing import List, Tuple
from ast import (
    literal_eval,
)  # For safely evaluating the string representation of the list
import numpy as np
import torch

classes = [
    "bitter pack",
    "bottle pack",
    "box",
    "can pack",
    "crate",
    "keg",
]

cache = {
    "bitter pack": {
        "alias": {
            "precision": 0.0,
            "recall": 0.0,
        },
    }
}

dataset = Dataset(FILE_PATH)
cap_ont = dataset.captions.captions_ontology

try:
    model = Dino(ontology=CaptionOntology(cap_ont))
except Exception as e:
    raise e


def convert_to_list_of_tuples(string_representation: str):
    """Converts a string representation of nested tuples to a List[Tuple[int, List[float]]].

    Args:
        string_representation: The string containing the nested tuple data.

    Returns:
        List[Tuple[int, List[float]]]: The parsed data as a list of tuples.
    """

    # Evaluate string into Python object structure
    try:
        data = literal_eval(str(string_representation))
        result = []

        for item in data:
            # Convert each string float list to an actual list of floats
            bbox = [float(coord) for coord in item[1]]
            # Ensure class_id is an integer
            class_id = int(item[0])

            result.append((class_id, bbox))

        return result
    except Exception as e:
        return str(e)


def preprocess_grounds(source_w, source_h, ground_truths, class_id=2):
    ground_truths = convert_to_list_of_tuples(ground_truths)

    ground_truths = [
        torch.Tensor(bbox) for idx, bbox in ground_truths if idx == class_id
    ]

    if len(ground_truths) == 0:
        return torch.Tensor([])
    ground_truths = torch.stack(ground_truths) * torch.Tensor(
        [source_w, source_h, source_w, source_h]
    )
    return ground_truths


# def single_predict(img: np.ndarray, alias, ground_truths, class_id):
#     predictions = model.gradio_predict(img, alias)
#     source_h, source_w, _ = img.shape
#     ground_truths = preprocess_grounds(source_w, source_h, ground_truths, class_id)
#     metrics = get_pr(predictions, ground_truths=ground_truths)
#     return (img, predictions), metrics


def predict(img: np.ndarray, aliases, ground_truths, class_id):
    class_id = classes.index(class_id)
    # remove spaces and split by "."
    aliases = aliases.replace(" ", "").split(".")
    predictions = model.gradio_multi_predict(img, aliases)
    source_h, source_w, _ = img.shape
    ground_truths = preprocess_grounds(source_w, source_h, ground_truths, class_id)
    pred_dict = {
        "aliases": aliases,
        "detections": predictions,
    }
    true_positives, false_positives, false_negatives = bench.get_confMatr(
        predictions, ground_truths
    )
    metrics_dict = {
        "class_id": class_id,
        "alias": aliases,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    print(f"metrics: {metrics}")

    return str(pred_dict), str(metrics_dict)


def validate_grounds(ground_str: str) -> str:
    ground_truths = convert_to_list_of_tuples(ground_str)
    if not all(isinstance(gt, tuple) for gt in ground_truths):
        return "Ground truths must be a list of tuples"
    return "Grounds are in the correct format"


with gr.Blocks() as demo:
    aliases = gr.Textbox(label="Label", placeholder="Enter your label here")
    class_id = gr.Dropdown(
        choices=classes, label="Class ID (corresponds to ordering)", value=classes[2]
    )
    input = gr.Image(label="Image", type="numpy")
    grounds = gr.TextArea(
        label="Ground Truths", placeholder="List[Tuple[int, List[float]]]"
    )
    ground_validation = gr.Textbox(label="GR Validity", interactive=False)
    grounds.input(fn=validate_grounds, inputs=grounds, outputs=ground_validation)
    pred_btn = gr.Button(value="Predict")
    output = gr.Code(label="Output", language="json")
    metrics = gr.Code(label="Metrics", language="json")
    pred_btn.click(
        predict,
        inputs=[input, aliases, grounds, class_id],
        outputs=[output, metrics],
        api_name="predict",
    )
