import gradio as gr
from dino import Dino
from autodistill.detection import CaptionOntology
from dataset import Dataset
import benchmark as bench
from typing import List, Tuple
import numpy as np

dataset = Dataset("/Users/francescotacinelli/Developer/datasets/pallets_sorted/test/")
cap_ont = dataset.captions.captions_ontology

try:
    model = Dino(ontology=CaptionOntology(cap_ont))
except Exception as e:
    raise e


def predict(img_path: np.ndarray, alias, ground_truths: List[Tuple[int, List[float]]]):
    predictions = model.gradio_predict(img_path, alias)
    metrics = get_pr(predictions, ground_truths=ground_truths)
    return (img_path, predictions), metrics


def get_pr(predictions, ground_truths, class_id=2):
    true_positives, false_positives, false_negatives = bench.get_confMatr(
        predictions, ground_truths, class_id
    )
    labels = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
    return labels


with gr.Blocks() as demo:
    label = gr.Textbox(label="Label", placeholder="Enter your label here")
    input = gr.Image(label="Image", type="numpy")
    grounds = gr.File(label="Ground Truth", type="filepath", file_types=[".txt"])
    pred_btn = gr.Button(value="Predict")
    output = gr.AnnotatedImage(
        label="Output",
    )
    metrics = gr.Label(label="Metrics")
    pred_btn.click(
        predict,
        inputs=[input, label, grounds],
        outputs=[output, metrics],
        api_name="predict",
    )
