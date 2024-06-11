import gradio as gr
from dino import Dino
from autodistill.detection import CaptionOntology
from dataset import Dataset
import benchmark as bench
from typing import List, Tuple
from ast import literal_eval  # For safely evaluating the string representation of the list
import numpy as np
dataset = Dataset("/home/francesco/dataset/labeled_pallets")
cap_ont = dataset.captions.captions_ontology

try:
    model = Dino(ontology=CaptionOntology(cap_ont))
except Exception as e:
    raise e



def convert_to_list_of_tuples(string_representation):
    """Converts a string representation of nested tuples to a List[Tuple[int, List[float]]].

    Args:
        string_representation: The string containing the nested tuple data.

    Returns:
        List[Tuple[int, List[float]]]: The parsed data as a list of tuples.
    """
    # Evaluate string into Python object structure
    data = literal_eval(string_representation)
    result = []

    for item in data:
        # Convert each string float list to an actual list of floats
        bbox = [float(coord) for coord in item[1]]
        # Ensure class_id is an integer
        class_id = int(item[0])

        result.append((class_id, bbox))  

    return result

def predict(img_path: np.ndarray, alias, ground_truths: List[Tuple[int, List[float]]]):
    predictions = model.gradio_predict(img_path, alias)
    print(ground_truths)
    ground_truths = convert_to_list_of_tuples(ground_truths)
    print(isinstance(ground_truths, list))
    print(ground_truths)
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

def validate_grounds(ground_truths):
    if not all(isinstance(gt, tuple) for gt in ground_truths):
        return "Ground truths must be a list of tuples"
    return "Grounds are in the correct format"

with gr.Blocks() as demo:
    label = gr.Textbox(label="Label", placeholder="Enter your label here")
    input = gr.Image(label="Image", type="numpy")
    grounds = gr.TextArea(label="Ground Truths", placeholder="List[Tuple[int, List[float]]]")
    ground_validation = gr.Textbox(label="GR Validity", value=grounds.value)
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
    ground_validation.submit(fn=validate_grounds, inputs=grounds, outputs=ground_validation)
