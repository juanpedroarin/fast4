import os
import gradio as gr
from fastai.vision.all import *

def is_cat(x): return x[0].isupper()

learn = None

def load_model():
    global learn
    learn = load_learner('model.pkl')

def predict(img):
    if learn is None:
        load_model()
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    labels = learn.dls.vocab
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="FastAI image classifier"
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))