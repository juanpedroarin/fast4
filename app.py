import gradio as gr
from fastai.vision.all import *

def is_cat(x): return x[0].isupper()

learn = load_learner('model.pkl')
labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="FastAI image classifier"
)

demo.launch(server_name="0.0.0.0", server_port=7860)