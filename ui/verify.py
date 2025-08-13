import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from handnet import find_device
from .utils import load_model_for_verify, embed_images


def render_verify_tab():
    st.subheader("Verify (cosine similarity)")
    v_weights = st.text_input("Model weights (.pt)", value=st.session_state.get("v_weights", ""), key="vwt")

    enroll_files = st.file_uploader("Enroll images (1-5)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    probe_file = st.file_uploader("Probe image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    if st.button("Compute Similarity"):
        if not v_weights:
            st.error("Please provide a model weights (.pt) file.")
        elif not enroll_files or probe_file is None:
            st.error("Please upload at least one enroll image and one probe image.")
        else:
            device = find_device("auto")
            model, ck_args = load_model_for_verify(v_weights, device)
            img_size = int(ck_args.get('img_size', 224))

            enroll_imgs = [Image.open(f).convert('RGB') for f in enroll_files]
            probe_img = Image.open(probe_file).convert('RGB')

            z_enroll = embed_images(model, enroll_imgs, img_size, device)
            z_probe = embed_images(model, [probe_img], img_size, device)

            z_mean = z_enroll.mean(0, keepdim=True)
            score = F.cosine_similarity(z_mean, z_probe).item()
            st.metric("Cosine similarity", f"{score:.4f}")

