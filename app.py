
"""
Image Recognition App

Uses pretrained Vision Transformer or ResNet models
from Hugging Face to recognize objects in images.
"""

import streamlit as st
from PIL import Image
from transformers import pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Image Recognition",
    page_icon="🖼️",
    layout="centered"
)


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "ViT Base (Google)": "google/vit-base-patch16-224",
    "ResNet-50 (Microsoft)": "microsoft/resnet-50"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_classifier(model_id):
    return pipeline(
        "image-classification",
        model=model_id
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "🤖 Select Model",
    list(MODELS.keys())
)

top_k = st.sidebar.slider(
    "🔢 Number of Predictions",
    1,
    10,
    5
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Models are trained on ImageNet "
    "with 1,000 common object categories."
)


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🖼️ Image Recognition")

st.write(
    "Upload an image or take a photo, "
    "and the AI model will identify what it sees."
)


# ============================================================
# IMAGE INPUT
# ============================================================

tab_upload, tab_camera = st.tabs(
    ["📁 Upload Image", "📷 Camera"]
)

image = None


# ============================================================
# UPLOAD TAB
# ============================================================

with tab_upload:

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )

    if uploaded_file is not None:

        try:
            image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception as error:
            st.error(
                f"Unable to open image: {error}"
            )


# ============================================================
# CAMERA TAB
# ============================================================

with tab_camera:

    camera_photo = st.camera_input(
        "Take a picture"
    )

    if camera_photo is not None:

        try:
            image = Image.open(
                camera_photo
            ).convert("RGB")

        except Exception as error:
            st.error(
                f"Unable to open camera image: {error}"
            )


# ============================================================
# DISPLAY IMAGE
# ============================================================

if image is not None:

    st.image(
        image,
        caption="📸 Input Image",
        use_container_width=True
    )

    st.markdown("---")

    # ========================================================
    # RECOGNIZE BUTTON
    # ========================================================

    if st.button(
        "🔍 Recognize Image",
        type="primary",
        use_container_width=True
    ):

        try:

            # Get selected model
            model_id = MODELS[model_name]

            # Load model
            with st.spinner(
                "🤖 Loading AI model..."
            ):
                classifier = load_classifier(
                    model_id
                )

            # Analyze image
            with st.spinner(
                "🔍 Analyzing image..."
            ):
                results = classifier(
                    image,
                    top_k=top_k
                )

            # =================================================
            # TOP RESULT
            # =================================================

            best = results[0]

            label = best["label"]
            score = best["score"] * 100

            st.success(
                f"🏆 Top Prediction: "
                f"**{label}** — {score:.2f}%"
            )

            # =================================================
            # ALL RESULTS
            # =================================================

            st.subheader(
                "📊 All Predictions"
            )

            for number, result in enumerate(
                results,
                start=1
            ):

                result_label = result["label"]
                result_score = result["score"]

                percentage = result_score * 100

                st.write(
                    f"**{number}. {result_label}** "
                    f"— {percentage:.2f}%"
                )

                st.progress(
                    float(result_score)
                )

        except Exception as error:

            st.error(
                "❌ Failed to recognize the image."
            )

            st.info(
                "Make sure the required packages are "
                "installed and that you have an internet "
                "connection for the first model download."
            )

            with st.expander(
                "Show technical error"
            ):
                st.code(
                    str(error)
                )


# ============================================================
# NO IMAGE MESSAGE
# ============================================================

else:

    st.info(
        "👆 Upload an image or use the camera "
        "to get started."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🖼️ Image Recognition | "
    "Powered by Streamlit + Hugging Face Transformers"
)
