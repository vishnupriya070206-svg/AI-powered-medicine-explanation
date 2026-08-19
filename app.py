import streamlit as st
from PIL import Image
import pandas as pd

from Model import predict
from gemini_explainer import explain_medicine


st.set_page_config(
    page_title="Medicine Classifier",
    page_icon="💊"
)


# Load medicine database
df = pd.read_excel(
    "drug list.xlsx",
    engine="openpyxl"
)


st.title("💊 Medicine Package Classifier")


uploaded_file = st.file_uploader(
    "Upload medicine image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    # Predict medicine
    prediction, confidence = predict(image)

    st.success(
        f"Prediction: {prediction}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # Search medicine details
    prediction_lower = prediction.strip().lower()

    # Normalize common naming differences such as "&" and "and"
    prediction_normalized = prediction_lower.replace("&", "and")

    # Match database medicine names against the predicted class name
    result = df[
        df["Name"].astype(str).apply(
            lambda name: (
                str(name).strip().lower().replace("&", "and")
                in prediction_normalized
            )
        )
    ]

    if not result.empty:

        st.subheader("💊 Medicine Information")

        st.dataframe(
            result
        )

        # Convert retrieved medicine information to text for Gemini
        medicine_info = result.iloc[0].to_dict()

        medicine_info_text = "\n".join(
            f"{key}: {value}"
            for key, value in medicine_info.items()
        )

        # Generate AI explanation
        st.subheader("🤖 AI Medicine Explanation")

        with st.spinner("Generating medicine explanation..."):

            try:

                explanation = explain_medicine(
                    prediction,
                    medicine_info_text
                )

                st.write(explanation)

            except Exception as e:

                st.error(
                    f"Gemini explanation could not be generated: {e}"
                )

    else:

        st.warning(
            "Medicine details not found in database."
        )

    st.warning(
        "⚠️ AI prediction only. Verify medicine packaging before use."
    )

    st.info(
        "This system provides informational medicine explanations "
        "and is not a medical prescription. Please consult a qualified "
        "healthcare professional before using or changing any medication."
    )