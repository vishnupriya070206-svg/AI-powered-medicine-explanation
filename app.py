import streamlit as st
from PIL import Image
import pandas as pd

from Model import predict


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


    prediction, confidence = predict(image)


    st.success(
        f"Prediction: {prediction}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )


    # Search medicine details
    medicine_name = prediction.split()[0]


    result = df[
        df["Name"].astype(str).str.contains(
            medicine_name,
            case=False,
            na=False
        )
    ]


    if not result.empty:

        st.subheader("💊 Medicine Information")

        st.dataframe(
            result
        )

    else:

        st.warning(
            "Medicine details not found in database."
        )


    st.warning(
        "AI prediction only. Verify medicine packaging before use."
    )