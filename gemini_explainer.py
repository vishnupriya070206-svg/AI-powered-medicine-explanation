import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def explain_medicine(medicine_name, medicine_info):
    prompt = f"""
You are an AI medicine information assistant.

The image classification system identified the medicine as:
{medicine_name}

Here is information retrieved from the application's medicine database:
{medicine_info}

Using ONLY the information provided above, explain the medicine in simple,
easy-to-understand language.

Include:
1. Medicine name
2. What it is / its purpose
3. How it is generally used, if the database provides this information
4. Important precautions or warnings, if provided
5. Side effects, if provided

Do NOT invent dosage instructions.
Do NOT prescribe the medicine.
Do NOT claim that the medicine is safe for a particular person.
If information is missing, clearly say that the information is not available
in the database.

End with:
"Please consult a qualified healthcare professional before using or changing
any medication."

This is an informational explanation, not a medical prescription.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "service_tier": "flex"
        }
    )

    return response.text