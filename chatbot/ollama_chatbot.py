import ollama

def ask_tinyllama(plant_name, plant_info, question):

    context = f"""
You are a medicinal plant expert.

Plant Name: {plant_name}

Scientific Name: {plant_info.get("scientific_name","")}
Family: {plant_info.get("family","")}
Description: {plant_info.get("description","")}

Uses:
{plant_info.get("uses","")}

Parts Used:
{plant_info.get("parts_used","")}

Dosage:
{plant_info.get("dosage","")}

Precautions:
{plant_info.get("precautions","")}

User Question:
{question}
"""

    response = ollama.chat(
        model="tinyllama",
        messages=[
            {"role": "user", "content": context}
        ]
    )

    return response["message"]["content"]