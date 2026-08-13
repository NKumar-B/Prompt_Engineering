from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Initialize Ollama model
model = ChatOllama(
    model="gemma3:270m"
)

openEndedPromptTemplate = PromptTemplate(
    input_variables=[
        "role",
        "subject",
        "topic",
        "plagarism",
        "grammar",
        "audience"
    ],
    template="""
Act like a professional {role}.

Your goal is to write a long chapter on the topic "{topic}"
from the subject "{subject}".

Maintain a plagiarism level of {plagarism}.
Throughout the chapter maintain grammar at the {grammar} level.
Ensure that an audience at the {audience} level can easily
understand the concept.
"""
)

final_prompt = openEndedPromptTemplate.format(
    role="university level professor",
    subject="Machine Learning",
    topic="Bias & Variance",
    plagarism="0%",
    grammar="beginner",
    audience="beginner"
)

response = model.invoke(final_prompt)

print(response.content)