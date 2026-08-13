from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

model = init_chat_model(
    model="gemma3:270m",
    model_provider="ollama"
)

closedEndedPromptTemplate = PromptTemplate(
    input_variables=["fact"],
    template="""Analyze the given fact: {fact}.
Just output whether the fact is right or wrong.
The output should contain only one word: right or wrong."""
)

final_prompt = closedEndedPromptTemplate.format(
    fact="Sun Rises In The East"
)

response = model.invoke(final_prompt)

print(response.content)

if response.content.strip().lower() == "right" or response.content.strip().lower()=="wrong":
    print("THE GIVEN PROMPT IS A CLOSED ENDED PROMPT")
else:
    print("THE PROMPT GIVEN IS NOT A CLOSED ENDED PROMPT")