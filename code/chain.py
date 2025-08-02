from langchain_community.llms import ctransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

model_file = "models/vinallm-7b-chat_q5_0.gguf"


def load_model(model_file: str):
    llm = ctransformers.CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.1
    )
    return llm

def create_prompt_chain(template: str):
    prompt = PromptTemplate(template = template, input_variables=["question"])
    return prompt
def create_chain(prompt, llm):
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt
    )
    return llm_chain

#test

template = '''<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
Hello world!<|im_end|>
<|im_start|>assistant'''

llm = load_model(model_file)
prompt = create_prompt_chain(template)
llm_chain = create_chain(prompt, llm)

question = "mot cong mot bang may?"
response = llm_chain.invoke({"question": question})
print(response)