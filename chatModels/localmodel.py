from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

# from model id this line downloads that particular model from hugging face to ur system storage
llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.03,
    )
)

chat_model=ChatHuggingFace(llm=llm)

response=chat_model.invoke("what is the IPL ?")

print(response.content)
