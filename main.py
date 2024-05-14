
import os
import dotenv
import dspy
from qdrant_client import QdrantClient
from datasets import load_dataset
from dspy.retrieve.qdrant_rm import QdrantRM

from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dspy.predict.langchain import LangChainModule, LangChainPredict
from langchain_openai import OpenAI

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# mistral_api_key = os.getenv("MISTRAL_API_KEY") 
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

""" client = QdrantClient("localhost", port=6333)
dataset = load_dataset('NebulaByte/E-Commerce_Customer_Support_Conversations')
df_pandas = dataset['train'].to_pandas()
documents = df_pandas['conversation'].tolist()
ids = list(range(1,len(documents)+1))
client.delete_collection(collection_name= "informations")
client.set_model("sentence-transformers/all-MiniLM-L6-v2") 
client.add(collection_name= "informations", documents=documents, ids=ids)   
 """
# Now set up the RAG model
""" collection_name = "informations"
llm= dspy.OllamaLocal(model='mistral') """
# llm = dspy.Mistral(model='mistral-small-latest', api_key=mistral_api_key)  
""" qdrant_retriever_model = QdrantRM(collection_name, client, k=10)

dspy.settings.configure(lm=llm, rm=qdrant_retriever_model)
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

uncompiled_rag = RAG()
example_query = "Tell me about the instances when the customer's camera broke"
response = uncompiled_rag(example_query)
print(response.answer) """

colbertv2 = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.settings.configure(llm=llm, rm=colbertv2)
set_llm_cache(SQLiteCache(database_path="cache.db"))

def retrieve(inputs):
    return [doc["text"] for doc in colbertv2(inputs["question"], k=5)]

print(colbertv2("cycling"))

prompt = PromptTemplate.from_template(
    "Given {context}, answer the question `{question}` as a tweet."
)

# This is how you'd normally build a chain with LCEL. This chain does retrieval then generation (RAG).
zeroshot_chain = (
    RunnablePassthrough.assign(context=retrieve)
    | LangChainPredict(prompt, llm)
    | StrOutputParser()
)

zeroshot_chain = LangChainModule(zeroshot_chain)
question = "In what region was Eddy Mazzoleni born?"
responses = zeroshot_chain.invoke({"question": question})
print(responses)