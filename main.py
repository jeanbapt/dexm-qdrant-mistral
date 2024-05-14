
import os
import dotenv
import dspy
from qdrant_client import QdrantClient
from datasets import load_dataset
from dspy.retrieve.qdrant_rm import QdrantRM

dotenv.load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY") 
client = QdrantClient("localhost", port=6333)

dataset = load_dataset('NebulaByte/E-Commerce_Customer_Support_Conversations')
df_pandas = dataset['train'].to_pandas()
documents = df_pandas['conversation'].tolist()

ids = list(range(1,len(documents)+1))
client.delete_collection(collection_name= "informations")
client.set_model("sentence-transformers/all-MiniLM-L6-v2") 
client.add(collection_name= "informations", documents=documents, ids=ids)   

# Now set up the RAG model
collection_name = "informations"
llm= dspy.OllamaLocal(model='mistral')
# llm = dspy.Mistral(model='mistral-small-latest', api_key=mistral_api_key)  
qdrant_retriever_model = QdrantRM(collection_name, client, k=10)

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
print(response.answer)