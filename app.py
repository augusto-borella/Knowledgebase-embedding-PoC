import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

# 1. Vectorise the sales response csv data
# need to guarantee csv structure
loader = CSVLoader(file_path="Cost Classification2.csv")
#loader = CSVLoader(file_path="Classification_test.csv")
documents = loader.load()
#print(documents)
#print(len(documents))

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=5)

    page_contents_array = [doc.page_content for doc in similar_response]

    #print(page_contents_array)

    return page_contents_array

#cost2Classification = "Localiza Aluguel de carros"

#print(retrieve_info(cost2Classification))

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are a world class accounting. 
I will share a cost description from the bank statement with you and you will give me the cost classification that 
I should apply to the expense based on past best practies, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practices, return just the classification

2/ If the best practice are irrelevant, then try to mimic the style of the best practice to cost description classification

Below is cost description I received from the bank statement:
{description}

Here is a list of best practices of how we normally classify cost description in similar scenarios:
{best_practice}

Please write the best response that I should classify the cost description based on the best practies above
"""

prompt = PromptTemplate(
    input_variables=["description", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(description=message, best_practice=best_practice)
    return response

#description = "Manuel e Joaquim rest"
#response = (generate_response(description))
#print(response)

# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Classificador de Custos Borella", page_icon=":rocket:")

    st.header("Descrição do Item no Extrato :rocket:")
    message = st.text_area("Sugestão de classificação de custo")

    if message:
        st.write("Generating category...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()
