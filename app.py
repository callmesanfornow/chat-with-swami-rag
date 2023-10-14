'''
Om Sri Sai Ram

Swami's Chatbot Alpha Version
'''

from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import textwrap
import gradio as gr
import time
import os

OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

vectordb = FAISS.load_local("faiss_index", OpenAIEmbeddings())

# --------------------------------------------------------------------------------

prompt_template = """
Answer "Sairam, How can I help you!" if you get Sairam as a Question.
Don't try to make up an answer, if you don't know just say that you don't know.
Answer in the same language the question was asked.
Use only the following pieces of context to answer the question at the end.

{context}

{history}

Question: {question}
Answer:"""


PROMPT = PromptTemplate(
    template= prompt_template,
    input_variables=["history","context", "question"]
)

chain = RetrievalQA.from_chain_type(llm= OpenAI(model_name= "gpt-3.5-turbo-0613", temperature= 0),
                                    chain_type="stuff",
                                    retriever= vectordb.as_retriever(),
                                    chain_type_kwargs= {'prompt': PROMPT,
                                                        "verbose": True,
                                                        "memory": ConversationBufferMemory(
                                                                  memory_key="history",
                                                                  input_key="question"),
                                                        },
                                    return_source_documents= True,
                                    verbose= True)

# --------------------------------------------------------------------------------

def wrap_text_preserve_newlines(text, width=200): # 110
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])
    print(llm_response)

    src = {'Bhagavatha Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Bhagavatha/BhagavathaVahiniInteractive.pdf',
            'Dharma Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Dharma/DharmaVahiniInteractive.pdf',
            'Dhyana Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Dhyana/DhyanaVahiniInteractive.pdf',
            'Gita Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Gita/GitaVahiniInteractive.pdf',
            'Jnana Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Jnana/JnanaVahiniInteractive.pdf',
            'Leela Kaivalya Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Leela/LeelaKaivalyaVahiniInteractive.pdf',
            'Prasanthi Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Prasanthi/PrasanthiVahiniInteractive.pdf',
            'Prasnothara Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Prasnottara/PrasnotharaVahiniInteractive.pdf',
            'Prema Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Prema/PremaVahiniInteractive.pdf',
            'Ramakatha Rasavahini 1.pdf' : 'https://www.sssbpt.info/vahinis/RamakathaI/RamakathaRasavahiniIInteractive.pdf',
            'Ramakatha Rasavahini 2.pdf' : 'https://www.sssbpt.info/vahinis/RamakathaII/RamakathaRasavahini2Interactive.pdf',
            'Sandeha Nivarini.pdf' : 'https://www.sssbpt.info/vahinis/Sandeha/SandehaNivariniInteractive.pdf',
            'Sathya Sai Speaks Volume 01 1953 to 1960.pdf' : 'https://www.sssbpt.info/ssspeaks/volume01/sss01.pdf',
            'Sathya Sai Speaks Volume 02 1961 to 1962.pdf' : 'https://www.sssbpt.info/ssspeaks/volume02/sss02.pdf',
            'Sathya Sai Speaks Volume 03 1963.pdf' : 'https://www.sssbpt.info/ssspeaks/volume03/sss03.pdf',
            'Sathya Sai Speaks Volume 04 1964.pdf' : 'https://www.sssbpt.info/ssspeaks/volume04/sss04.pdf',
            'Sathya Sai Speaks Volume 05 1965.pdf' : 'https://www.sssbpt.info/ssspeaks/volume05/sss05.pdf ',
            'Sathya Sai Speaks Volume 06 1966.pdf' : 'https://www.sssbpt.info/ssspeaks/volume06/sss06.pdf',
            'Sathya Sai Speaks Volume 07 1967.pdf' : 'https://www.sssbpt.info/ssspeaks/volume07/sss07.pdf',
            'Sathya Sai Speaks Volume 08 1968.pdf' : 'https://www.sssbpt.info/ssspeaks/volume08/sss08.pdf',
            'Sathya Sai Speaks Volume 09 1969.pdf' : 'https://www.sssbpt.info/ssspeaks/volume09/sss09.pdf',
            'Sathya Sai Speaks Volume 10 1970.pdf' : 'https://www.sssbpt.info/ssspeaks/volume10/sss10.pdf',
            'Sathya Sai Speaks Volume 11 1971 to 1972.pdf' : 'https://www.sssbpt.info/ssspeaks/volume11/sss11.pdf',
            'Sathya Sai Speaks Volume 12 1973 to 1974.pdf' : 'https://www.sssbpt.info/ssspeaks/volume12/sss12.pdf',
            'Sathya Sai Speaks Volume 13 1975 to 1977.pdf' : 'https://www.sssbpt.info/ssspeaks/volume13/sss13.pdf',
            'Sathya Sai Speaks Volume 14 1978 to 1980.pdf' : 'https://www.sssbpt.info/ssspeaks/volume14/sss14.pdf',
            'Sathya Sai Speaks Volume 15 1981 to 1982.pdf' : 'https://www.sssbpt.info/ssspeaks/volume15/sss15.pdf',
            'Sathya Sai Speaks Volume 16 1983.pdf' : 'https://www.sssbpt.info/ssspeaks/volume16/sss16.pdf',
            'Sathya Sai Speaks Volume 17 1984.pdf' : 'https://www.sssbpt.info/ssspeaks/volume17/sss17.pdf',
            'Sathya Sai Speaks Volume 18 1985.pdf' : 'https://www.sssbpt.info/ssspeaks/volume18/sss18.pdf',
            'Sathya Sai Speaks Volume 19 1986.pdf' : 'https://www.sssbpt.info/ssspeaks/volume19/sss19.pdf',
            'Sathya Sai Speaks Volume 20 1987.pdf' : 'https://www.sssbpt.info/ssspeaks/volume20/sss20.pdf',
            'Sathya Sai Speaks Volume 21 1988.pdf' : 'https://www.sssbpt.info/ssspeaks/volume21/sss21.pdf',
            'Sathya Sai Speaks Volume 22 1989.pdf' : 'https://www.sssbpt.info/ssspeaks/volume22/sss22.pdf',
            'Sathya Sai Speaks Volume 23 1990.pdf' : 'https://www.sssbpt.info/ssspeaks/volume23/sss23.pdf',
            'Sathya Sai Speaks Volume 24 1991.pdf' : 'https://www.sssbpt.info/ssspeaks/volume24/sss24.pdf',
            'Sathya Sai Speaks Volume 25 1992.pdf' : 'https://www.sssbpt.info/ssspeaks/volume25/sss25.pdf',
            'Sathya Sai Speaks Volume 26 1993.pdf' : 'https://www.sssbpt.info/ssspeaks/volume26/sss26.pdf',
            'Sathya Sai Speaks Volume 27 1994.pdf' : 'https://www.sssbpt.info/ssspeaks/volume27/sss27.pdf',
            'Sathya Sai Speaks Volume 28 1995.pdf' : 'https://www.sssbpt.info/ssspeaks/volume28/sss28.pdf',
            'Sathya Sai Speaks Volume 29 1996.pdf' : 'https://www.sssbpt.info/ssspeaks/volume29/sss29.pdf',
            'Sathya Sai Speaks Volume 30 1997.pdf' : 'https://www.sssbpt.info/ssspeaks/volume30/sss30.pdf',
            'Sathya Sai Speaks Volume 31 1998.pdf' : 'https://www.sssbpt.info/ssspeaks/volume31/sss31.pdf',
            'Sathya Sai Speaks Volume 32 Part 1 1999.pdf' : 'https://www.sssbpt.info/ssspeaks/volume32/sss32p1.pdf',
            'Sathya Sai Speaks Volume 32 Part 2 1999.pdf' : 'https://www.sssbpt.info/ssspeaks/volume32/sss32p2.pdf',
            'Sathya Sai Speaks Volume 33 2000.pdf' : 'https://www.sssbpt.info/ssspeaks/volume33/sss33.pdf',
            'Sathya Sai Speaks Volume 34 2001.pdf' : 'https://www.sssbpt.info/ssspeaks/volume34/sss34.pdf',
            'Sathya Sai Speaks Volume 35 2002.pdf' : 'https://www.sssbpt.info/ssspeaks/volume35/sss35.pdf',
            'Sathya Sai Speaks Volume 36 2003.pdf' : 'https://www.sssbpt.info/ssspeaks/volume36/sss36.pdf',
            'Sathya Sai Speaks Volume 37 2004.pdf' : 'https://www.sssbpt.info/ssspeaks/volume37/sss37.pdf',
            'Sathya Sai Speaks Volume 38 2005.pdf' : 'https://www.sssbpt.info/ssspeaks/volume38/sss38.pdf',
            'Sathya Sai Speaks Volume 39 2006.pdf' : 'https://www.sssbpt.info/ssspeaks/volume39/sss39.pdf',
            'Sathya Sai Speaks Volume 40 2007.pdf' : 'https://www.sssbpt.info/ssspeaks/volume40/sss40.pdf',
            'Sathya Sai Speaks Volume 41 2008.pdf' : 'https://www.sssbpt.info/ssspeaks/volume41/sss41.pdf',
            'Sathya Sai Speaks Volume 42 2009.pdf' : 'https://www.sssbpt.info/ssspeaks/volume42/sss42.pdf',
            'Sathya Sai Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Sathyasai/SathyaSaiVahiniInteractive.pdf',
            'Summer Roses On The Blue Mountains 1976.pdf' : 'https://www.sssbpt.info/summershowers/ss1976/ss1976.pdf',
            'Summer Showers 1990.pdf' : 'https://www.sssbpt.info/summershowers/ss1990/ss1990.pdf',
            'Summer Showers In Brindavan 1972.pdf' : 'https://www.sssbpt.info/summershowers/ss1972/ss1972.pdf',
            'Summer Showers In Brindavan 1973.pdf' : 'https://www.sssbpt.info/summershowers/ss1973/ss1973.pdf',
            'Summer Showers In Brindavan 1974 Part 1.pdf' : 'https://www.sssbpt.info/summershowers/ss1974/ss1974part1.pdf',
            'Summer Showers In Brindavan 1974 Part 2.pdf' : 'https://www.sssbpt.info/summershowers/ss1974/ss1974part2.pdf',
            'Summer Showers In Brindavan 1977.pdf' : 'https://www.sssbpt.info/summershowers/ss1977/ss1977.pdf',
            'Summer Showers In Brindavan 1978.pdf' : 'https://www.sssbpt.info/summershowers/ss1978/ss1978.pdf',
            'Summer Showers In Brindavan 1979.pdf' : ' https://www.sssbpt.info/summershowers/ss1979/ss1979.pdf',
            'Summer Showers In Brindavan 1991.pdf' : 'https://www.sssbpt.info/summershowers/ss1991/ss1991.pdf',
            'Summer Showers In Brindavan 1993.pdf' : 'https://www.sssbpt.info/summershowers/ss1993/ss1993.pdf',
            'Summer Showers In Brindavan 1995.pdf' : 'https://www.sssbpt.info/summershowers/ss1995/ss1995.pdf',
            'Summer Showers In Brindavan 1996.pdf' : 'https://www.sssbpt.info/summershowers/ss1996/ss1996.pdf',
            'Summer Showers In Brindavan 2000.pdf' : 'https://www.sssbpt.info/summershowers/ss2000/ss2000.pdf',
            'Summer Showers In Brindavan 2002.pdf' : 'https://www.sssbpt.info/summershowers/ss2002/ss2002.pdf',
            'Sutra Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Sutra/SutraVahiniInteractive.pdf',
            'Upanishad Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Upanishad/UpanishadVahiniInteractive.pdf',
            'Vidya Vahini.pdf' : 'https://www.sssbpt.info/vahinis/Vidya/VidyaVahiniInteractive.pdf',
}

    resp = [" I don't know.","I don't know.", "I'm sorry, I don't understand the question. Can you please provide more context or rephrase it?", "Sairam, How can I help you!", "I'm sorry, but I don't know the answer to your question.", "I am confused", "I do not know"
           , "I don't know because I am an AI and I do not possess the capability to know or understand such concepts.", "Yes. I am an AI."]

    if llm_response['result'] in resp :
      return ans
    sources_used = ' \n'.join([str(source.metadata['source'].split('/')[-1][:-4]) + "\tPage: " + str(source.metadata['page']) + "\nLink: " + str(src[source.metadata['source'].split('/')[-1]]) for source in llm_response['source_documents']])
    ans = ans + '\n\nSources: \n' + sources_used
    return ans

def llm_ans(query):

    llm_response = chain(query)
    ans = process_llm_response(llm_response)

    return ans

def predict(message, history):
    # output = message # debug mode

    output = str(llm_ans(message))
    return output

contribution_docstring = """
## An offering of love and gratitude by the II MSc Data Science and Computing Batch of 2023-24.
### Work done as part of the Deep Learning and Natural Language Processing Lab.
### Guided by Prakash PVSS
#### Links to Books: [Click Here](https://www.sssbpt.info/english/index.html)
"""

demo = gr.ChatInterface(predict,
                        title = f'SAI Speaks',
                        description = contribution_docstring )

if __name__ == "__main__":
    demo.launch()
