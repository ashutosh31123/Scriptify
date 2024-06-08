from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun

#Function to generate video script
def generate_script(prompt,video_length,creativity,api_key):
    
    #template for generating "Title"
    title_template= PromptTemplate(
        input_variables = ['subject'],
        template='Please come up with a title for a YouTube Video on {subject}.'
    )
    
    #Template for generating "Video Script" using search engine
    script_template=PromptTemplate(
        input_variables = ['title', 'DuckDuckGo_Search','duration'], 
        template='Create a script for a YouTube video based on this title for me. TITLE: {title} of duration: {duration} minutes using this search data {DuckDuckGo_Search} '
    )
    
    #setting up OpenAI LLM
    llm= ChatOpenAI(temperature=creativity,openai_api_key=api_key,model_name='gpt-3.5-turbo')
    
    #Creating chain for "Title and Video Script"
    title_chain=LLMChain(llm=llm,prompt=title_template,verbose=True)
    script_chain=LLMChain(llm=llm,prompt=script_template,verbose=True)
    
    search=DuckDuckGoSearchRun()
    
    #Executing the chains we created for 'Title'
    title=title_chain.invoke(prompt)
    
    #Executing the chains we created for "Video Script" by taking help of search engine "DuckDuckGo"
    search_result=search.run(prompt)
    script=script_chain.run(title=title,DuckDuckGo_Search=search_result,duration=video_length)
    
    #Returning the output
    return search_result,title['text'] if 'text' in title else title,script