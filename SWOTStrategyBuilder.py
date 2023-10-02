import os
import streamlit as st
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

# Fetch API keys from environment variables
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['CODEBOX_API_KEY'] = st.secrets['CODEBOX_API_KEY']

import semantic_kernel as sk

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from IPython.display import display, Markdown

kernel = sk.Kernel()

useAzureOpenAI = False

if useAzureOpenAI:
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureopenai", AzureChatCompletion(deployment, endpoint, api_key))
else:
    api_key = os.getenv('OPEN_AI_KEY')
    kernel.add_text_completion_service("openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key))
    
swot_interview= """
1. **Strengths**
    - What unique recipes or ingredients does the pizza shop use?
    - What are the skills and experience of the staff?
    - Does the pizza shop have a strong reputation in the local area?
    - Are there any unique features of the shop or its location that attract customers?
2. **Weaknesses**
    - What are the operational challenges of the pizza shop? (e.g., slow service, high staff turnover)
    - Are there financial constraints that limit growth or improvements?
    - Are there any gaps in the product offering?
    - Are there customer complaints or negative reviews that need to be addressed?
3. **Opportunities**
    - Is there potential for new products or services (e.g., catering, delivery)?
    - Are there under-served customer segments or market areas?
    - Can new technologies or systems enhance the business operations?
    - Are there partnerships or local events that can be leveraged for marketing?
4. **Threats**
    - Who are the major competitors and what are they offering?
    - Are there potential negative impacts due to changes in the local area (e.g., construction, closure of nearby businesses)?
    - Are there economic or industry trends that could impact the business negatively (e.g., increased ingredient costs)?
    - Is there any risk due to changes in regulations or legislation (e.g., health and safety, employment)?"""


sk_prompt = """
{{$input}}

Convert the analysis provided above to the business domain of {{$domain}} (ALWAYS in the form of SWOT questions).
"""
shift_domain_function = kernel.create_semantic_function(prompt_template=sk_prompt,
                                                    description="Translate an idea to another domain.",
                                                    max_tokens=1000,
                                                    temperature=0.1,
                                                    top_p=0.5)
my_context = kernel.create_new_context()

my_context['input'] = swot_interview
my_context['domain'] = "construction management"

async def exquisite_function():
    result = await kernel.run_async(shift_domain_function, input_context=my_context)
    return result

async def run_kernel_async(shift_domain_function, my_context):
    result_domain_shift = await kernel.run_async(shift_domain_function, input_context=my_context)
    return result_domain_shift

def run_asyncio_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(run_kernel_async(shift_domain_function, my_context))

st.title("SWOT Strategy Builder")
domain = st.text_input("Enter the business domain (e.g., construction management):", "")

# Update the Semantic Kernel context based on user input
my_context['domain'] = domain

llm = ChatOpenAI(temperature=0.1)  # Similar to OpenAIChatCompletion setup in Semantic Kernel.

user_input_swot = ''

first_prompt = ChatPromptTemplate.from_template("""Every business benefits from four courses of action to be used together or in combination:

1. Grow the existing business
2. Save money and time
3. Add completely new business
4. Prepare for the unknown

Growth as built upon a business' existing strengths is always a great place to start. And by finding any kinds of savings, those benefits can either be banked or reinvested. Sometimes a new line of business can be pursued with successful growth or savings initiatives. And as we all learned from the pandemic era, there's a certain benefit to businesses that are resilient to unexpected shifts that sit out their control. This leads us to a simplified way of thinking how best to tackle one's business challenges:

* Strengths: Build upon them and grow
* Weaknesses: Don't damage the core
* Opportunities: Make a few bets
* Threats: Be ready for undesirable changes

Help a business in the domain of \"""" + domain + """.

Address the following specific answers from the user: \"""" + user_input_swot + """. For more unstructured answers from users (such as in a paragraph that doesn't explicitly state whether the user is talking about 'strengths' or 'weaknesses' for example), attempt to extract the meaning in relation to the below formats.

Add a markdown ### heading entitled "Building on strengths can immediately improve the business"

Define at least four ways in which the business can improve.
The format should read in markdown format as:
| Title | Strength | Description | Example |
| ----- | -------- | ----------- | ------- |
| concise name of the business strategy | strength that is being built upon | a short description of the strategy | example |
| concise name of the business strategy | strength that is being built upon | a short description of the strategy | example |
| concise name of the business strategy | strength that is being built upon | a short description of the strategy | example |
| concise name of the business strategy | strength that is being built upon | a short description of the strategy | example |

Before the next section add a markdown ### heading entitled "Addressing weaknesses to strengthen the core business"

Define at least four ways in which the business can mitigate weaknesses.
The format should read in markdown format as:
| Title | Weakness | Description | Example |
| ----- | -------- | ----------- | ------- |
| concise name of the business strategy | weakness that is being addressed | a short description of the strategy | example |
| concise name of the business strategy | weakness that is being addressed | a short description of the strategy | example |
| concise name of the business strategy | weakness that is being addressed | a short description of the strategy | example |
| concise name of the business strategy | weakness that is being addressed | a short description of the strategy | example |

Before the next section add a markdown ### heading entitled "Taking advantage of opportunities whenever possible"

Define at least four ways in which the business can grow new revenue.
The format should read in markdown format as:
| Title | Opportunity | Description | Example |
| ----- | ----------- | ----------- | ------- |
| concise name of the growth strategy | opportunity that is being built upon | a short description of the strategy | example |
| concise name of the growth strategy | opportunity that is being built upon | a short description of the strategy | example |
| concise name of the growth strategy | opportunity that is being built upon | a short description of the strategy | example |
| concise name of the growth strategy | opportunity that is being built upon | a short description of the strategy | example |

Before the next section add a markdown ### heading entitled "Building resilience to threats is always a good idea"

Define at least four ways in which the business can become more resilient to threats.
The format should read in markdown format as:
| Title | Threat | Description | Example |
| ----- | ------ | ----------- | ------- |
| concise name of the resilience strategy | threat that is being addressed | a short description of the strategy | example |
| concise name of the resilience strategy | threat that is being addressed | a short description of the strategy | example |
| concise name of the resilience strategy | threat that is being addressed | a short description of the strategy | example |
| concise name of the resilience strategy | threat that is being addressed | a short description of the strategy | example |
""")

chain_one = LLMChain(llm=llm, prompt=first_prompt)

result_domain_shift = None  # or ''

if 'show_text_area' not in st.session_state:
    st.session_state.show_text_area = False

if st.button("Create Questions"):
    result_domain_shift = run_asyncio_loop()
    
    def display_results():    
        st.markdown(f"### ✨ Shifted SWOT Interview Questions to {domain}")
        st.markdown(str(result_domain_shift))
    
    display_results() # Call the function here to display the results
    st.session_state.show_text_area = True  # Update the state variable
    
    # Langchain execution and new text area for SWOT questions goes here
if st.session_state.show_text_area:
    user_input_swot = st.text_area("Please provide your answers to the SWOT questions:")
    if st.button("Submit Answers"):    
        input_dict = {'input': result_domain_shift}
        result_chain_one = chain_one.run(input_dict)
        st.markdown(f"### ✨ Business strategy thinking based on SWOT analysis")
        st.markdown(str(result_chain_one))