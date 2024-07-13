import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, Tool, AgentExecutor
from langchain import hub
from chains.hospital_review_chain import reviews_vector_chain
from chains.hospital_cypher_chain import hospital_cypher_chain
from tools.wait_times import (
    get_current_wait_times,
    get_most_available_hospital,
)

HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL")

hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [
    Tool(
        name="Experiences",
        func=reviews_vector_chain.invoke,
        description=""" Useful when you need to answer questions about patient experiences,
        feelings, or any other qualitative question that could be answered
        about a patient using semantic search. Not useful for answering
        objective questions that involve counting, percentages, aggregations,
        or listing facts. Use the entire prompt as input to the tool. For instance,
        if the prompt is "Are patients satisfied with their care?", the input should
        be "Are patients satisfied with their care?".

        This tool leverages semantic search capabilities to find relevant patient
        reviews and experiences from a large corpus of data. It can provide insights
        into patient sentiments, common complaints, positive feedback, and other
        qualitative aspects of their hospital visits.

        Example Usage:
        - "What do patients say about the nursing staff at Mercy Hospital?"
        - "How do patients feel about the emergency services at General Hospital?"

        Example Response:
        {
            "question": "What do patients say about the nursing staff at Mercy Hospital?",
            "response": "Patients generally report positive experiences with the nursing staff at Mercy Hospital. Common feedback includes comments on their attentiveness, kindness, and professionalism. However, some patients noted that during peak hours, the response times can be slower."
        }

        In this example, the response provides a summarized view of patient feedback regarding the nursing staff at Mercy Hospital, highlighting both positive and negative aspects.

        Additional Notes:
        - This tool excels at capturing the nuances and sentiments in patient feedback, providing a more humanized view of patient experiences.
        - It is ideal for qualitative analysis and understanding the broader context of patient reviews.
        - Ensure that the prompt accurately reflects the qualitative nature of the question to get the most relevant insights.
        """,
    ),
    Tool(
        name="Graph",
        func=hospital_cypher_chain.invoke,
        description="""Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?".
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_times,
        description="""Use when asked about current wait times at a specific hospital. This tool can
        only get the current wait time at a hospital and does not have any information
        about aggregate or historical wait times. Do not pass the word "hospital" as
        input, only the hospital name itself. For example, if the prompt is "What is the
        current wait time at Jordan Inc Hospital?", the input should be "Jordan Inc".

        This tool provides real-time data on the wait times at individual hospitals,
        helping patients to plan their visits more efficiently. It fetches the latest
        available data, ensuring that the information is up-to-date and accurate.

        Example Usage:
        - "What is the current wait time at Mercy Medical Center?"
        - "How long will I have to wait at General Hospital?"

        Example Response:
        {
            "hospital_name": "Mercy Medical Center",
            "current_wait_time": 25
        }

        In this example, the response indicates that the current wait time at Mercy Medical
        Center is 25 minutes. This information allows patients to make timely decisions
        about where to seek medical care based on the most recent wait time data.
        
        Additional Notes:
        - Ensure the hospital name is correctly spelled to get accurate wait time information.
        - This tool is designed to provide instantaneous data and does not support queries
        related to past or future wait times.
        - The wait time is typically measured in minutes and reflects the average time a 
        patient might wait to receive medical attention upon arrival..
            """,
    ),
    Tool(
        name="Availability",
        func=get_most_available_hospital,
        description="""
        Use when you need to find out which hospital has the shortest
        wait time. This tool does not have any information about aggregate
        or historical wait times. This tool returns a dictionary with the
        hospital name as the key and the wait time in minutes as the value.
        This information is useful for patients needing immediate care and 
        looking to minimize their waiting period. The tool fetches real-time
        data and is updated frequently to reflect the current status at each
        hospital. 

        Example Usage:
        - "Which hospital has the shortest wait time right now?"
        - "Find the hospital with the quickest availability for immediate care."

        Example Response:
        {
            "General Hospital": 15,
            "Mercy Hospital": 20,
            "City Clinic": 10
        }

        In this example, the hospital with the shortest wait time is "City Clinic"
        with a wait time of 10 minutes. This allows patients to make informed 
        decisions based on the latest available data, potentially reducing their 
        waiting period and receiving faster medical attention.
        """,
    ),
]

chat_model = ChatOpenAI(
    model=HOSPITAL_AGENT_MODEL,
    temperature=0,
)

hospital_rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)

hospital_rag_agent_executor = AgentExecutor(
    agent=hospital_rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
