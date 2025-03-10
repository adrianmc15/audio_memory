import os
from groq import Groq

def read_transcript(file_path):
    """
    Reads the transcript from the given file path.
    
    :param file_path: Path to the transcript file.
    :return: Content of the transcript as a string.
    """
    with open(file_path, 'r') as file:
        transcript = file.read()
    return transcript

def query_llm_with_groq(api_key, prompt):
    """
    Queries the Groq LLM with the given prompt.
    
    :param api_key: Your Groq API key.
    :param prompt: The prompt to send to the LLM.
    :return: The response from the LLM.
    """
    # initialize the Groq client
    client = Groq(api_key=api_key)
    
    # send the prompt to the LLM
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant, tasked to answer questions about the transcript provided to you. When asked questions about the transcript, do not quote it, just explain the answer. If you do not know the answer based on the transcript, explain that you do not know, do not make something up."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.3-70b-versatile"  # Replace with the correct model name
    )
    
    # extract and return the response
    return chat_completion.choices[0].message.content

def main(file_path, api_key):
    """
    Main function to read the transcript and interactively query the LLM.
    
    :param file_path: Path to the transcript file.
    :param api_key: Your Groq API key.
    """
    # read the transcript
    transcript = read_transcript(file_path)
    print("Transcript loaded successfully.")
    
    while True:
        # get user question
        question = input("\nEnter your question about the transcript (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Exiting...")
            break
        
        # prep the prompt for the LLM
        prompt = f"The following is a transcript:\n\n{transcript}\n\nQuestion: {question}\n\nAnswer:"
        
        try:
            # query the LLM
            answer = query_llm_with_groq(api_key, prompt)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    # replace with your actual file path and Groq API key
    file_path = "data/output/transcript.txt"
    api_key = "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"
    
    if not api_key:
        raise ValueError("Please set the GROQ_API_KEY environment variable.")
    
    main(file_path, api_key)