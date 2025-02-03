import pandas as pd
import openai
import os
import glob

def generate_question_and_answer(text_chunk, client, model_name="llama3.2:latest"):
    # Define the question prompt
    question_prompt = f"You are a Professor writing an exam. Using the provided context: '{text_chunk}', formulate a single question that captures an important fact or insight from the context, e.g. 'Who was Aristophanes?' or 'What are latifundia?' or 'What is ostracism?' or 'Where did Xerxes cross the Hellespont?' or 'When did the battle of Platea occur?' or 'Why did Christianity appeal to slaves?' or 'How did Athens stop class warfare during the Periclean age?'. Restrict the question to the context information provided."

    # Generate a question unconditionally
    question_response = client.completions.create(model=model_name, prompt=question_prompt, max_tokens=100)
    question = question_response.choices[0].text.strip()
    
    # Generate an answer unconditionally
    answer_prompt = f"Given the context: '{text_chunk}', give a detailed, complete answer to the question: '{question}'. Use only the context to answer, do not give references. Simply answer the question without editorial comments."
    answer_response = client.completions.create(model=model_name, prompt=answer_prompt, max_tokens=350)
    answer = answer_response.choices[0].text.strip()

    return question, answer

# Point to the local server
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

# Directory containing text files
directory_path = "./input"

# List to store Q&A pairs
qa_data = []

# Iterate over each file in the directory
for file_path in glob.glob(os.path.join(directory_path, '*.txt')):
    print(file_path)
    with open(file_path, 'r') as file:
        text_chunk = file.read()

    # Generate question and answer
    question, answer = generate_question_and_answer(text_chunk, client)

    # Append the generated Q&A to the list
    qa_data.append({"Context": text_chunk, "Question": question, "Answer": answer})

qa_df = pd.DataFrame(qa_data)

qa_df.to_csv("./output/dataset.csv", index=False)