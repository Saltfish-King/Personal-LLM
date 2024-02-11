# These codes are originally designed to be run on Apple 
output_dir = 'my-docs' # Path to output directory
input_dir = 'files_used' # Path to input directory
weaviate_url = "http://localhost:8080"
embedding_model_name = 'all-MiniLM-L6-v2'
device = 'mps'

import subprocess
import os
from typing import List, Dict

def process_local(output_dir: str, num_processes: int, input_path: str):
        command = [
          "unstructured-ingest",
          "local",
          "--input-path", input_path,
          "--output-dir", output_dir,
          "--num-processes", str(num_processes),
          "--recursive",
          "--verbose",
        ]

        # Run the command
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()

        # Print output
        if process.returncode == 0:
            print('Command executed successfully. Output:')
            print(output.decode())
        else:
            print('Command failed. Error:')
            print(error.decode())

def get_result_files(folder_path) -> List[Dict]:
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

import uuid
import weaviate
from weaviate.util import get_valid_uuid

def create_local_weaviate_client(db_url: str):
    return weaviate.Client(
        url=db_url,
    )

def get_schema(vectorizer: str = "none"):
    return {
        "classes": [
            {
                "class": "Doc",
                "description": "A generic document class",
                "vectorizer": vectorizer,
                "properties": [
                    {
                        "name": "last_modified",
                        "dataType": ["text"],
                        "description": "Last modified date for the document",
                    },
                    {
                        "name": "player",
                        "dataType": ["text"],
                        "description": "Player related to the document",
                    },
                    {
                        "name": "position",
                        "dataType": ["text"],
                        "description": "Player Position related to the document",
                    },
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "Text content for the document",
                    },
                ],
            },
        ],
    }

def upload_schema(my_schema, weaviate):
    weaviate.schema.delete_all()
    weaviate.schema.create(my_schema)

def count_documents(client: weaviate.Client) -> Dict:
    response = (
        client.query
        .aggregate("Doc")
        .with_meta_count()
        .do()
    )
    count = response
    return count

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import DataSourceMetadata
from unstructured.partition.json import partition_json
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(embedding_model_name, device=device)

def compute_embedding(chunk_text: List[str]):
    embeddings = embedding_model.encode(chunk_text, device=device)
    return embeddings

def get_chunks(elements, chunk_under_n_chars=500, chunk_new_after_n_chars=1500):
    for element in elements:
        if not type(element.metadata.data_source) is DataSourceMetadata:
            delattr(element.metadata, "data_source")

        if hasattr(element.metadata, "coordinates"):
            delattr(element.metadata, "coordinates")

    chunks = chunk_by_title(
        elements,
        combine_under_n_chars=chunk_under_n_chars,
        new_after_n_chars=chunk_new_after_n_chars
    )

    for i in range(len(chunks)):
        chunks[i] = {"last_modified": chunks[i].metadata.last_modified, "text": chunks[i].text}

    chunk_texts = [x['text'] for x in chunks]
    embeddings = compute_embedding(chunk_texts)
    return chunks, embeddings


def add_data_to_weaviate(files, client, chunk_under_n_chars=500, chunk_new_after_n_chars=1500):
    for filename in files:
        try:
            elements = partition_json(filename=filename)
            chunks, embeddings = get_chunks(elements, chunk_under_n_chars, chunk_new_after_n_chars)
        except IndexError as e:
            print(e)
            continue

        print(f"Uploading {len(chunks)} chunks for {str(filename)}.")
        for i, chunk in enumerate(chunks):
            client.batch.add_data_object(
                data_object=chunk,
                class_name="doc",
                uuid=get_valid_uuid(uuid.uuid4()),
                vector=embeddings[i]
            )
        
    client.batch.flush()

from langchain.llms import LlamaCpp
from langchain.vectorstores.weaviate import Weaviate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

def question_answer(question: str, vectorstore: Weaviate):
    embedding = compute_embedding(question)
    similar_docs = vectorstore.max_marginal_relevance_search_by_vector(embedding)
    content = [x.page_content for x in similar_docs]
    prompt_template = PromptTemplate.from_template(
    """\
    Given context about the subject, answer the question based on the context provided to the best of your ability.
    Context: {context}
    Question:
    {question}
    Answer:
    """
    )
    prompt = prompt_template.format(context=content, question=question)
    answer = llm(prompt)
    return answer, similar_docs

if __name__ == "__main__":
    process_local(output_dir=output_dir, num_processes=2, input_path=input_dir)
    files = get_result_files(output_dir)

    client = create_local_weaviate_client(db_url=weaviate_url)
    my_schema = get_schema()
    upload_schema(my_schema, weaviate=client)
    
    add_data_to_weaviate(
        files=files,
        client=client,
        chunk_under_n_chars=250,
        chunk_new_after_n_chars=500
    )

    print(count_documents(client=client)['data']['Aggregate']['Doc'])

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 1  # Metal set to 1 is enough.
    n_batch = 100  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path="model_files/llama-2-7b-chat.Q4_K_S.gguf",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=2048, # context window. By default 512
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True, # Verbose is required to pass to the callback manager
    )

    client = weaviate.Client(weaviate_url)
    vectorstore = Weaviate(client, "Doc", "text")

    question = "Give a summary of NFL Draft 2020 Scouting Reports: RB Jonathan Taylor, Wisconsin?"

    answer, similar_docs = question_answer(question, vectorstore)

    print("\n\n\n-------------------------")
    print(f"QUERY: {question}")
    print("\n\n\n-------------------------")
    print(f"Answer: {answer}")
    print("\n\n\n-------------------------")
    for index, result in enumerate(similar_docs):
        print(f"\n\n-- RESULT {index+1}:\n")
        print(result)