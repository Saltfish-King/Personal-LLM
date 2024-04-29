# This script uses Weaviate Python Client V4

import subprocess
import os
from typing import List, Dict
import uuid
import weaviate
import weaviate.classes.config as wvcc
from weaviate.util import get_valid_uuid
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import DataSourceMetadata
from unstructured.partition.json import partition_json
from sentence_transformers import SentenceTransformer
from langchain_community.llms import LlamaCpp
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

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

def create_local_weaviate_client(hostname: str):
    return weaviate.connect_to_local(host=hostname) #Connect to a local Weaviate instance deployed using Docker compose with standard port configurations.

def delete_all_collections(weaviate):
    weaviate.collections.delete_all()

def create_collection(weaviate, vectorizer=None):
    # in weaviate client v4, the schema API was removed in favor of the Collections API.
    weaviate.collections.create(
        name="Doc", # a generic document class
        # vectorizer_config = vectorizer, None for now
        properties=[
            wvcc.Property(
                name="last_modified",
                data_type=wvcc.DataType.TEXT
                # description": "Last modified date for the document
            ),
            wvcc.Property(
                name="text",
                data_type=wvcc.DataType.TEXT
                # description": "Text content for the document
            ),
        ]
    )

def count_documents(client: weaviate.Client) -> Dict:
    jeopardy = client.collections.get("Doc")
    count = jeopardy.aggregate.over_all(total_count=True)
    return count

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

def get_target_collection(collection_name):
    return client.collections.get(collection_name)

def add_data_to_weaviate(files, collection, chunk_under_n_chars=500, chunk_new_after_n_chars=1500):
    for filename in files:
        try:
            elements = partition_json(filename=filename)
            chunks, embeddings = get_chunks(elements, chunk_under_n_chars, chunk_new_after_n_chars)
        except IndexError as e:
            print(e)
            continue

        print(f"Uploading {len(chunks)} chunks for {str(filename)}.")
        for i, chunk in enumerate(chunks):
            with collection.batch.dynamic() as batch:
                batch.add_object(
                    properties=chunk,
                    uuid=get_valid_uuid(uuid.uuid4()),
                    vector=embeddings[i]
            )
        
    # client.batch.flush()

def hybrid_search(query: str, collection: str, limit: int):
    # This is the inherent hybrid search method of Weaviate
    jeopardy = client.collections.get(collection)
    embedding = compute_embedding(query)
    response = jeopardy.query.hybrid(
        query=query,
        vector= embedding.tolist(), #use tolist() to convert a numpy.ndarray object to list 
        limit=limit
    )
    return response

def question_answer(question: str, vectorstore: WeaviateVectorStore = None):
    # similar_docs = vectorstore.max_marginal_relevance_search_by_vector(embedding)
    # content = [x.page_content for x in similar_docs]
    similar_docs = hybrid_search(question, "Doc", 3)
    
    content = [x.properties['text'] for x in similar_docs.objects]
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
    return answer, content

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', default="files_used", type=str, help='The file source directory')
    parser.add_argument('--output', default="my-docs", type=str, help='The embedded files')
    parser.add_argument('--embedding_model_name', default="all-MiniLM-L6-v2", type=str, help='embedding model')
    parser.add_argument('--device', default="cuda", type=str, help='device to use')
    parser.add_argument('--question', default="Give a summary of NFL Draft 2020 Scouting Reports: RB Jonathan Taylor, Wisconsin?", type=str, help='a default question')
    parser.add_argument('--model_path', default="model_files/llama-2-7b-chat.Q4_K_S.gguf", type=str, help='path to LLM model')
    args = parser.parse_args()

    output_dir = args.output
    input_dir = args.input
    embedding_model_name = args.embedding_model_name
    device = args.device
    question = args.question

    process_local(output_dir=output_dir, num_processes=2, input_path=input_dir)
    files = get_result_files(output_dir)

    # weaviate_url = "http://yuzhou-weaviate-1:8080"
    client = create_local_weaviate_client("yuzhou-weaviate-1")
    delete_all_collections(weaviate=client)
    create_collection(weaviate=client)

    embedding_model = SentenceTransformer(embedding_model_name, device=device)

    tgt_collection = get_target_collection("Doc")

    add_data_to_weaviate(
        files=files,
        collection=tgt_collection,
        chunk_under_n_chars=250,
        chunk_new_after_n_chars=500,
    )

    print("file number count: {}".format(count_documents(client=client).total_count))

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 1  # set to 1 as default
    n_batch = 100  # Should be between 1 and n_ctx, consider the RAM amount
    # Make sure the model path is correct for the system!
    llm = LlamaCpp(
        model_path=args.model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=2048, # context window. By default 512
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True, # Verbose is required to pass to the callback manager
    )

    # client = weaviate.Client(weaviate_url)
    # langchain supports weaviate python client 4.0 using langchain_weaviate library
    # But it seems like it is not necessary to use lang-chain since in this prototype, since it is only carrying out marginal_search
    # vectorstore = WeaviateVectorStore(client, "Doc", "text")

    while True:        
        user_input = input("Enter your question here (type 'quit' to exit): ")
        if user_input == "quit":
            break
        if user_input != '':
            question = user_input        
        answer, similar_docs = question_answer(question)
        print("\n\n\n-------------------------")
        print(f"QUERY: {question}")
        print("\n\n\n-------------------------")
        print(f"Answer: {answer}")
        print("\n\n\n-------------------------")
        for index, result in enumerate(similar_docs):
            print(f"\n\n-- RESULT {index+1}:\n")
            print(result)

