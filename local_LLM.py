# This script uses Weaviate Python Client V4

import subprocess
import os
from typing import List, Dict
import uuid
import weaviate
import weaviate.classes.config as wvcc
from weaviate.classes.config import Configure, VectorDistances
from weaviate.classes.query import MetadataQuery
from weaviate.util import get_valid_uuid
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import DataSourceMetadata
from unstructured.partition.json import partition_json
from sentence_transformers import SentenceTransformer, CrossEncoder
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

def create_collection(weaviate, dataset, vectorizer=None):
    # in weaviate client v4, the schema API was removed in favor of the Collections API.
    weaviate.collections.create(
        name=dataset, # a generic document class
        # vectorizer_config = vectorizer, None for now
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE
        ),
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

def count_documents(client: weaviate.Client, dataset) -> Dict:
    jeopardy = client.collections.get(dataset)
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
        return_metadata=MetadataQuery(score=True, explain_score=True),
        limit=limit
    )
    return response


def rerank_contents(similar_doc, query, rerank_model_name):
    # reranking use CrossEncoder
    model = CrossEncoder(rerank_model_name, max_length=3000)
    context_list = []
    for obj in similar_doc.objects:
        context_list.append([query, obj.properties['text']])

    scores = model.predict(context_list)
    sorted_pairs = sorted(zip(scores, context_list), reverse=True)
    sorted_scores, sorted_context = zip(*sorted_pairs) # unzip the pairs
    return sorted_context[:10]


def question_answer(question: str, similar_context, vectorstore: WeaviateVectorStore = None):
    prompt_template = PromptTemplate.from_template(
    """\
    Given context about the subject, answer the question based on the context provided to the best of your ability. 
    Be aware that there could be unnecessary context in the provided context, select only the relavent and correct ones.
    Context: {context}
    Question:
    {question}
    Answer:
    """
    )
    prompt = prompt_template.format(context=similar_context, question=question)
    answer = llm(prompt)
    return answer, similar_context

def rephrase_question(question: str):
    prompt_template = PromptTemplate.from_template(
    """{question}\n
    Rephrase and expand the question, and respond.
    """
    )
    '''
    'other variations of the prompt remain within our methodology and also provide improvement in performance.
    Such prompts include but not limited to the following:
        • Reword and elaborate on the inquiry, then provide an answer.
        • Reframe the question with additional context and detail, then provide an answer.
        • Modify the original question for clarity and detail, then offer an answer.
        • Restate and elaborate on the inquiry before proceeding with a response.
    '''
    prompt = prompt_template.format(question=question)
    answer = llm(prompt)
    return answer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', default="files_used", type=str, help='The file source directory')
    parser.add_argument('--output', default="my-docs", type=str, help='The embedded files')
    parser.add_argument('--embedding_model_name', default="all-MiniLM-L6-v2", type=str, help='embedding model')
    parser.add_argument('--device', default="cuda", type=str, help='device to use')
    parser.add_argument('--question', default="Give a summary of NFL Draft 2020 Scouting Reports: RB Jonathan Taylor, Wisconsin?", type=str, help='a default question')
    parser.add_argument('--model_path', default="model_files/llama-2-7b-chat.Q4_K_S.gguf", type=str, help='path to LLM model')
    parser.add_argument('--rar', default=False, type=bool, help='whether to use Respose and Rephrase')
    parser.add_argument('--upload_data', default=False, type=bool, help='whether to upload the designated dataset')
    parser.add_argument('--clean_up_database', default=False, type=bool, help='whether to delete all contents in database')
    parser.add_argument('--dataset_name', default="Doc", type=str, help='dataset used in this process')
    parser.add_argument('--combine_under_n_chars', default=250, type=int, help='object character length')
    parser.add_argument('--new_after_n_chars', default=500, type=int, help='cut off new section once meet this length')
    parser.add_argument('--rerank', default=False, type=bool, help='whether to use rerank module or not')
    parser.add_argument('--rerank_model_name', default="cross-encoder/ms-marco-MiniLM-L-6-v2", type=str)
    parser.add_argument('--context_window', default=16348, type=int, help='context window')

    args = parser.parse_args()

    output_dir = args.output
    input_dir = args.input
    embedding_model_name = args.embedding_model_name
    device = args.device
    question = args.question
    use_rar = args.rar
    upload_data = args.upload_data
    delete_all = args.clean_up_database
    collection = args.dataset_name
    chunk_under_n_chars = args.combine_under_n_chars
    chunk_new_after_n_chars = args.new_after_n_chars
    rerank = args.rerank
    rerank_model_name = args.rerank_model_name
    ctx_window = args.context_window

    process_local(output_dir=output_dir, num_processes=2, input_path=input_dir)
    files = get_result_files(output_dir)

    # weaviate_url = "http://yuzhou-weaviate-1:8080"
    client = create_local_weaviate_client("yuzhou-weaviate-1")
    embedding_model = SentenceTransformer(embedding_model_name, device=device)

    if delete_all:
        delete_all_collections(weaviate=client)
    
    if upload_data:
        create_collection(dataset=collection, weaviate=client)
        tgt_collection = get_target_collection(collection)
        add_data_to_weaviate(
            files=files,
            collection=tgt_collection,
            chunk_under_n_chars=chunk_under_n_chars,
            chunk_new_after_n_chars=chunk_new_after_n_chars,
        )
        print("file number count: {}".format(count_documents(client=client, dataset=collection).total_count))

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 1  # set to 1 as default
    n_batch = 100  # Should be between 1 and n_ctx, consider the RAM amount
    # Make sure the model path is correct for the system!
    llm = LlamaCpp(
        model_path=args.model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=ctx_window, # context window. By default 16348
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
        if use_rar:
            print("Question (before rephrase): ", question)
            question = rephrase_question(question)    
        

        # similar_docs = vectorstore.max_marginal_relevance_search_by_vector(embedding)
        # content = [x.page_content for x in similar_docs]

        if rerank:
            similar_docs = hybrid_search(question, collection, 100)
            original_order_content = [x.properties['text'] for x in similar_docs.objects]
            content = rerank_contents(similar_docs, question, rerank_model_name)
        else:
            similar_docs = hybrid_search(question, collection, 20)
            content = [x.properties['text'] for x in similar_docs.objects]

        # for o in similar_docs.objects:
        #     print(o.properties)
        #     print(o.metadata.score, o.metadata.explain_score)

        answer, similar_docs = question_answer(question, content)
        print("\n\n\n-------------------------")
        if use_rar:
            print(f"Question (after rephrase):\n {question}")
        else:
            print(f"QUERY: {question}")
        print("\n\n\n-------------------------")
        print(f"Answer: {answer}")
        print("\n\n\n-------------------------")
        for index, result in enumerate(similar_docs):
            print(f"\n\n-- RESULT {index+1}:\n")
            print(result)

