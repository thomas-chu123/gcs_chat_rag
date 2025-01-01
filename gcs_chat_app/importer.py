# import all the file form wedding_list folder and emdbeding the file to database.
import os
import base64
import uuid
import vertexai
from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAIImageCaptioning

from dotenv import load_dotenv

output_path = os.getcwd() + "/img_output"

def clean_output_path():
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return "data:image/jpg;base64," + base64.b64encode(f.read()).decode('utf-8')

def parser_path_to_text(file_path):
    #H Group
    #H Region
    #H Country
    #H Name
    hotel_info = file_path.split("wedding_poc/")[1]
    hotel_dict = {}
    hotel_list = hotel_info.split("/")
    if len(hotel_list) > 4:
        hotel_dict["group"] = hotel_list[0]
        hotel_dict["region"] = hotel_list[1]
        hotel_dict["country"] = hotel_list[2]
        hotel_dict["name"] = hotel_list[-2]
    if len(hotel_list) == 4:
        hotel_dict["group"] = hotel_list[0]
        hotel_dict["region"] = hotel_list[1]
        hotel_dict["country"] = hotel_list[2]
        hotel_dict["name"] = hotel_list[3]
    elif len(hotel_list) == 3:
        hotel_dict["group"] = hotel_list[0]
        hotel_dict["country"] = hotel_list[1]
        hotel_dict["name"] = hotel_list[2]
    return hotel_dict


def summarize_image(encoded_image):
    prompt = [
        SystemMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {
                "type": "text",
                "text": "Describe the contents of this image."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    response = VertexAIImageCaptioning().invoke(input=encoded_image)
    return response

def extract_image(file_name):
    # Open the PDF file
    reader = PdfReader(file_name)
    for page in reader.pages:
        for image in page.images:
            with open(output_path + "/" + image.name, "wb") as fp:
                fp.write(image.data)
    print(f"All images extracted to {output_path}")

def parser_pdf_image(file_name):
    text_elements = []
    table_elements = []

    text_summaries = []
    table_summaries = []

    try:
        # summary_prompt = """
        # Summarize the following {element_type}:
        # {element}
        # """
        # summary_chain = LLMChain(
        #     llm=ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024),
        #     prompt=PromptTemplate.from_template(summary_prompt)
        # )

        # Get elements
        # raw_pdf_elements = partition_pdf(
        #     filename=file_name,
        #     extract_images_in_pdf=True,
        #     infer_table_structure=True,
        #     chunking_strategy="by_title",
        #     max_characters=4000,
        #     new_after_n_chars=3800,
        #     combine_text_under_n_chars=2000,
        #     extract_image_block_output_dir=output_path,
        # )
        # raw_pdf_elements = partition_pdf(filename=file_name,
        #                                  extract_images_in_pdf=True,
        #                                  extract_image_block_output_dir=output_path)

        extract_image(file_name)

        # for elem in raw_pdf_elements:
        #     if 'CompositeElement' in repr(elem):
        #         text_elements.append(elem.text)
        #         summary = summary_chain.run({'element_type': 'text', 'element': e})
        #         text_summaries.append(summary)
        #
        #     elif 'Table' in repr(elem):
        #         table_elements.append(elem.text)
        #         summary = summary_chain.run({'element_type': 'table', 'element': e})
        #         table_summaries.append(summary)

        # Get image summaries
        image_elements = []
        image_summaries = []
        documents = []
        retrieve_contents = []

        for img in sorted(os.listdir(output_path),key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)):
            if img.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(output_path, img)
                encoded_image = encode_image(image_path)
                image_elements.append(encoded_image)
                summary = summarize_image(encoded_image)
                image_summaries.append(summary)
                print(f"Image summary with {image_path}, caption: {summary}")

        for elem, summary in zip(image_elements, image_summaries):
            uuid_index = str(uuid.uuid4())
            doc = Document(
                page_content=summary,
                metadata={
                    'id': uuid_index,
                    'type': 'image',
                    'original_content': elem,
                    'source': file_name,
                    'page': 0
                }
            )
            retrieve_contents.append((uuid_index, summary))
            documents.append(doc)
        return documents

    except Exception as e:
        print(f"Error in parser_pdf_image {file_name}")
        print(e)
        return []


def main():
    env = load_dotenv()

    PROJECT_ID = "ctg-rag-model-001"
    LOCATION = "us-central1"
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    if os.environ.get("stage") == 'dev':
        CONNECTION_STRING = "postgresql+psycopg2://user:password@127.0.0.1:5432/vector-db"
    else:
        CONNECTION_STRING = "postgresql+psycopg2://postgres:P{vLX90{]{Q39$ZA@10.128.128.3:5432/vector-db"
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    file_folder = os.getcwd() + "/wedding_poc/"
    for root, dirs, files in os.walk(file_folder):
        for file in files:
            if file.endswith(".pdf"):
                try:
                    print(f"Parser with {os.path.join(root,file)}")
                    clean_output_path()
                    metadata = parser_path_to_text(os.path.join(root,file))
                    try:
                        documents = PyPDFLoader(file_path=os.path.join(root,file),extract_images=True).load()
                        # docs = PdfReader(os.path.join(root,file), ).pages
                        # for i, page in enumerate(docs):
                        #     texts = page.extract_text()
                        #     for image in page.images:
                        #         with open(output_path + "/" + image.name, "wb") as fp:
                        #             fp.write(image.data)
                    except Exception as e:
                        if "/Filter" in repr(e):
                            documents = PyPDFLoader(file_path=os.path.join(root,file)).load()
                        else:
                            print(f"Error in loading PDF {os.path.join(root,file)}")
                            print(e)
                            continue
                    # image_documents = parser_pdf_image(os.path.join(root,file))
                    # documents.extend(image_documents)

                    text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=80)
                    texts = text_splitter.split_documents(documents)
                    for doc in texts:
                        doc.metadata.update(metadata)

                    COLLECTION_NAME = 'test_collection'
                    db = PGVector.from_documents(
                        embedding=embeddings,
                        documents=texts,
                        connection_string=CONNECTION_STRING,
                        collection_name=COLLECTION_NAME,
                    )
                    print(f"Embedding is done with {os.path.join(root,file)}")
                except Exception as e:
                    print(f"Error in embedding {os.path.join(root,file)}")
                    print(e)


if __name__ == "__main__":
    main()