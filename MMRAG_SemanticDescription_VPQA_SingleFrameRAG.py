#####################################################################################################################################################
########################### MultiModal Semantic Descriptions Generation with A-Prompt GPT4o for Video Input##########################################
#####################################################################################################################################################
import os
import io
import cv2
import base64
import uuid
import argparse
import warnings
import time
import hashlib
import chromadb
import langchain
import numpy as np
import pandas as pd
import sys
import glob
import shutil
import re
from fpdf import FPDF
from PIL import Image
from io import BytesIO
from typing import List
from pathlib import Path
from schedule import Scheduler
from dotenv import load_dotenv
from collections import defaultdict
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from components.utils import AzureOpenAIEmbeddingFunc
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import UnstructuredFileLoader, PyMuPDFLoader, PyPDFLoader
from sklearn.metrics.pairwise import cosine_similarity
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from chromadb.config import Settings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from datasets import Dataset
from operator import itemgetter
from IPython.display import HTML, display
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langfuse.callback import CallbackHandler
from ragas.testset import TestsetGenerator
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (faithfulness, answer_relevancy, multimodal_faithness, multimodal_relevance, context_recall, context_precision, answer_correctness, context_entity_recall, answer_similarity)
from ragas.run_config import RunConfig  

print('##################################################################################')
#####################################################################################################################################
###################################################USER VARIABLES####################################################################
video_folder = "inputVideos"
output_folder = "outputDescriptions"
FILE_savePath = "saveVideos4oneWeek"
script_directory = "I:/Z_AI4DCstaff/999_personalFolders/Aly/MasterThesis/10_VPQA_Evaluation_Pipeline/VPQA_Evaluation"

extractedFrames_folder = "tmp_extractedFrames"                # where extracted frames are saved
frame_interval = 6                                            # frame_interval = 1 means that each frame will be considered, 1 equals to 30 fbs in case of using iphone camera
                                                              # for 1 frame per second, frame_interval should be 30 in that case
                                                               
frameDescriptions_text_folder = "docs_2_crawl/FrameDescriptionText/"
audioTranscript_text_folder = "docs_2_crawl/AudioTranscriptText/"
# frame_descriptions_db_path = 'chroma_mmtext_db_A-Prompt-Agent_HW-Troubleshooting_1fps_99th'
# audio_db_path = 'chroma_audio_db_HW-Troubleshooting'
VectorEmbeddings_similarity_threshold = 0.99        
                                                       
###############################################################################################################################
####################################################AGENT DEFINITIONS##########################################################

# Prepare the input prompt Instructions/definition for Llama3.2-Vision-90b model
Llama90b_agent_def = {           
    "agent_role": 
        """You are an expert in image analysis. Your task is to provide precise, highly detailed, 
        and structured descriptions of images in Markdown format. Ensure every element, including objects, 
        colors, positions, sizes, interactions, and counts, is meticulously described.
        """,
    "agent_instructions": 
        """Analyze the given image with the highest level of detail possible. 
        - Break the description into structured sections: 
          - **Overall Composition**: Describe the image's overall layout and theme.
          - **Foreground Elements**: List all visible elements in the foreground, their shapes, sizes, and positions relative to each other.
          - **Background Elements**: Describe the background objects, colors, and textures.
          - **Colors & Lighting**: Mention dominant colors, shading, and how lighting affects the scene.
          - **Object Count & Arrangement**: Provide accurate counts for repetitive objects and their spatial organization.
          - **Interactions & Movement**: If elements interact (e.g., overlapping, shadows, reflections, motion), describe them explicitly.
        - Use precise adjectives and spatial relationships (e.g., 'centered', 'adjacent to', 'partially overlapping').
        """,
    }

# Prepare the input prompt Instructions/definition for gpt4o model
GPT4o_agent_def = {           
    "agent_role":
        """You are a helpful assistant that responds in Markdown.
        """,
    "agent_instructions": 
        """Provide a detailed semantic description about the provided image; including all segments, 
        their relative positions in the image, their colors, their relative sizes, 
        their precise counts if more than one from same segment, 
        their relative motion and interaction between them.
        """,
    }


#######################################################################################################################################
##############################################LLM INSTANCES SETUP#################################################################

## Importing system Variables
#load_dotenv("I:/Z_AI4DCstaff/999_personalFolders/Aly/MasterThesis/10_VPQA_Evaluation_Pipeline/VPQA_Evaluation/.env")     
load_dotenv()         
os.environ['http_proxy'] = os.getenv("GenAI_Proxy")
os.environ['https_proxy'] = os.getenv("GenAI_Proxy")
azure_api_key = os.getenv("azure_api_key")
azure_endpoint = os.getenv("azure_endpoint")
langfuse_secret_key = os.getenv('langfuse_secret_key')
langfuse_public_key = os.getenv('langfuse_public_key')

## Disable PostHog Analytics
client = chromadb.Client(Settings(
    anonymized_telemetry=False  # Disable telemetry to PostHog
))
langchain.debug = False  # This disables PostHog telemetry

## Calling langfuse handler
langfuse_handler = CallbackHandler(
    public_key= langfuse_public_key,
    secret_key= langfuse_secret_key,
    host="http://localhost:3000"
)

# Embedding model AzureOpenAI instance 
llm_embedding_txt = AzureOpenAIEmbeddings(
    api_key=azure_api_key,  # type: ignore
    azure_endpoint=azure_endpoint,
    azure_deployment= "text-embedding-3-large",
    api_version="2023-05-15" )
emb_fn_txt = AzureOpenAIEmbeddingFunc(llm_embedding_txt)

# ## Llama model Ollama instance 
# llm_Llama = ChatOllama(
   # base_url="http://10.5.58.243:5000/",
   # model="llama3.2-vision:90b",
   # temperature=0
# )

## GPT model AzureOpenAI instance 
llm_GPT4o = AzureChatOpenAI(
    api_key=azure_api_key,  # type: ignore
    azure_endpoint=azure_endpoint,
    azure_deployment= "gpt-4o",
    api_version="2023-05-15" ,
    temperature=0)

llm_gpt = llm_GPT4o              ## set the llm model with llm_GPT4o or llm_Llama90b to select the deployed model

##################################################################################################################
##############################################EVALUATION LLM INSTANCES SETUP######################################
generator_llm = AzureChatOpenAI(
    api_key=azure_api_key,  # type: ignore
    azure_endpoint=azure_endpoint,
    azure_deployment= "gpt-4",
    api_version="2023-05-15",
    temperature=0)

critic_llm = AzureChatOpenAI(
    api_key=azure_api_key,  # type: ignore
    azure_endpoint=azure_endpoint,
    azure_deployment= "gpt-4o",
    api_version="2023-05-15",
    temperature=0)

#context_llm_embedding = llm_embedding
context_llm_embedding_txt = llm_embedding_txt
 
##################################################################################################################
##############################################VECTOR BASE SETUP####################################################

# # ### Create vector store collections
# # Multimodal Descriptions Text Vector Store
# client1_text = chromadb.PersistentClient(path=frame_descriptions_db_path)
# collection1_text = 'text_collection'
# mmtext_vector_store = client1_text.get_or_create_collection(
    # name=collection1_text,
    # embedding_function=emb_fn_txt,
    # metadata={"hnsw:space": "cosine"})

# # Audio Transcripts text Vector Store
# client2_text = chromadb.PersistentClient(path=audio_db_path)
# collection2_text = 'text_collection'
# audio_vector_store = client2_text.get_or_create_collection(
    # name=collection2_text,
    # embedding_function=emb_fn_txt,
    # metadata={"hnsw:space": "cosine"})


#################################################################################################################
##################################################################################################################

## Create multimodal retriever tool 
class MultiModalRetriever:
    def __init__(self, audio_retriever, text_retriever, window_size=5):
        self.audio_retriever = audio_retriever
        self.text_retriever = text_retriever
        self.window_size = window_size  # Number of consecutive frames to retrieve

    def retrieve(self, query, top_k= 5):
        # Retrieve relevant frame descriptions
        text_results = self.text_retriever.invoke(query)[:top_k]
        audio_results = self.audio_retriever.invoke(query)[:top_k]
        
        # Fetch additional frames (context window) for sequential concept
        extended_text_results = text_results #self.get_sequential_frames(text_results, 1)
        #extended_audio_results = self.get_sequential_frames(audio_results)
        
        return {
            "texts": extended_text_results,
            "audio": audio_results,
        }

    def get_sequential_frames(self, results, top_k= 1):
        sequential_results = []
        for res in results:
            frame_id = res.metadata["frame_id"]  # Assume each text result has a frame ID
            for offset in range(-self.window_size, self.window_size + 1):
                neighbor_frame_id = int(frame_id) + offset
                neighbor_res = self.text_retriever.invoke(str(neighbor_frame_id))[:top_k]
                sequential_results.extend(neighbor_res)
        return sequential_results

    

## Helper functions 
def seconds_to_hhmm(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}"


def move_files(FILES_sourcePath, FILE_savePath):
    """
    Move all .mp4 files from source_folder to destination_folder.

    Parameters:
    source_folder (str): The path to the source folder.
    destination_folder (str): The path to the destination folder.
    """
    # Ensure the destination folder exists
    current_time = time.time()
    os.makedirs(FILE_savePath, exist_ok=True)

    # Iterate over all files in the source folder
    for filename in os.listdir(FILES_sourcePath):
        if filename.endswith('.mp4') or filename.endswith('.mp3') or filename.endswith('.wav'):
            # Construct full file path
            source_file = os.path.join(FILES_sourcePath, filename)
            destination_file = os.path.join(FILE_savePath, filename)
            
            # Move the file
            shutil.move(source_file, destination_file)
            os.utime(destination_file, (current_time, current_time))
            print(f'Moved: {source_file} to {destination_file}')
                        
 
def delete_old_files_of_type(folder_path, file_extension):
    """
    Delete all files in the specified folder with a given file extension that are older than 1 week.

    Parameters:
    folder_path (str): The path to the folder from which to delete old files.
    file_extension (str): The file extension to filter files (e.g., '.txt', '.mp4').
    """
    # Get the current time
    current_time = time.time()
    
    # Define the age limit (1 week in seconds)
    age_limit = 7 * 24 * 60 * 60  # 7 days

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file (not a directory) and has the specified extension
        if os.path.isfile(file_path) and filename.endswith(file_extension):
            # Get the file's last modified time
            file_age = current_time - os.path.getmtime(file_path)
            
            # Check if the file is older than the age limit
            if file_age > age_limit:
                os.remove(file_path)
                print(f'Deleted: {file_path}')



def text_to_pdf(text_file, pdf_file):
    # Create instance of FPDF class
    print(f'### Saving pdf of {text_file}') 
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # Set font
    pdf.set_font("Arial", size=12)

    # Define the width of the cell
    page_width = pdf.w - 2 * pdf.l_margin

    try:
        with open(text_file, "r", encoding="utf-8", errors="replace") as file:  # Handle non-UTF-8 characters
            for line in file:
                pdf.multi_cell(page_width, 10, txt=line.encode('utf-8', 'replace').decode('latin-1'), border=0, align='L')

        pdf.output(pdf_file)
        print(f"PDF saved: {pdf_file}")

    except Exception as e:
        print(f"Error converting {text_file} to PDF: {e}")    
    
    
def format_timestamp(seconds):
    """Convert seconds into [hh:mm:ss - hh:mm:ss] format."""
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    return f"{hh:02}:{mm:02}:{ss:02}"
    
## Extract Frames from Video
def extract_frames(video_path, path_output_folder, N_frames_perSecond_2_process):

    print(f'#### Extracting frames of {video_path}')

    os.makedirs(path_output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    savedframe_count = 0
    success, frame = cap.read()
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS as an integer
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
    duration = total_frames / fps if fps > 0 else 0  # Calculate video duration
    
    while success:
        if frame_count % N_frames_perSecond_2_process == 0:
            frame_path = os.path.join(path_output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            savedframe_count += 1
        success, frame = cap.read()
        frame_count += 1

    cap.release()
    print(f"{savedframe_count} Frames saved to {path_output_folder}")
    print(f"The used camera has fps= {fps}")
    print(f"Video duration = {duration:.2f} seconds") 
    
    return fps
    
    
## Remove redundant frames from database
def remove_duplicates_from_vector_store(vector_store, similarity_threshold=0.99):
    # Step 1: Retrieve embeddings and metadata from the vector store
    data = vector_store.get(include=["embeddings", "metadatas"])
    embeddings = data["embeddings"]
    metadata = data["metadatas"]
    ids = data["ids"]
    
    print(f"Total embeddings before deduplication: {len(embeddings)}")
    
    # Step 2: Deduplicate embeddings
    normalized_embeddings = np.array(embeddings)  # Ensure embeddings are a NumPy array
    similarity_matrix = cosine_similarity(normalized_embeddings)
    
    # Identify duplicate indices
    to_remove = set()
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > similarity_threshold:
                to_remove.add(j)  # Mark duplicates for removal

    keep_indices = set(range(len(embeddings))) - to_remove
    keep_indices = sorted(keep_indices)  # Preserve order

    # Collect IDs or fallback metadata to delete
    ids_to_delete = []
    duplicates_metadata = []
    for i in to_remove:
        if ids[i] is not None:
            ids_to_delete.append(ids[i])
                

    print(f"Indices marked for removal: {sorted(to_remove)}")
    print(f"No. of indices marked for removal: {len(to_remove)}")
    print(f"IDs marked for deletion: {ids_to_delete}")

    # check the metadata 
    # for i, meta in enumerate(metadata):
    #     print(f"Metadata at index {i}: {meta}")
    
    # Step 3: Delete duplicates
    if ids_to_delete:
        vector_store.delete(ids=ids_to_delete)
        print("Duplicates successfully removed using IDs.")
    else:
        print("No duplicates were removed. Check metadata validity.")
    
    print(f"Total embeddings after deduplication: {len(keep_indices)}")
    print(f"Duplicates removed: {len(to_remove)}")
    
    return vector_store
    
    
## Generate stored frames semantic descriptions
def generate_semantic_description(agent_def, main_folder_path, output_textfile, fps): 
    AI_images_contexts = []

    print(f'#### Generating semantic descriptions of {main_folder_path}')
    # Process each image in the folder and its subfolders
    countI = -1 
    for root, _, files in os.walk(main_folder_path):  # Walk through all subdirectories
        time_startI = time.time()
        for filename in sorted(files):  # Ensure frames are processed in order
            countI = countI + 1 
            timeI = time.time()
            execution_timeI = timeI - time_startI
            if countI == 0:
                timeILeft = 0  # Or handle it another way
            else:    
                timeILeft = (execution_timeI / countI * len(files) )
                
            print("    processing image ", countI , " of " , len(files), ", ETA ", str(timedelta(seconds=int(timeILeft)))  ) 
            
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Filter for image files
                file_path = os.path.join(root, filename)
                
                # Extract frame number from filename (assuming format "frame_XX.jpg")
                try:
                    frame_number = int(filename.split("_")[-1].split(".")[0])
                    start_time = frame_number / fps  # Start timestamp in seconds
                    end_time = (frame_number + 1) / fps  # End timestamp in seconds
                except ValueError:
                    start_time, end_time = None, None  # Handle unexpected filename formats
                
                # Format timestamps
                if start_time is not None and end_time is not None:
                    timestamp_str = f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}]"
                else:
                    timestamp_str = "[Unknown - Unknown]"
                
                
                # Read the image as bytes and encode to base64
                with open(file_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

                messages = [           
                    {"role": "system", "content": agent_def['agent_role']},
                    {"role": "user", "content": [
                        {"type": "text", "text": agent_def['agent_instructions']},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ]                
                
                # Invoke the llm model (replace `llm_model.invoke` with your function to query the model)
                modality_content = llm_model.invoke(messages)
                
                # Append the model's response to the list
                AI_images_contexts.append({
                    "image_name": filename,
                    "folder": os.path.basename(root),  # Add the folder name for context
                    "frame_timestamp": timestamp_str,
                    "description": modality_content.content
                })
                
                # wait for some secs specially in case of ollama models as server kina does not support high rate requests 
                time.sleep(5)    
                
    # Example: Save descriptions to a file
    with open(output_textfile, "w") as f:
        for item in AI_images_contexts:
            f.write(f"Folder: {item['folder']}\nImage: {item['image_name']}\nTimestamp: {item['frame_timestamp']}\nDescription: {item['description']}\n\n")


def delete_old_files_of_type(folder_path, file_extension):
    """
    Delete all files in the specified folder with a given file extension that are older than 1 week.

    Parameters:
    folder_path (str): The path to the folder from which to delete old files.
    file_extension (str): The file extension to filter files (e.g., '.txt', '.mp4').
    """
    # Get the current time
    current_time = time.time()
    
    # Define the age limit (1 week in seconds)
    age_limit = 7 * 24 * 60 * 60  # 7 days

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file (not a directory) and has the specified extension
        if os.path.isfile(file_path) and filename.endswith(file_extension):
            # Get the file's last modified time
            file_age = current_time - os.path.getmtime(file_path)
            
            # Check if the file is older than the age limit
            if file_age > age_limit:
                os.remove(file_path)
                print(f'Deleted: {file_path}')
                
                
                
def move_files(video_folder, FILE_savePath):
    """
    Move all .mp4 files from source_folder to destination_folder.

    Parameters:
    source_folder (str): The path to the source folder.
    destination_folder (str): The path to the destination folder.
    """
    # Ensure the destination folder exists
    current_time = time.time()
    os.makedirs(FILE_savePath, exist_ok=True)

    # Iterate over all files in the source folder
    for filename in os.listdir(video_folder):
        if filename.endswith('.mp4') or filename.endswith('.mp3') or filename.endswith('.wav'):
            # Construct full file path
            source_file = os.path.join(video_folder, filename)
            destination_file = os.path.join(FILE_savePath, filename)
            
            # Move the file
            shutil.move(source_file, destination_file)
            os.utime(destination_file, (current_time, current_time))
            print(f'Moved: {source_file} to {destination_file}')     
            

## RAG Helper functions

def plt_img_base64(img_base64):
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    # Display the image by rendering the HTML
    display(HTML(image_html))
        
        
def load_documents(text_folder=None, text_list=None):
    """
    Loads documents from a folder or a list and returns them in a format suitable for further processing.
    """
    documents = []
    
    if text_folder:
        print("Loading documents from folder...")
        for file in os.listdir(text_folder):
            file_path = os.path.join(text_folder, file)
            try:
                # Handle .txt files
                if file.endswith(".txt"):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                            documents.append(Document(page_content=text, metadata={"file_name": file}))
                    except Exception as e:
                        print(f"Error reading .txt file {file}: {e}")
                
                # Handle .pptx files using UnstructuredFileLoader
                elif file.endswith(".pptx"):
                    loader = UnstructuredFileLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                
                # Handle PDFs using PyMuPDFLoader
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                
                else:
                    print(f"Skipping unsupported file type: {file}")
            
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    
    elif text_list:
        print("Loading text documents from provided list...")
        for i, text in enumerate(text_list):
            documents.append(Document(page_content=text, metadata={"document_id": f"doc_{i}"}))
    else:
        raise ValueError("Either 'text_folder' or 'text_list' must be provided.")

    print('no. of Crawled_Documents: ', len(documents))
    return documents
    

def chunk_documents(documents, embeddings):
    """
    Chunks documents using SemanticChunker.
    """
    if not embeddings:
        raise ValueError("Embeddings must be provided for semantic chunking.")

    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=100
    )
    chunked_documents = text_splitter.split_documents(documents)
    print('no.of Chunked_Documents: ', len(chunked_documents))
    
    # page_content = [doc.page_content for doc in chunked_documents]
    # metadata = [doc.metadata for doc in chunked_documents]
    ids = [
        
        str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content))
        for doc in chunked_documents
    ]
    unique_ids = list(set(ids))
    seen_ids = set()
    unique_docs = [
        doc for doc, idx in zip(chunked_documents, ids)
        if idx not in seen_ids and (seen_ids.add(idx) or True)
    ]

    metadata = [i.metadata for i in unique_docs]
    page_content = [i.page_content for i in unique_docs]

    return page_content, metadata, unique_ids


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string.

    Args:
    base64_string (str): Base64 string of the original image.
    size (tuple): Desired size of the image as (width, height).

    Returns:
    str: Base64 string of the resized image.
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def split_image_text_types(docs):
    """Split numpy array images and texts"""
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        # if is_base64(doc):
        #     # Resize image to avoid OAI server error
        #     images.append(
        #         resize_base64_image(doc, size=(250, 250))
        #     )  # base64 encoded str
        # else:
        text.append(doc)
    return {"images": images, "texts": text}



def extract_frames_from_pdf(text_folder, max_pages=10):
    """
    Reads PDFs, extracts complete descriptions for each frame (spanning multiple pages),
    and ensures merging does not break at page boundaries.
    Process in chunks of max_pages to optimize performance.
    """
    frame_texts = {}  # Store frame descriptions with frame_id as key

    for filename in os.listdir(text_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(text_folder, filename)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            num_pages = len(pages)
            for start in range(0, num_pages):
                # Get the chunk of pages to process
                end = min(start + max_pages, num_pages)
                chunk_pages = pages[start:end]

                # Combine the text from the chunk of pages
                combined_text = "\n".join(page.page_content.strip() for page in chunk_pages)
                lines = combined_text.split("\n")

                current_frame_id = None
                buffer = []  # Store description text temporarily

                for line in lines:
                    frame_match = re.search(r"Image:\s*frame_(\d+)\.jpg", line)

                    if frame_match:
                        # If a new frame starts, store previous frame data
                        if current_frame_id is not None:
                            if buffer:  # Only store if buffer is not empty
                                frame_texts[current_frame_id] = "\n".join(buffer).strip()
                                buffer = []  # Reset buffer for new frame

                        # Start a new frame
                        current_frame_id = frame_match.group(1)

                    # If we have a current frame, accumulate text in the buffer
                    if current_frame_id:
                        buffer.append(line.strip())

                # Store the last frame after finishing the chunk
                if current_frame_id is not None and buffer:
                    frame_texts[current_frame_id] = "\n".join(buffer).strip()

    return frame_texts



def process_and_store_frames(frame_texts, embeddings):
    """
    Converts extracted frame descriptions into chunked data and stores them.
    """
    chunked_texts = []
    metadata_list = []
    ids = []

    for frame_id, full_text in frame_texts.items():
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, full_text))

        metadata = {
            "frame_id": frame_id,
        }

        chunked_texts.append(full_text)
        metadata_list.append(metadata)
        ids.append(doc_id)

    return chunked_texts, metadata_list, ids


def prompt_func(data_dict):
    # Extract text from each Document object and combine them into a structured format
    formatted_texts = "\n".join(doc.page_content for doc in data_dict["context"]["texts"])

    # # Create the message for analysis
    # text_message_content = (
        # "Analyze the retrieved video context and infer interactions between objects.\n"
        # f"User-provided question: {data_dict['question']}\n\n"
        # "Here are time-sequenced descriptions from the video:\n"
        # f"{formatted_texts}\n\n"
        # "Determine the following:\n"
        # "1. Identify objects that are interacting.\n"
        # "2. Detect any unusual behavior in the video sequence.\n"
        # "3. Summarize the observed actions and movements.\n"
        # "4. If any anomaly is detected, describe it explicitly.\n"
        # "If not asked about motion analsing, interactions or anomalies; Answer the questions directly based on frame_wise information without following the time-sequenced descriptions instructions."

    # )
    
    # Create the baseline message for analysis
    text_message_content = (
            "You are an expert AI assistant answering questions based on retrieved video frame descriptions. "
            "Use the provided contexts carefully to generate accurate, concise, and relevant responses. "
            #"If the answer is not found in the context, say 'I do not have enough information. '\n\n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Here are frames descriptions from the video:\n"
            f"{formatted_texts}\n\n"
    )
    # Return the message directly as a HumanMessage
    return [HumanMessage(content=text_message_content)]

            
############################################################################################################################   
##########################################################MAIN##############################################################         
if __name__ == "__main__":
    start_time = time.time()

    # STEP 1: Generate the semantic descriptions for all input videos
    for root, _, files in os.walk(video_folder):  # Walk through all subfolders
        for video_file in files:
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Check if it's a video
                video_path = os.path.join(root, video_file)
                print(f"#### Processing video: {video_path}")

                # Extract the relative subfolder name (if any)
                relative_subfolder = os.path.relpath(root, video_folder)

                # Extract video name (without extension)
                video_name = os.path.splitext(video_file)[0]

                # Define the corresponding output subfolder
                video_output_folder = os.path.join(extractedFrames_folder, relative_subfolder, video_name)
                os.makedirs(video_output_folder, exist_ok=True)

                # Extract frames and get FPS
                fps = extract_frames(video_path, video_output_folder, frame_interval_i)

                # Define unique text and PDF file names, keeping subfolder structure
                output_subfolder = os.path.join(output_folder, relative_subfolder)
                os.makedirs(output_subfolder, exist_ok=True)

                video_text_file = os.path.join(output_subfolder, f"{video_name}.txt")
                video_pdf_file = os.path.join(output_subfolder, f"{video_name}.pdf")

                # Generate semantic descriptions and save per video
                generate_semantic_description(GPT4o_agent_def, video_output_folder, video_text_file, fps)

                # Convert the text file into a PDF
                text_to_pdf(video_text_file, video_pdf_file)

    fps_values = []
    f_values = []
    ar_values = []
    cr_values = []
    cp_values = []
    
    ## STEP 7: Loading the prepared testset 
    SurgicalRobots_VPQA_testset = pd.read_excel('Dataset/VPQA_testset_SurgicalRobots233.xlsx')
    #display(HW-Troubleshooting_VPQA_testset)
    test_questions = SurgicalRobots_VPQA_testset['Question'].values.tolist()
    test_groundtruths = SurgicalRobots_VPQA_testset['Answer'].values.tolist()   
    
    
    HW_Troubleshooting_VPQA_testset = pd.read_excel('Dataset/VPQA_testset_HWTroubleshooting2.xlsx')
    #display(HW-Troubleshooting_VPQA_testset)
    test_questions = HW_Troubleshooting_VPQA_testset['Question'].values.tolist()
    test_groundtruths = HW_Troubleshooting_VPQA_testset['Answer'].values.tolist()     
    
    ## STEP 2: Add semantic descriptions into multimodal collection
    # Step 21: Load documents frame wise
    frameDescriptions_data = extract_frames_from_pdf(frameDescriptions_text_folder, 10)
    frame_interval = [6,] #[1, 2, 5, 10, 15, 30]
    for frame_interval_i in frame_interval:
        fps_i = (30/frame_interval_i)
        #frame_descriptions_db_path = f"chroma_mmtext_db_A-Prompt-Agent_SurgicalRobotsAnormal_{fps_i}fps_99th"
        frame_descriptions_db_path = f"chroma_mmtext_db_A-Prompt-Agent_HW-Troubleshooting_{fps_i}fps_99th"
        #os.makedirs(frame_descriptions_db_path, exist_ok=True)
        #audio_db_path = f"chroma_audio_db_SurgicalRobotsAnormal_{fps_i}fps"
        audio_db_path = f"chroma_audio_db_HW-Troubleshooting_{fps_i}fps"
        #os.makedirs(audio_db_path, exist_ok=True)
        
        ##################################################################################################################
        ##############################################VECTOR BASE SETUP####################################################

        ### Create vector store collections
        # Multimodal Descriptions Text Vector Store
        client1_text = chromadb.PersistentClient(path=frame_descriptions_db_path)
        collection1_text = 'text_collection'
        mmtext_vector_store = client1_text.get_or_create_collection(
            name=collection1_text,
            embedding_function=emb_fn_txt,
            metadata={"hnsw:space": "cosine"})

        # Audio Transcripts text Vector Store
        client2_text = chromadb.PersistentClient(path=audio_db_path)
        collection2_text = 'text_collection'
        audio_vector_store = client2_text.get_or_create_collection(
            name=collection2_text,
            embedding_function=emb_fn_txt,
            metadata={"hnsw:space": "cosine"})


        #################################################################################################################
        ##################################################################################################################
        
        if frame_interval_i == 1:
            # print('Already done before')
            # frame_descriptions_db_path = f"chroma_mmtext_db_A-Prompt-Agent_SurgicalRobotsAnormal_{int(fps_i)}fps_99th"
            # audio_db_path = f"chroma_audio_db_SurgicalRobotsAnormal_{int(fps_i)}fps"
            # continue
            globals()[f"frameDescriptions_data_{frame_interval_i}"] = frameDescriptions_data
        else:
            # Create a new filtered variable dynamically
            filtered_data = {str(frame_id): frameDescriptions_data[str(frame_id)]
                             for frame_id in range(0, max(map(int, frameDescriptions_data.keys())), frame_interval_i)
                             #for frame_id in np.arange(0, max(map(int, frameDescriptions_data.keys())), frame_interval_i)
                             if str(frame_id) in frameDescriptions_data}

            # Assign the filtered data to a dynamically named variable
            globals()[f"frameDescriptions_data_{frame_interval_i}"] = filtered_data

            #print(f"Created variable: frameDescriptions_data_{frame_interval_i}")

        # Step 22: Chunk documents
        frameDescriptions_chunked_texts, frames_metadata_list, frames_ids = process_and_store_frames(globals().get(f"frameDescriptions_data_{frame_interval_i}"), embeddings=emb_fn_txt)
        # Step 23: Add chunks into collection
        for i in range(0,len(frameDescriptions_chunked_texts),50):
            mmtext_vector_store.add(documents=frameDescriptions_chunked_texts[i:(i+50)], metadatas=frames_metadata_list[i:(i+50)], ids=frames_ids[i:(i+50)])
        print(len(frameDescriptions_chunked_texts), 'chunks added to frame_text_collection vectorstore')

        ## STEP 3: Add wihsper generated audio transcript into text collection
        # Step 31: Load documents
        audioTranscript_documents = load_documents(text_folder=audioTranscript_text_folder)
        # Step 32: Chunk documents
        audioTranscript_chunked_texts, audio_metadata_list, audio_ids = chunk_documents(audioTranscript_documents, embeddings=emb_fn_txt)
        # Step 33: Add chunks into collection
        for i in range(0,len(audioTranscript_chunked_texts),50):
            audio_vector_store.add(documents=audioTranscript_chunked_texts[i:(i+50)], metadatas=audio_metadata_list[i:(i+50)], ids=audio_ids[i:(i+50)])    
        print(len(audioTranscript_chunked_texts), 'chunks added to audio_text_collection vectorstore')

        ## STEP 4: Remove redundant embeddings for similar frames from multimodal collection
        mmtext_vector_store = remove_duplicates_from_vector_store(mmtext_vector_store, VectorEmbeddings_similarity_threshold)

        #print('done!!')

        ## STEP 5: Define Multimodal Retriever
        audio_vector_store = Chroma(persist_directory= audio_db_path, collection_name= 'text_collection', embedding_function=emb_fn_txt)
        audio_retriever = audio_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        mmtext_vector_store = Chroma(persist_directory= frame_descriptions_db_path, collection_name= 'text_collection', embedding_function=emb_fn_txt)
        mm_framedescription_retriever = mmtext_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


        # ##### A-Prompt top k retrieved descriptions
        # question = "what is the color of the wire that the guy in the video is holding indicating that it has an issue?"
        # print("top 5 documents to this question: ")
        # docs = mm_framedescription_retriever.invoke(question, k=3)
        # for doc in docs:
            # if is_base64(doc.page_content):
                # plt_img_base64(doc.page_content)
            # else:
                # print(doc.page_content)
                # print('**********************************************************************************')
                
       
        ## STEP 6: Define RAG pipeline
        # Update retriever with sequential frame retrieval
        retriever_runnable = RunnableLambda(lambda query: MultiModalRetriever(audio_retriever, mm_framedescription_retriever, window_size=5).retrieve(query))

        # Update Sequential RAG chain
        sequential_mmrag_chain = (
            {
                "context": retriever_runnable,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(prompt_func)
            | llm_gpt
            | StrOutputParser()
        )


        # ##### Here invoke mmRAG for QAs
        # response= sequential_mmrag_chain.invoke("what is the color of the wire that the guy in the video is holding indicating that it has an issue?")
        # print(response)
       
        # response= sequential_mmrag_chain.invoke("How do you know the correct hypertac connection lockation at the test bench?")
        # print(response)
        
        # response= sequential_mmrag_chain.invoke("What is the tool used to unlock the clamps and inject wires?")
        # print(response)
        
        # response= sequential_mmrag_chain.invoke("What is the company name written in the hardware testbench that contains all the hardware ECUs and wiring?")
        # print(response)    
        
        # response= sequential_mmrag_chain.invoke("how did the guy know the effect of the problem from the software user interface?")
        # print(response)
        

        ##### Here questions addressing the quality of the Sequential-MMRAG
        # response= sequential_mmrag_chain.invoke("What is the color of the first cylinderical object that surgical tool misdrop it while having troubles putting it into the box, and at which time stamp that happened?")
        # print('###################Q11:: ', response)

        # response= sequential_mmrag_chain.invoke("What is the color of the second cylinderical object that surgical tool misdrop it while having troubles holding  it, and at which time stamp that happened?")
        # print('###################Q21:: ', response)
        
        # response= sequential_mmrag_chain.invoke("whatis the color of the object that the tool on right hand side is holding?")
        # print('###################Q31:: ', response)
        
        # response= sequential_mmrag_chain.invoke("whatis the color of the object that the tool on the left hand side is holding?")
        # print('###################Q41:: ', response)

        # response= sequential_mmrag_chain.invoke("What is the color of the first cylinderical object that surgical tool misdrop it while having troubles putting it into the box, and at which time stamp that happened?")
        # print('###################Q12:: ', response)
        
        # response= sequential_mmrag_chain.invoke("What is the color of the second cylinderical object that surgical tool misdrop it while having troubles holding  it, and at which time stamp that happened?")
        # print('###################Q22:: ', response)

        # response= sequential_mmrag_chain.invoke("whatis the color of the object that the tool on right hand side is holding?")
        # print('###################Q32:: ', response)
        
        # response= sequential_mmrag_chain.invoke("whatis the color of the object that the tool on the left hand side is holding?")
        # print('###################Q42:: ', response)

        # response= sequential_mmrag_chain.invoke("What is the color of the first cylinderical object that surgical tool misdrop it while having troubles putting it into the box, and at which time stamp that happened?")
        # print('###################Q13:: ', response)
        
        # response= sequential_mmrag_chain.invoke("What is the color of the second cylinderical object that surgical tool misdrop it while having troubles holding  it, and at which time stamp that happened?")
        # print('###################Q23:: ', response)

        # response= sequential_mmrag_chain.invoke("whatis the color of the object that the tool on right hand side is holding?")
        # print('###################Q33:: ', response)
        
        # response= sequential_mmrag_chain.invoke("whatis the color of the object that the tool on the left hand side is holding?")
        # print('###################Q43:: ', response)
        
        
        ## STEP 8: Evaluation of MMRAG on VPQA using RAGAS
        # loading questions
        answers = []
        contexts = []

        for question in test_questions[:15]:
            response = sequential_mmrag_chain.invoke(question)
            answers.append(response)
            relevant_docs = mm_framedescription_retriever.invoke(question, k= 3)
            contexts.append([doc.page_content for doc in relevant_docs])
               
        response_dataset = Dataset.from_dict(
            {
                "question": test_questions[:15],
                "ground_truth": test_groundtruths[:15],
                "answer": answers,
                "contexts": contexts,
            }
        )

        response_dataset_df = response_dataset.to_pandas()
        #print('answer to the 1st question: ', response_dataset_df.loc[0]['answer'])
        #print('ground_truth to the 1st question: ', response_dataset_df.loc[0]['ground_truth'])
        #display(response_dataset_df)
       
        # evaluating using gpt4 and embedding-3-large
        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
        run_config = RunConfig(timeout=600)  # Increase timeout to 600 seconds  
        results = evaluate(response_dataset, metrics=metrics, llm=critic_llm, embeddings=context_llm_embedding_txt, run_config=run_config, raise_exceptions=False)
        results_df = results.to_pandas()

        fps_values.append(fps_i)
        f_values.append(results_df['faithfulness'].mean())
        ar_values.append(results_df['answer_relevancy'].mean())
        cr_values.append(results_df['context_recall'].mean())
        cp_values.append(results_df['context_precision'].mean())
        
        print(f'for fps = {fps_i} ,,, the evaluation metrics are ::')
        print('faithfulness:' + str(results_df['faithfulness'].mean()))
        print('answer_relevancy:' + str(results_df['answer_relevancy'].mean()))
        print('context_recall:' + str(results_df['context_recall'].mean()))
        print('context_precision:' + str(results_df['context_precision'].mean()))   
    
    results_df.to_excel('Eval_Results/Evaluation_20VPQA_MMRAG_A-Agent-SingleFrame_HW_Troubleshooting.xlsx')
    #results_df.to_excel('Eval_Results/Evaluation_20VPQA_MMRAG_A-Agent-Sequential_Surgical_abnormal222.xlsx')


    ## STEP 9: Storing the fps with according eval metrics
    eval_analysis_data = {
    "FPS": fps_values,  # List of FPS values
    "Faithfulness": f_values,  
    "AnswerRelevancy": ar_values,
    "ContextRecall": cr_values,
    "ContextPrecision": cp_values
    }

    # Create a DataFrame
    eval_analysis_data_df = pd.DataFrame(eval_analysis_data)

    # Save to an Excel file
    #eval_analysis_data_df.to_excel("Eval_Results/eval_analysis_metrics_anomaly_SequentialFrameRAG222.xlsx", index=False)
    eval_analysis_data_df.to_excel("Eval_Results/eval_analysis_metrics_anomaly_HW_Troubleshooting.xlsx", index=False)


    # # script_directory = os.path.dirname(os.path.abspath(__file__))
    # # #print("script_directory", script_directory ) 
    # if (0): 
        # move_files(video_folder, FILE_savePath)
        # delete_old_files_of_type(FILE_savePath, ".mp4")            
        # delete_old_files_of_type(output_folder, ".txt")    
        # delete_old_files_of_type(output_folder, ".pdf")       
        # delete_old_files_of_type(script_directory, ".sh")
        # delete_old_files_of_type(script_directory, ".txt")
    
    # # print(" your videos have been moved to {FILE_savePath} and will be deleted after one week")
    # # print(" check your transcripts in {transcript_saving_path} and move them to your workfolder as they will be deleted after 1 week " ) 

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total processing time: {execution_time:.2f} seconds")

    print("-------------- Script Finished ---------------")

##################################################################
    

