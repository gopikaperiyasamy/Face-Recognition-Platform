# Face-Recognition-Platform
Core Functionalities

- Face Registration: Capture and store webcam images with timestamps in MongoDB.
- Face Encoding & Vectorization: Convert facial images into embeddings using a Pickle-based pipeline and store them in Qdrant.
- Live Recognition: Stream webcam input and recognize registered faces in real-time.
- AI Q&A Interface:Ask semantic questions (e.g., "Who was the last registered person?") using LangChain + OpenAI.



 Registration
- Captures webcam images.
- Assigns a user name and stores the image and metadata in MongoDB.
- Converts image to face embeddings and stores in Qdrant DB.

 Real-Time Recognition
- Reads from webcam.
- Compares live embeddings with stored vectors.
- Displays recognized names with bounding boxes.

 AI Chat Interface
- Users can ask natural-language questions about the data.
- Uses LangChain, Qdrant, and OpenAI to perform semantic search and answer.


 Tech Stack

|---------------------------------------------------------------------|
| Module            | Technology                                      |
|-------------------|-------------------------------------------------|
| Frontend          | Gradio                                          |
| Backend           | Python Flask                                    |
| Face Recognition  | Python (`face_recognition`, `OpenCV`)           |
| RAG Engine        | Python (`LangChain`, `Qdrant`, `OpenAI`)        |
| Image Storage     | MongoDB                                         |
| Embedding Storage | Qdrant                                          |
| Environment       | `.env` (for keys, URIs, config)                 |
-----------------------------------------------------------------------

 Prerequisites

- Python 3.8+
- MongoDB (local or cloud URI)
- Qdrant (run locally or use cloud Qdrant)
- Webcam
- OpenAI API key

requirements.txt

flask
gradio
opencv-python
face_recognition
langchain
openai
qdrant-client
pymongo
numpy
python-dotenv
tenacity
3. Configure Environment Variables
Create a .env file in the root directory:
OPENAI_API_KEY=your_openai_api_key
MONGO_URI=mongodb://localhost:27017/face_db
QDRANT_HOST=http://localhost:6333
4. Run the Flask Backend
python backend/backend.py
6. Launch Gradio UI
python frontend/Ui_frontend.py
Visit the Gradio app at: http://localhost:7860
 
 
 Architecture diagram
 [](../Untitled.fig)
 
“This project is a part of a hackathon run by https://katomaran.com ”