## **Building a Knowledge Graph-Powered AI Assistant: A Step-by-Step Guide**  
*Deploy a Dockerized system that combines LLMs, frontend, Neo4j, and semantic search*  

---

### **Why This Matters**  
Imagine asking an AI assistant a question and getting answers backed by both real-time data *and* structured knowledge. That’s exactly what this framework does. Under the hood, it uses:  
- **Neo4j**: A graph database to store relationships (e.g., "Einstein studied *Physics*").  
- **Ollama**: Runs open-source LLMs (like Llama 3 or DeepSeek) locally.  
- **Qdrant**: A vector database for semantic search (finding "similar" text chunks).  
- **Chainlit**: A user-friendly UI for chatting and uploading files.  

This short tutorial will guide you through the deployment.  

---

### **Step 1: Setting Up the Basics**  
#### **Prerequisites**  
1. **Docker & Docker Compose**: These tools package everything into isolated containers (like mini virtual machines).  
   - *Windows*: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and enable WSL2 (Windows Subsystem for Linux) under Settings.  
   - *Ubuntu*: Run:  
     ```bash  
     sudo apt update
     curl -fsSL https://get.docker.com | sudo sh  
     sudo curl -L "https://github.com/docker/compose/releases/download/v2.35.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose  
     sudo chmod +x /usr/local/bin/docker-compose  
     ```  
      Alternatively after the second step, you can run:
      ```bash
      sudo apt install docker-compose-plugin
      ```  
      This will install docker-compose as well, but every 'docker-compose' command will change to 'docker compose' (no hyphen).
      <br>
      <br>
      Add your user to the docker group, so you don't have to use sudo everytime. And check the installation. 
      ```bash
      sudo usermod -aG docker $USER
      newgrp docker
      
      # Verify installation
      docker --version
      docker compose version
      ```
2. **Git**: To download the project. 
   - Run:
     ```bash
     git clone https://github.com/ginxy/KG_RAG_chainlit.git /path/to/projects
     ```

---

### **Step 2: Understanding the Project Structure**  
After cloning the repository, you’ll see:  
```
├── .env                            <-- Configuration (ports, model settings, model names) 
├── config/
│   └── models.yaml                 <-- Predefined model configurations and instructions 
├── docker-compose.yml              <-- Orchestrates Neo4j, Ollama, Qdrant, and the app
├── Dockerfile                      <-- Setup for app environment
├── LICENSE
├── logs/
│   └── .gitkeep
├── ollama/                         <-- Custom setup for the LLM server
│   └── Dockerfile
├── progr-data-upload-to-dbs.sh     <-- Script to upload data to databased programmatically
├── requirements.txt
├── restart-docker-with-reset.sh    <-- Commands to Cleanup docker containers and rebuild everything
├── scripts/                        
│   └── start_ollama.sh             <-- Custom setup script for LLM Server
├── show-logs.sh                    <-- Command to get app logs printed to terminal
├── src/                            <-- Core logic for processing data and querying databased, streaming, etc.
│   ├── llm_processor.py
│   ├── main.py
│   ├── model_registry.py
│   ├── db_operations.py
│   ├── pdf_search.cypher
│   ├── src_utils.py
│   ├── upload_data.py
│   └── __init__.py
└── start-docker.sh                 <- Start docker container and just build what needs (re-)building   
```  

**Key Files Explained**:  
- **`.env`**: The "settings" file. For example:  
  - `OLLAMA_MODEL=deepseek-r1:1.5b` specifies which AI model to use.  
  - `KG_SCORE_THRESHOLD=0.15` adjusts how strict the knowledge graph search is.  
- **`docker-compose.yml`**: Defines four services:  
  1. **Neo4j**: The graph database (stores entities like "Quantum Physics").  
  2. **Ollama**: Hosts the LLM (locally, so no OpenAI costs!).  
  3. **Qdrant**: Handles vector embeddings for semantic search.  
  4. **App**: The Python backend + Chainlit UI.  

---

### **Step 3: Launching the System**  
Run this command in the project folder:  
```bash  
docker-compose up --build  
```  

**What’s Happening Behind the Scenes?**  
1. **Docker Images**: Downloads Neo4j, Qdrant, and Ollama images.  
2. **Model Download**: Ollama automatically pulls the `deepseek-r1:1.5b` model (≈1.1GB).  
3. **Python Environment**: Installs dependencies like PyPDF2 and Sentence Transformers.  

**First Run Tip**: Initial build might take up to 5 minutes.  

---

### **Step 4: Using the System**  
#### **1. Chat Interface (http://localhost:8000)**  
- Upload a PDF (e.g., a research paper), and ask questions like:  
  > *"Summarize the key findings about quantum computing."*  
- **Behind the Scenes**:  
  - The PDF is split into chunks, embedded into Qdrant, and linked entities (e.g., "Qubit") are added to Neo4j.  
  - The LLM combines results from both databases to generate answers.  

#### **2. Neo4j Browser (http://localhost:7474)**  
- Log in with `neo4j`/`secure_2025!` to visualize the knowledge graph:  
  ```cypher  
  MATCH (e:Entity) RETURN e LIMIT 25  <!-- Shows entities and relationships -->  
  ```  

#### **3. Ollama API (http://localhost:11434)**  
- Test the LLM directly:  
  ```bash  
  curl http://localhost:11434/api/generate -d '{  
    "model": "deepseek-r1:1.5b",  
    "prompt": "Explain quantum entanglement in one sentence."  
  }'  
  ```  

---

### **Step 5: Uploading Your Own Data**  
#### **Option 1: Drag-and-Drop (UI)**  
1. Go to http://localhost:8000.  
2. Drag a PDF/JSON file into the chat window.  
3. Wait for the confirmation message.  

#### **Option 2: Command Line**  
```bash  
# Get the name of the app container 
docker container list -n 4

# Run the upload script replacing the name (kg_rag_chainlit-app-1) and indicating the path from data
docker exec -it kg_rag_chainlit-app-1 python src/upload_data.py data/filename

# If no file is indicated, the example file is taken
docker exec -it kg_rag_chainlit-app-1 python src/upload_data.py 
```  

**What Happens to Your Data?**  
- PDFs are split into text chunks (with overlap to preserve context).  
- Entities (like people, concepts) are extracted using Presidio and stored in Neo4j.  
- Chunks are embedded into Qdrant for semantic search.  

---

### **Step 6: Customization**  
#### **1. Switch LLM Models**  
Edit `.env`:  
```ini  
OLLAMA_MODEL=deepseek-r1:7b  # Larger model, slower but more sophisticated  
```  
Edit `ollama/Dockerfile`:
```ini
ARG OLLAMA_MODEL=deepseek-r1:7b # Second position to change the model name manually
```
Restart with `docker-compose up --build`. The new model will be downloaded and the server started. 

#### **2. Adjust Chunk Sizes**  
Modify in `.env`:  
```ini  
MAX_TEXT_CHUNK=2000      <!-- Smaller chunks for detailed analysis  
TEXT_OVERLAP=500         <!-- Overlap to avoid losing context  
```  

#### **3. Add Custom Prompts**  
Edit `config/models.yaml` to change how the LLM formats responses:  
```yaml  
- name: deepseek-r1:1.5b  
  template: |  
    [CONTEXT]  
    {context}  
    [QUESTION]  
    {query}
    [INSTRUCTIONS]
    - ANSWER IN PIRATE LANGUAGE  <!-- Yes, you can do this 🏴‍☠️  
```  

---

### **Troubleshooting**  
#### **1. "Ollama Model Not Found"**  
Check the Ollama logs:  
```bash  
docker logs ollama  
```  
If the model isn’t downloading, manually pull it:  
```bash  
docker exec -it ollama ollama pull deepseek-r1:1.5b  
```  

#### **2. Neo4j Connection Issues**  
Ensure your `.env` matches the `docker-compose.yml` credentials (NEO4J_AUTH variable).  

#### **3. Slow Performance**  
- Reduce `MAX_RETRIEVAL_RESULTS` in `.env` to limit search results.  
- Allocate more RAM to Docker (Settings → Resources in Docker Desktop). 
- Change memory and mem_limit in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 6g # <- Change memory allocated
mem_limit: 6g    # <- and memory limits  
```
---

### **Final Thoughts**  
This framework is a playground for experimenting with hybrid AI systems. Want to go further? Try:  
- Adding web scraping to auto-populate the knowledge graph.  
- Integrating voice input with Whisper.  
- Fine-tuning the LLM on your domain-specific data.  

The key is to start small — deploy the system, upload a PDF, and ask a question. Once you see it working, the possibilities feel endless.  

*Got stuck? Feel free to contact me, and I’ll try to help!* 🚀  

--- 

Please note: This repository is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
