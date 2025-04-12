# After starting containers get the name of the container running the app
docker container list -n 4
docker exec -it kg_rag_chainlit-app-1 python src/upload_data.py data/filename
# Then use the Chainlit UI to ask questions