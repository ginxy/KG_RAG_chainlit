import asyncio
import sys

sys.path.extend(['../','./'])

import chainlit as cl
from db_operations import Neo4jKG
from llm_processor import LLMProcessor
from dotenv import load_dotenv
import os
import sys
import tempfile
import traceback
from neo4j.exceptions import AuthError
from src_utils import setup_logger, async_error_handler, log_container_startup

import concurrent.futures

PDF_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("PDF_WORKERS", "2")))

# Set up logging first thing
logger = setup_logger(__name__)

log_container_startup()

# Then load environment variables
load_dotenv()

# Log application startup
logger.info("Application starting")


@cl.on_chat_start
async def init_kg():
    logger.info("New chat session started")
    try:
        # Create and connect Neo4j knowledge graph
        logger.info("Initializing Neo4j connection")
        kg = Neo4jKG()

        if not await kg.connect():
            error_msg = "Failed to connect to Neo4j database"
            logger.error(error_msg)
            await cl.Message(content=error_msg).send()
            return

        # Initialize the knowledge graph
        logger.info("Initializing knowledge graph")
        await kg.initialize()

        # Store in user session
        logger.info("Setting up LLM processor")
        cl.user_session.set("kg", kg)
        cl.user_session.set("llm", LLMProcessor())

        logger.info("Chat session initialized successfully")

    except AuthError as e:
        error_msg = f"Neo4j authentication error: {str(e)}"
        logger.error(error_msg)
        await cl.Message(content=error_msg).send()
        raise
    except Exception as e:
        # Get full stack trace
        stack_trace = traceback.format_exc()
        error_msg = f"Error during initialization: {str(e)}"
        logger.error(f"{error_msg}\n{stack_trace}")
        await cl.Message(content=error_msg).send()
        raise


@cl.on_message
@async_error_handler
async def process_query(message: cl.Message):
    logger.info(f"Processing new message: {message.content[:50]}...")

    # Get KG and LLM from session
    kg = cl.user_session.get("kg")
    llm = cl.user_session.get("llm")

    if not kg or not llm:
        error_msg = "Session not properly initialized. Please refresh the page."
        logger.error(error_msg)
        await cl.Message(content=error_msg).send()
        return

    # Start response message
    response = cl.Message(content="")
    await response.send()

    file_context = []
    temp_files = []  # Track temp files for proper cleanup

    try:
        # Check for attached files
        if message.elements:
            logger.info(f"Processing {len(message.elements)} attached elements")

        # Process attached files
        for element in message.elements:
            if isinstance(element, cl.File):
                logger.info(f"Processing file: {element.name}")
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        # Copy content from Chainlit's temp storage
                        with open(element.path, "rb") as src_file:
                            content = src_file.read()
                            logger.debug(f"Read {len(content)} bytes from {element.name}")
                        tmp_file.write(content)
                        tmp_path = tmp_file.name
                        logger.debug(f"Wrote content to temp file: {tmp_path}")

                    temp_files.append(tmp_path)  # Track for cleanup

                    if element.name.endswith(".pdf"):
                        logger.info(f"Ingesting PDF: {element.name}")
                        try:
                            # Add a user message to show processing status
                            await response.stream_token(f"📄 Processing PDF: {element.name}...")

                            extracted_text = await kg.ingest_pdf(tmp_path, PDF_EXECUTOR, original_filename=element.name)

                            if extracted_text.startswith("Error"):
                                # Show error message to user
                                logger.error(f"PDF processing error: {extracted_text}")
                                await response.stream_token(f"\n⚠️ {extracted_text}")
                            else:
                                # Show success message
                                logger.info(f"Successfully processed PDF ({len(extracted_text)} chars)")
                                await response.stream_token(
                                    f"\n✅ PDF {element.name} processed successfully! Generating reply...\n")
                                # file_context.append(f"PDF Content: {extracted_text[:500]}...")
                                file_context.append(f"PDF Content: {extracted_text}...")
                        except Exception as e:
                            error_msg = f"Error processing PDF: {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            await response.stream_token(f"\n❌ {error_msg}")
                    elif element.name.endswith(".json"):
                        logger.info(f"Ingesting JSON: {element.name}")
                        await kg.ingest_json(tmp_path)
                        file_context.append("JSON data imported successfully")
                        logger.info("Successfully processed JSON file")
                    else:
                        logger.warning(f"Unsupported file type: {element.name}")
                except Exception as e:
                    logger.error(f"Error processing file {element.name}: {str(e)}", exc_info=True)
                    await response.stream_token(f"Error processing file {element.name}: {str(e)}")

        # Perform knowledge graph retrieval
        logger.info(f"Performing augmented retrieval for query: {message.content[:50]}...")
        kg_results = await kg.augmented_retrieval(message.content)

        # Log retrieval results
        if kg_results:
            total_results = len(kg_results.get("entities", [])) + len(kg_results.get("chunks", []))
            logger.info(
                f"Found {total_results} knowledge graph results (Entities: {len(kg_results.get('entities', []))}, "
                f"Chunks: {len(kg_results.get('chunks', []))})")
            logger.debug(f"KG results: {kg_results}")
        else:
            logger.info("No knowledge graph results found")

        structured_context = {
            "entities": kg_results.get("entities", []), "chunks": kg_results.get("chunks", []), "files": file_context
            }

        if not structured_context:
            structured_context = "No relevant knowledge found, using general knowledge"
            logger.warning("No context available for LLM response")
        # Generate LLM response
        logger.info("Generating LLM response")
        try:
            stream = llm.generate_response(query=message.content, context=structured_context)
            logger.info(f"LLM response generated successfully.")
            async for token in stream:
                await response.stream_token(token)

        except asyncio.TimeoutError:
            await response.stream_token("\n\n⚠️ Response timed out")

        except Exception as e:
            logger.error(f"LLM response generation failed: {str(e)}", exc_info=True)
            await response.stream_token(f"Error generating response: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        await response.stream_token(f"Error: {str(e)}")
    finally:
        # Clean up all temp files
        logger.debug(f"Cleaning up {len(temp_files)} temporary files")
        for tmp_path in temp_files:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    logger.debug(f"Removed temp file: {tmp_path}")
            except Exception as e:
                logger.error(f"Failed to clean up temp file {tmp_path}: {str(e)}")

        await response.update()
        logger.info("Message processing complete")


@cl.on_chat_end
async def cleanup():
    logger.info("Chat session ending")
    # Clean up resources
    kg = cl.user_session.get("kg")
    if kg:
        try:
            await kg.close()
            logger.info("Neo4j connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}")


# Capture uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

if __name__ == "__main__":
    logger.info("Starting Chainlit server")
    try:
        cl.run()
    except Exception as e:
        logger.critical(f"Failed to start Chainlit server: {str(e)}", exc_info=True)
        sys.exit(1)