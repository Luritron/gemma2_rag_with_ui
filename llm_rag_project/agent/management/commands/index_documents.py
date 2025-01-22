from django.core.management.base import BaseCommand
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

class Command(BaseCommand):
    help = "Indexes documents from a directory into the ChromaDB vector store."

    def add_arguments(self, parser):
        parser.add_argument(
            '--directory',
            type=str,
            required=True,
            help="Directory containing documents to index (e.g., './documents')."
        )

    def handle(self, *args, **options):
        directory = options['directory']
        if not os.path.exists(directory):
            self.stderr.write(self.style.ERROR(f"Directory '{directory}' does not exist."))
            return

        self.stdout.write(f"Loading documents from directory: {directory}")
        loader = DirectoryLoader(directory, glob="**/*.txt")
        documents = loader.load()
        self.stdout.write(f"Loaded {len(documents)} documents.")

        embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            add_start_index=True,
        )

        self.stdout.write("Splitting documents into chunks...")
        texts = text_splitter.split_documents(documents)

        persist_dir = "./db-hormozi"
        self.stdout.write(f"Persisting vector store to directory: {persist_dir}")
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        self.stdout.write(self.style.SUCCESS("Indexing completed successfully!"))
