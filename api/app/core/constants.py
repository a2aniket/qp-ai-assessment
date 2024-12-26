# API related constants
EMBEDDING_SUCCESS_MSG = "Embedding created successfully"
CHAT_RESPONSE_MSG = "Chat response generated"

# Error messages
INVALID_FILE_TYPE_ERROR = "Invalid file type. Only PDF and Word documents are allowed."
EMBEDDING_ERROR = "Error occurred while creating embedding"
CHAT_ERROR = "Error occurred while generating chat response"

# File paths
LOG_DIR = "logs"
FILE_UPLOAD_DIR = "misc/fileupload/"

# Database related
SQLITE_URL_PREFIX = "sqlite:///"

# Embedding related
EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5"
EMBEDDING_PATH="misc/embeddings/"
MODEL_KWARGS = {'device': 'cpu'}
ENCODE_KWARGS = {'normalize_embeddings': True}