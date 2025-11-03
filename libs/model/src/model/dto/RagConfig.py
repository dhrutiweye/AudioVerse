from dataclasses import dataclass
@dataclass
class RagConfig:
    EMBED_MODEL: str
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    COLLECTION_NAME: str
    CONTENT_DENSITY: float
    CHAR_LENGTH: int
    DEVICE: str = "cpu"

    def to_dict(self):
        return vars(self)