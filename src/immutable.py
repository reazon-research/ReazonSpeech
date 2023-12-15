from pydantic import BaseModel


class ImmutableModel(BaseModel):
    class Config:
        allow_mutation = False
