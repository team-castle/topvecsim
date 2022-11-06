from aredis_om import HashModel, Field


class Paper(HashModel):
    paper_id: str = Field(index=True)
    title: str = Field(index=True, full_text_search=True)
    year: int = Field(index=True)
    authors: str
    categories: str
    abstract: str = Field(index=True, full_text_search=True)
    input: str
