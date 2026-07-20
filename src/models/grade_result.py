"""
Grade result model.
"""

from pydantic import BaseModel, Field


class GradeResult(BaseModel):
    """Model for document relevance grading."""

    binary_score: str = Field(
        description="Relevance score: 'yes' or 'no'"
    )
