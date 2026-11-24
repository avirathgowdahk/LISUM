import os
import sys
import pytest

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app as flask_app, JOBS
from models.transcript import TranscriptSegment, Sentence

@pytest.fixture(scope="session")
def client():
    """Provides Flask test client with testing configurations."""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

@pytest.fixture
def sample_segments():
    """Provides sample transcript segments representing a lecture."""
    return [
        TranscriptSegment(1, 0.0, 10.0, "Good morning everyone and welcome to the database lecture.", 0.98),
        TranscriptSegment(2, 10.5, 25.0, "A composite primary key is defined as a primary key consisting of two or more columns.", 0.95),
        TranscriptSegment(3, 25.5, 42.0, "The fundamental concept is that each individual column might contain duplicate values.", 0.92),
        TranscriptSegment(4, 42.5, 60.0, "For example, in an enrollment table, student ID and course ID together uniquely identify a record.", 0.94),
        TranscriptSegment(5, 60.5, 78.0, "In conclusion, composite keys maintain uniqueness and entity integrity across multiple related attributes.", 0.96)
    ]

@pytest.fixture
def sample_sentences():
    """Provides sample Sentence objects with educational attributes."""
    return [
        Sentence(1, 1, "Good morning everyone and welcome to the database lecture.", 0.0, 10.0, educational_category="admin_noise"),
        Sentence(2, 2, "A composite primary key is defined as a primary key consisting of two or more columns.", 10.5, 25.0, educational_category="definition"),
        Sentence(3, 3, "The fundamental concept is that each individual column might contain duplicate values.", 25.5, 42.0, educational_category="concept"),
        Sentence(4, 4, "For example, in an enrollment table, student ID and course ID together uniquely identify a record.", 42.5, 60.0, educational_category="example"),
        Sentence(5, 5, "In conclusion, composite keys maintain uniqueness and entity integrity across multiple related attributes.", 60.5, 78.0, educational_category="conclusion")
    ]
