import os
import datetime
import time
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

# Get database URL from environment variable
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create SQLAlchemy engine with better connection pool settings
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Check if connection is valid before using it
    pool_recycle=3600,   # Recycle connections after 1 hour
    connect_args={"connect_timeout": 10}  # Connection timeout of 10 seconds
)

# Create base class for models
Base = declarative_base()

# Define model for classification results
class ClassificationResult(Base):
    __tablename__ = 'classification_results'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    filename = Column(String(255), nullable=True)
    is_sample_data = Column(Integer, default=0)  # 0 = file upload, 1 = sample data
    classification = Column(String(50), nullable=False)  # "drone" or "bird"
    confidence = Column(Float, nullable=False)
    notes = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<ClassificationResult(id={self.id}, classification={self.classification}, confidence={self.confidence})>"

# Create all tables with retries
max_retries = 3
retry_delay = 2  # seconds

for attempt in range(max_retries):
    try:
        Base.metadata.create_all(engine)
        print("Database tables created successfully")
        break
    except OperationalError as e:
        if attempt < max_retries - 1:
            print(f"Database connection failed, retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(retry_delay)
        else:
            print(f"Failed to connect to database after {max_retries} attempts: {str(e)}")
            # We'll continue anyway and handle errors at the function level

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to get a database session
def get_db_session():
    session = SessionLocal()
    try:
        return session
    finally:
        session.close()

# Function to save classification result
def save_classification_result(filename, is_sample_data, classification, confidence, notes=None):
    session = SessionLocal()
    try:
        # Convert numpy types to Python native types if needed
        if hasattr(classification, 'item'):
            classification = classification.item()
        if hasattr(confidence, 'item'):
            confidence = confidence.item()
        
        result = ClassificationResult(
            filename=filename,
            is_sample_data=1 if is_sample_data else 0,
            classification=str(classification),
            confidence=float(confidence),
            notes=notes
        )
        session.add(result)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error saving classification result: {str(e)}")
        return False
    finally:
        session.close()

# Function to get all classification results with retry
def get_all_classification_results():
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        session = SessionLocal()
        try:
            results = session.query(ClassificationResult).order_by(ClassificationResult.timestamp.desc()).all()
            return results
        except OperationalError as e:
            if attempt < max_retries - 1:
                print(f"Database connection failed, retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Failed to retrieve classification results after {max_retries} attempts: {str(e)}")
                return []  # Return empty list to avoid app crashes
        except Exception as e:
            print(f"Error retrieving classification results: {str(e)}")
            return []
        finally:
            session.close()

# Function to get recent classification results (limit by count) with retry
def get_recent_classification_results(limit=10):
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        session = SessionLocal()
        try:
            results = session.query(ClassificationResult).order_by(ClassificationResult.timestamp.desc()).limit(limit).all()
            return results
        except OperationalError as e:
            if attempt < max_retries - 1:
                print(f"Database connection failed, retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Failed to retrieve recent classification results after {max_retries} attempts: {str(e)}")
                return []  # Return empty list to avoid app crashes
        except Exception as e:
            print(f"Error retrieving recent classification results: {str(e)}")
            return []
        finally:
            session.close()
        
# Function to clear all classification history
def clear_classification_history():
    session = SessionLocal()
    try:
        # Delete all records
        session.query(ClassificationResult).delete()
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error clearing classification history: {str(e)}")
        return False
    finally:
        session.close()
