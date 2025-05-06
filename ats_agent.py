# Import necessaries
import torch
import pdfplumber
import os
import shutil
from transformers import AutoTokenizer, AutoModel
from sentence_transformers.util import cos_sim

# Load the tokenizer and model for embedding generation
model_id = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

class ATS:
  '''Applicant Tracking System (ATS) for matching CVs to a job description based on semantic similarity.'''

  def __init__(self, job_path, cv_path):
    '''
    Initialize ATS instance with job description and CV file paths.

    Args:
      job_path (str): Path to the job description text file.
      cv_path (str): Path to the applicant's resume in PDF format.
    '''
    self.job = job_path
    self.cv = cv_path

  def Read_resume(self) -> str:
    '''
    Reads the resume PDF and extracts text content.

    Returns:
      str: Full text extracted from the PDF resume.
    '''
    with pdfplumber.open(self.cv) as pdf_file:
      resume = str()
      for page in pdf_file.pages:
        resume += page.extract_text()
      return resume

  def Read_job(self) -> str:
    '''
    Reads the job description from a text file.

    Returns:
      str: Content of the job description file.
    '''
    with open(self.job, 'r') as text_file:
      return text_file.read()

  def get_embedding(self, text):
    '''
    Generates a sentence embedding for the given text using the pretrained model.

    Args:
      text (str): Input text to embed.

    Returns:
      torch.Tensor: Mean pooled embedding tensor.
    '''
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

  def Score(self) -> int:
    '''
    Computes similarity score between resume and job description.

    Returns:
      int: Similarity score (0–100) based on cosine similarity.
    '''
    resume_embedding = self.get_embedding(self.Read_resume())
    job_embedding = self.get_embedding(self.Read_job())
    similarity = cos_sim(resume_embedding, job_embedding).item()
    return int(similarity * 100)

def Select_applicants(job_path, cvs_path,threshold) -> None:
  '''
  Processes all resumes in a folder and selects those with similarity above a given threshold.

  Args:
    job_path (str): Path to job description file.
    cvs_path (str): Path to folder containing all applicant CVs.
    threshold (int): Minimum similarity score threshold for selection.
  '''
  for cv in os.listdir(cvs_path):
    try:
      My_ats = ATS(job_path, f'{cvs_path}/{cv}')
      print(f'{cv} is under processing...\n')
      if My_ats.Score() >= threshold:
        shutil.copyfile(f'{cvs_path}/{cv}', f'/content/Selected applicants/{cv}')
      print(f'{"-"*100}')
    except:
      print(f'Something went wrong with {cv}, so ignoring!')
  print(f'{len(os.listdir("/content/Selected applicants"))} Cvs are stored in Selected applicants folder')

# Run the selection process
Select_applicants('/content/Job.txt', '/content/CVs', 70)