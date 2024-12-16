from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.memory import ConversationBufferMemory
from models.pdf_extractor import extract_text_from_pdf
from dotenv import load_dotenv
import os

# Initialize Router
router = APIRouter()

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key is missing. Set it in the .env file as GOOGLE_API_KEY.")

# Initialize Gemini Flash LLM
gemini_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Initialize memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

ats_analysis_prompt = PromptTemplate.from_template(
    """
    ### System
    You are an expert in optimizing resumes for Applicant Tracking System (ATS) compatibility, taking into account the most current industry trends, job descriptions, and the applicant's experience level. Your goal is to ensure that the resume achieves a **100% ATS score**, without including irrelevant information, fuzzy words, or buzzwords. The resume should reflect **active voice**, **active adverbs**, and **quantifiable achievements** that align with the job description and industry standards. Avoid redundant information and soft skills listed directly—integrate them within the context of experience.

    ---
    
    ### Instructions
    1. **Analyze ATS Score**:
        - Compare the resume content with the job description. Ensure strong alignment with **keywords**, **skills**, and **experience** mentioned in the JD.
        - **Assign an ATS Score out of 100**, based on the resume's alignment with job description keywords and the industry trends for the role.
        - Do not list irrelevant or soft skills explicitly. Soft skills should be **woven into** the experience and achievements sections where they naturally fit.

    2. **Optimizing the Resume**:
        - **Job-Specific Keywords**: Extract the **relevant keywords** from the job description, including specific **skills**, **technologies**, **tools**, and **methodologies**, and ensure they are **strategically placed** in the resume.
        - **Skills Section**: Highlight **mandatory skills** (technical, certifications, tools) directly mentioned in the job description.
        - **Experience Section**: Emphasize **quantifiable achievements** and **results** that reflect the **core job responsibilities** in the JD. Ensure action verbs and metrics are used.
        - **Education Section**: Align the educational qualifications and any relevant certifications to the JD.
        - **Resume Formatting**: 
            - Stick to **simple, clear formatting** (standard fonts like Arial, Times New Roman, Calibri).
            - Use **bullets** for clarity and to ensure **ATS readability**.
            - Ensure **standard headings** for sections: "Experience", "Skills", "Education", "Projects", etc.
            - Avoid **images**, **non-standard fonts**, **complex tables**, or **fancy graphics**.
            - Use **clear, concise statements** with relevant **keywords** placed naturally.

    3. **ATS Score Evaluation**: 
        - Ensure the resume is **aligned with the job description**, **industry trends**, and optimized to pass ATS scans effectively.
        - The resume should be tailored to the specific role, using **active voice**, **measurable results**, and **action verbs** (e.g., managed, optimized, improved).
        - The **skills section** should be tailored to include only **relevant** and **explicitly mentioned** skills.
    
    4. **Fresher Optimization**:
        - For freshers, **highlight internships**, **academic projects**, and any **relevant coursework** or **volunteer work** that aligns with the job description.
        - Focus on **transferable skills** like problem-solving, analytical skills, and technical expertise that are relevant to the job description.

    5. **Experience under 2 Years Optimization**:
        - Focus on **professional experience** that showcases **direct relevance** to the job description.
        - Ensure achievements are clearly stated with **metrics** (e.g., “improved website traffic by 30% within 6 months”).
    
    6. **Experience over 2 Years Optimization**:
        - Focus on **leadership**, **strategic impact**, and **quantifiable achievements** in past roles.
        - Highlight **key projects** and **solutions** that directly correlate with the job requirements.
        - Ensure **skills** mentioned in the JD are properly reflected in the resume.

    7. **Professional Email and WhatsApp Message**:
        - Write a **concise and professional email** and WhatsApp message for the candidate to send to the recruiter, highlighting the candidate’s interest and qualifications.
    
    8. **Provide Updated Resume**:
        - Ensure the **updated resume** is **ATS-compliant**, well-structured, and formatted in a way that is **easy for ATS systems** to read and interpret.
        - The final resume should include **actionable metrics**, **industry-standard language**, and **clear alignment** with the job description.

    ---
    
    ### Input
    - **Previous Chat History**: {chat_history}
    - **Combined Input**: {combined_input}  *(Contains the resume, job description, and experience level.)*

    ---
    
    ### Output
    1. **ATS Score**: [Score/100]
    2. **Suggested Updates**:
       - Provide **specific updates** to each section (Summary, Skills, Experience, Education, Projects) to make the resume ATS-friendly and aligned with the JD.
    3. **Email to Recruiter**:
       Subject: [Clear subject line]
       Body: 
       ```
       Dear [Recruiter Name],
    
       I hope this email finds you well. I am writing to express my interest in the [Job Title] position at [Company Name]. After reviewing the job description, I am confident that my skills in [Key Skills] and experience with [Relevant Experience/Project] make me an excellent fit for this role. I would love to discuss how my background aligns with your team's needs.
    
       Please find my resume attached for your review. Thank you for your time and consideration. I look forward to hearing from you.
    
       Best regards,  
       [Your Name]
       ```
    4. **WhatsApp Message**:
       ```
       Hi [Recruiter Name],  
       I hope you’re doing well. I’m [Your Name], and I’m very interested in the [Job Title] role at [Company Name]. I believe my skills in [Key Skills] and experience with [Relevant Experience/Project] make me a strong candidate. I’d love to discuss my qualifications further. Looking forward to hearing from you.
       ```
    5. **Updated Resume**:
       - Provide the **final, updated resume** that incorporates all improvements, ensuring that each section is **ATS-optimized**, **aligned with the job description**, and formatted correctly for ATS parsing.
       - Avoid unnecessary or irrelevant content, focusing solely on **achievements**, **skills**, and **experiences** that match the job description.
    """
)



# ATS Analysis Endpoint
@router.post("/analyze_ats")
async def analyze_ats(pdf_file: UploadFile = File(...), job_description: str = Form(...), experience_level: str = Form(...)):
    """
    Endpoint to analyze resume ATS compatibility against a job description.
    
    - **pdf_file**: Uploaded resume in PDF format.
    - **job_description**: Job description as input.
    - **experience_level**: Experience level of the candidate (Fresher, 2 Years of Experience, More than 2 Years).
    """
    try:
        # Validate uploaded file type
        if not pdf_file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
        # Extract text from PDF
        resume_text = await extract_text_from_pdf(pdf_file)

        # Determine experience level and incorporate into combined_input
        experience = "Fresher" if experience_level.lower() == "fresher" else "2 Years of Experience" if experience_level.lower() == "2 years" else "More than 2 Years of Experience"

        # Combine resume text, job description, and experience level into one input string
        combined_input = f"""
        Resume: {resume_text}
        
        Job Description: {job_description}
        
        Experience Level: {experience}
        """

        # Define and run the ATS analysis chain
        ats_analysis_chain = LLMChain(
            llm=gemini_flash,
            prompt=ats_analysis_prompt,
            memory=memory,
            output_parser=StrOutputParser()
        )

        # Invoke the chain asynchronously
        ats_analysis_result = await ats_analysis_chain.ainvoke({"combined_input": combined_input})
        return {"analysis_result": ats_analysis_result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Clear Memory Endpoint
@router.post("/clear_memory")
async def clear_memory():
    """
    Endpoint to clear the conversation memory.
    """
    memory.clear()
    return {"message": "Memory cleared successfully."}
