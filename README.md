# AI_Powered_Agents_in_Action_Resume_Screening_with_Transformers
🤖 AI-Powered Agents in Action – Resume Screening with Transformers

In today’s world, AI-powered agents are transforming how tasks are automated. These agents are software systems that act autonomously to perceive input, process it using intelligent methods (like machine learning), and perform actions—all without direct human supervision. They're being used in everything from chatbots and recommendation systems to healthcare diagnostics and HR automation.

🎯 I wanted to build a simple AI-powered agent that could help streamline the resume screening process. HR teams often receive hundreds of CVs for a single role. Manually reviewing them to find the best matches can be time-consuming, inconsistent, and prone to bias.

🛠️ Here’s how I did it:

I built an AI-based screening tool that reads job descriptions and applicant CVs.

It uses the BAAI/bge-large-en-v1.5 Transformer model to generate semantic embeddings for both the job description and resumes.

It computes the cosine similarity between them to evaluate how well each resume matches the job post.

If the similarity score exceeds a defined threshold, the CV is automatically selected and copied to a "Selected Applicants" folder.

All of this is done autonomously, making it a true lightweight AI agent for resume filtering.

This project blends Natural Language Processing (NLP) with real-world automation to demonstrate how AI can be applied to recruitment workflows in a scalable and intelligent way.
