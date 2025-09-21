#!/usr/bin/env python3
"""
Create sample PDF resume for testing
"""
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import os

def create_sample_resume():
    """Create a sample PDF resume"""
    filename = "sample_data/sample_resume_rajesh_kumar.pdf"
    os.makedirs("sample_data", exist_ok=True)
    
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, height-72, "RAJESH KUMAR")
    
    c.setFont("Helvetica", 14)
    c.drawString(72, height-100, "Senior Python Developer")
    
    # Contact Info
    c.setFont("Helvetica", 10)
    c.drawString(72, height-130, "Email: rajesh.kumar@email.com | Phone: +91-9876543210")
    c.drawString(72, height-145, "LinkedIn: linkedin.com/in/rajeshkumar | GitHub: github.com/rajeshkumar")
    c.drawString(72, height-160, "Location: Hyderabad, India")
    
    # Professional Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height-190, "PROFESSIONAL SUMMARY")
    
    c.setFont("Helvetica", 10)
    summary_text = [
        "Senior Python Developer with 5+ years of experience in full-stack web development.",
        "Expert in Django, Flask, PostgreSQL, and AWS cloud technologies.",
        "Strong background in RESTful API development and database optimization.",
        "Proven track record of building scalable applications serving 100K+ users."
    ]
    
    y_pos = height - 210
    for line in summary_text:
        c.drawString(72, y_pos, line)
        y_pos -= 15
    
    # Technical Skills
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height-290, "TECHNICAL SKILLS")
    
    c.setFont("Helvetica", 10)
    skills = [
        "Programming Languages: Python, JavaScript, SQL, Java",
        "Frameworks: Django, Flask, Django REST Framework, FastAPI",
        "Databases: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch",
        "Cloud & DevOps: AWS (EC2, S3, RDS, Lambda), Docker, Kubernetes, Jenkins",
        "Tools: Git, GitHub, Postman, Jira, VS Code, PyCharm",
        "Testing: Unit Testing, pytest, Integration Testing, TDD"
    ]
    
    y_pos = height - 310
    for skill in skills:
        c.drawString(72, y_pos, skill)
        y_pos -= 15
    
    # Experience
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height-420, "PROFESSIONAL EXPERIENCE")
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, height-440, "Senior Python Developer | TechSoft Solutions | 2021 - Present")
    
    c.setFont("Helvetica", 10)
    exp1 = [
        "• Developed scalable web applications using Django and PostgreSQL serving 50K+ users",
        "• Built RESTful APIs with Django REST Framework handling 100K+ daily requests",
        "• Implemented microservices architecture using Docker and Kubernetes",
        "• Optimized database queries resulting in 40% performance improvement",
        "• Led code reviews and mentored 3 junior developers",
        "• Integrated AWS services (EC2, S3, RDS) for cloud deployment"
    ]
    
    y_pos = height - 460
    for item in exp1:
        c.drawString(72, y_pos, item)
        y_pos -= 15
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, y_pos - 10, "Python Developer | InnovateTech | 2019 - 2021")
    
    c.setFont("Helvetica", 10)
    exp2 = [
        "• Developed web applications using Flask and MySQL databases",
        "• Created automated testing suites with pytest achieving 85% code coverage",
        "• Implemented Redis caching reducing API response time by 60%",
        "• Collaborated with frontend team using React for full-stack development",
        "• Worked in Agile environment with 2-week sprint cycles"
    ]
    
    y_pos -= 30
    for item in exp2:
        c.drawString(72, y_pos, item)
        y_pos -= 15
    
    # Projects
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y_pos - 20, "KEY PROJECTS")
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, y_pos - 40, "E-commerce Platform | Python, Django, PostgreSQL, AWS")
    
    c.setFont("Helvetica", 10)
    projects = [
        "• Built complete e-commerce solution with payment gateway integration",
        "• Implemented product catalog, shopping cart, and order management",
        "• Used Celery for background task processing (email notifications, reports)",
        "• Deployed on AWS with auto-scaling and load balancing",
        "",
        "Task Management System | Flask, React, MongoDB, Docker",
        "• Developed real-time task management system with WebSocket integration",
        "• Created REST APIs for task CRUD operations and user management",
        "• Containerized application using Docker for easy deployment"
    ]
    
    y_pos -= 60
    for item in projects:
        if item:
            c.drawString(72, y_pos, item)
        y_pos -= 15
    
    # Education
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y_pos - 20, "EDUCATION")
    
    c.setFont("Helvetica", 10)
    c.drawString(72, y_pos - 40, "Bachelor of Technology in Computer Science Engineering")
    c.drawString(72, y_pos - 55, "JNTU Hyderabad | 2015 - 2019 | CGPA: 8.2/10")
    
    # Certifications
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y_pos - 80, "CERTIFICATIONS")
    
    c.setFont("Helvetica", 10)
    certs = [
        "• AWS Certified Developer - Associate (2022)",
        "• Docker Certified Associate (2021)",
        "• Python Institute - PCAP Certification (2020)"
    ]
    
    y_pos -= 100
    for cert in certs:
        c.drawString(72, y_pos, cert)
        y_pos -= 15
    
    c.save()
    print(f"✅ Created sample resume: {filename}")
    return filename

if __name__ == "__main__":
    create_sample_resume()