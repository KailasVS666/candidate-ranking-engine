"""
scripts/generate_sample_data.py
---------------------------------
Generates realistic synthetic resumes in data/sample_resumes/ so you can
test the system immediately without collecting real CVs.

Run with:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_RESUMES = {
    "alice_chen_resume.txt": """
Alice Chen
alice.chen@email.com | linkedin.com/in/alicech | San Francisco, CA

SUMMARY
Senior Data Scientist with 7 years of experience building production ML systems.
Expert in Python, TensorFlow, PyTorch, and cloud-native ML pipelines on AWS and GCP.

SKILLS
Python, SQL, PostgreSQL, Machine Learning, Deep Learning, TensorFlow, PyTorch,
scikit-learn, pandas, NumPy, NLP, BERT, Hugging Face, Transformers,
Docker, Kubernetes, AWS, GCP, Spark, Airflow, MLOps, A/B Testing,
Feature Engineering, Statistics, Data Visualization, Tableau

EXPERIENCE
Senior Data Scientist — TechCorp AI (2021–Present)
- Built end-to-end ML pipelines using Airflow + Spark processing 50M records/day
- Deployed BERT-based NLP models on Kubernetes (AWS EKS), reducing latency by 40%
- Led A/B testing framework adopted by 5 product teams
- Mentored 3 junior data scientists

Data Scientist — DataLabs Inc (2019–2021)
- Developed customer churn models (XGBoost) achieving 92% AUC
- Built real-time recommendation engine using collaborative filtering + PostgreSQL
- Automated ETL pipelines with Airflow on GCP

EDUCATION
M.S. Computer Science (Machine Learning) — Stanford University, 2019
B.S. Statistics — UC Berkeley, 2017

CERTIFICATIONS
AWS Certified Machine Learning – Specialty
Google Professional Data Engineer
""",

    "bob_martinez_resume.txt": """
Bob Martinez
bob.martinez@email.com | github.com/bobml | Austin, TX

SUMMARY
Data Scientist with 4 years of experience in Python and scikit-learn.
Strong background in statistical modelling and SQL.

SKILLS
Python, SQL, MySQL, scikit-learn, pandas, NumPy, Matplotlib, Seaborn,
Machine Learning, Regression, Classification, Clustering, Statistics,
Flask, REST API, Git, Linux, Bash, Excel

EXPERIENCE
Data Scientist — RetailAI (2022–Present)
- Built price optimisation models using gradient boosting (scikit-learn)
- Designed ETL pipelines in Python + MySQL, reducing report time by 60%
- Created dashboards in Tableau for executive reporting

Junior Data Analyst — Analytics Co (2020–2022)
- Performed exploratory data analysis on customer datasets (pandas, SQL)
- Automated weekly reports with Python scripts
- Supported A/B test analysis for marketing campaigns

EDUCATION
B.S. Mathematics — University of Texas, Austin, 2020
""",

    "carol_kim_resume.txt": """
Carol Kim
carol.kim@email.com | Seoul, South Korea

SUMMARY
Machine Learning Engineer with 6 years specialising in computer vision and NLP.
Proficient in PyTorch, Hugging Face Transformers, and MLOps on Azure.

SKILLS
Python, PyTorch, TensorFlow, Keras, Hugging Face, Transformers, BERT, GPT,
spaCy, NLTK, NLP, Computer Vision, OpenCV, scikit-learn, pandas, NumPy,
Docker, Kubernetes, Azure, MLOps, Prometheus, Grafana, PostgreSQL, SQL,
CI/CD, GitHub Actions, Unit Testing, PyTest

EXPERIENCE
ML Engineer — NLP Solutions (2020–Present)
- Fine-tuned BERT and GPT models for document classification (99% accuracy)
- Built real-time NLP inference service with FastAPI + Docker on Azure
- Implemented model monitoring with Prometheus and Grafana

Research Engineer — Computer Vision Lab (2018–2020)
- Developed object detection pipeline with OpenCV + PyTorch
- Published 2 papers on image segmentation at CVPR workshops

EDUCATION
M.S. Artificial Intelligence — KAIST, 2018
B.S. Computer Science — Seoul National University, 2016
""",

    "dave_wilson_resume.txt": """
Dave Wilson
dave.wilson@email.com | Chicago, IL

SUMMARY
Software Engineer transitioning into Data Science. Completed ML bootcamp.
Comfortable with Python basics and SQL. Eager to learn.

SKILLS
Python, SQL, Excel, Jupyter Notebook, pandas (basic), scikit-learn (basic),
HTML, CSS, JavaScript, React, Node.js, Git

EXPERIENCE
Software Engineer — WebDev Agency (2021–Present)
- Built responsive web applications using React and Node.js
- Designed REST APIs with Express and MySQL
- Collaborated in Agile sprints using Jira and Confluence

EDUCATION
B.S. Computer Science — DePaul University, 2021
Completed: Coursera Machine Learning Specialisation (Andrew Ng), 2023
""",

    "eve_johnson_resume.txt": """
Eve Johnson
eve.johnson@email.com | New York, NY

SUMMARY
Data Engineer & ML practitioner with 5 years building large-scale data pipelines.
Expert in Spark, Kafka, Airflow, and cloud-native architectures on AWS.

SKILLS
Python, SQL, PostgreSQL, MongoDB, Redis, Apache Spark, Kafka, Airflow, dbt,
AWS (S3, EMR, Glue, Lambda, SageMaker), GCP (BigQuery, Dataflow),
Docker, Kubernetes, Terraform, CI/CD, Jenkins, Machine Learning,
scikit-learn, pandas, NumPy, Data Engineering, ETL, Data Quality,
Feature Engineering, Feature Store, Statistics

EXPERIENCE
Senior Data Engineer — FinTech Corp (2021–Present)
- Architected real-time streaming pipelines with Kafka + Spark (5B events/day)
- Built Feature Store on AWS SageMaker used by 10 ML teams
- Reduced cloud costs by 35% through query optimisation (dbt + Redshift)

Data Engineer — CloudData Inc (2019–2021)
- Migrated on-prem Hadoop cluster to AWS EMR, cutting latency by 50%
- Built Airflow DAGs for automated ML retraining pipelines

EDUCATION
B.S. Computer Engineering — Columbia University, 2019
""",

    "frank_lee_resume.txt": """
Frank Lee
frank.lee@email.com | Seattle, WA

SUMMARY
Research Scientist with PhD in Statistics, specialising in Bayesian methods,
causal inference, and experimental design (A/B testing).

SKILLS
Python, R, SQL, Statistics, Bayesian Statistics, Causal Inference,
A/B Testing, Hypothesis Testing, Regression, Mixed Models, Time Series,
scikit-learn, Stan, PyMC, pandas, NumPy, Matplotlib, Seaborn, Jupyter,
Machine Learning, Deep Learning, TensorFlow (basic), Communication, Leadership

EXPERIENCE
Research Scientist — Amazon Science (2020–Present)
- Led experimentation platform serving 500K A/B tests/year
- Developed causal ML models for ad attribution (Python + Stan)
- Published 4 papers on causal inference at NeurIPS and ICML

Statistician — Biotech Corp (2018–2020)
- Designed clinical trial statistical analyses (R, mixed models)
- Automated reporting pipelines reducing turnaround from 5 days to 4 hours

EDUCATION
Ph.D. Statistics — University of Washington, 2018
B.S. Mathematics — MIT, 2013
""",

    "grace_taylor_resume.txt": """
Grace Taylor
grace.taylor@email.com | Boston, MA

SUMMARY
Entry-level Data Analyst with 1 year of experience.
Passionate about data storytelling and visualisation.

SKILLS
Python, SQL, Excel, Tableau, Power BI, pandas, Matplotlib, Seaborn,
Google Analytics, Communication, Teamwork, Problem Solving

EXPERIENCE
Data Analyst Intern — Marketing Agency (2024–Present)
- Created weekly dashboards in Tableau for campaign performance
- Wrote SQL queries for customer segmentation analysis
- Assisted with A/B test reporting for email campaigns

EDUCATION
B.S. Information Systems — Northeastern University, 2024
""",

    "henry_patel_resume.txt": """
Henry Patel
henry.patel@email.com | London, UK

SUMMARY
MLOps Engineer with 4 years of experience automating ML workflows,
CI/CD pipelines, and production model monitoring.

SKILLS
Python, SQL, Docker, Kubernetes, Terraform, Ansible, Jenkins, GitHub Actions,
CI/CD, MLOps, Airflow, Kubeflow, MLflow, Prometheus, Grafana, Elasticsearch,
AWS, GCP, Azure, Linux, Bash, Shell Scripting, Machine Learning,
scikit-learn, TensorFlow, Model Monitoring, Drift Detection, Feature Store

EXPERIENCE
MLOps Engineer — AI Platform Team (2022–Present)
- Designed ML deployment platform serving 20 production models on Kubernetes
- Implemented automated model retraining with drift detection (Prometheus)
- Built CI/CD pipelines for model packaging and deployment (GitHub Actions)

DevOps Engineer — CloudOps Ltd (2020–2022)
- Managed Kubernetes clusters serving 99.99% uptime SLA
- Automated infrastructure provisioning with Terraform + Ansible

EDUCATION
B.Eng. Software Engineering — Imperial College London, 2020
""",
}


def generate() -> None:
    output_dir = ROOT / "data" / "sample_resumes"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in SAMPLE_RESUMES.items():
        path = output_dir / filename
        path.write_text(content.strip(), encoding="utf-8")
        print(f"  ✓ Generated: {filename}")

    print(f"\n  Generated {len(SAMPLE_RESUMES)} sample résumés in {output_dir}\n")


if __name__ == "__main__":
    print("\n🗂  Generating sample résumés …\n")
    generate()
    print("  Done! Run `python scripts/demo.py` to test the full pipeline.\n")
