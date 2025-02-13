import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample input text
text = """Rahul wakes up early every day. He goes to college in the morning and comes back at 3 pm. 
At present, Rahul is outside. He has to buy the snacks for all of us. 
John should complete the report by 5 pm today. Lisa needs to submit the assignment tomorrow."""

# ----------------------------------------
# 1️⃣ STEP 1: Simple Coreference Resolution (Manual Heuristic)
# ----------------------------------------
def resolve_coreferences(text):
    """Replaces pronouns (he/she) with the most recent subject (proper noun)."""
    sentences = sent_tokenize(text)  # Split text into sentences
    last_subject = None
    resolved_sentences = []

    for sent in sentences:
        words = word_tokenize(sent)
        pos_tags = pos_tag(words)

        # Identify the subject (Proper Noun - NNP)
        for word, tag in pos_tags:
            if tag == "NNP":  # Proper Noun (e.g., "Rahul", "John")
                last_subject = word
                break

        # Replace "he" or "she" with the last found subject
        sent = re.sub(r'\b(he|she)\b', last_subject, sent, flags=re.IGNORECASE)
        resolved_sentences.append(sent)

    return resolved_sentences

resolved_sentences = resolve_coreferences(text)  # Replace pronouns

# ----------------------------------------
# 2️⃣ STEP 2: Task Identification
# ----------------------------------------
def extract_tasks(sentences):
    """Identifies sentences that contain actionable tasks."""
    task_keywords = ["has to", "should", "needs to", "must", "required to"]
    return [sent for sent in sentences if any(keyword in sent for keyword in task_keywords)]

task_sentences = extract_tasks(resolved_sentences)  # Extract sentences containing tasks

# ----------------------------------------
# 3️⃣ STEP 3: Extract WHO & WHEN
# ----------------------------------------
def extract_details(task_sentences):
    """Extracts the responsible person (who) and deadline (when) from the task."""
    extracted_tasks = []
    time_patterns = [r"\d+\s?(am|pm)", r"today", r"tomorrow", r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"]

    for sent in task_sentences:
        words = word_tokenize(sent)
        pos_tags = pos_tag(words)

        entity, deadline = None, None

        # Extract Subject (Who is responsible)
        for word, tag in pos_tags:
            if tag in ["NNP", "NN"]:  # Look for Proper Nouns (Persons)
                entity = word
                break

        # Extract Deadline using regex
        for pattern in time_patterns:
            match = re.search(pattern, sent, re.IGNORECASE)
            if match:
                deadline = match.group(0)
                break

        extracted_tasks.append({
            "task": sent,
            "assigned_to": entity,
            "deadline": deadline
        })

    return extracted_tasks

task_details = extract_details(task_sentences)  # Extract details

# ----------------------------------------
# 4️⃣ STEP 4: Categorization (TF-IDF)
# ----------------------------------------
def categorize_tasks(tasks):
    """Categorizes tasks based on TF-IDF (Extracting most important words)."""
    vectorizer = TfidfVectorizer()
    task_texts = [task["task"] for task in tasks]
    
    X = vectorizer.fit_transform(task_texts)
    feature_names = vectorizer.get_feature_names_out()

    categorized = defaultdict(list)
    for i, task in enumerate(tasks):
        category = feature_names[X[i].argmax()]  # Most important word as category
        categorized[category].append(task)

    return categorized

categorized_tasks = categorize_tasks(task_details)

# ----------------------------------------
# Output Results
# ----------------------------------------
print("📌 Extracted Tasks:")
for task in task_details:
    print(task)

print("\n📌 Categorized Tasks:")
for category, tasks in categorized_tasks.items():
    print(f"\nCategory: {category}")
    for t in tasks:
        print(t)