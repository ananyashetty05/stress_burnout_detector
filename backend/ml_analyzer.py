from transformers import pipeline
import nltk
from nltk.corpus import opinion_lexicon
import re

# Initialize DistilRoBERTa emotion classifier
try:
    emotion_classifier = pipeline("text-classification", 
                                model="j-hartmann/emotion-english-distilroberta-base",
                                return_all_scores=True)
    print("✅ DistilRoBERTa loaded successfully")
except Exception as e:
    print(f"❌ Error loading DistilRoBERTa: {e}")
    emotion_classifier = None

# Enhanced NRC EmoLex with critical stress keywords
NRC_EMOTIONS = {
    'anger': ['angry', 'rage', 'furious', 'mad', 'irritated', 'annoyed'],
    'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'panic', 'attacks', 'breaking', 'cope'],
    'sadness': ['sad', 'depressed', 'miserable', 'unhappy', 'grief', 'sorrow', 'burned out', 'burnt out', 'burnout', 'exhausted', 'drained', 'hopeless', 'breaking me down', 'killing', 'mental health'],
    'joy': ['happy', 'joyful', 'cheerful', 'glad', 'pleased', 'delighted'],
    'disgust': ['disgusted', 'revolted', 'sick', 'nauseated'],
    'surprise': ['surprised', 'amazed', 'shocked', 'astonished'],
    'trust': ['trust', 'confident', 'secure', 'safe'],
    'anticipation': ['excited', 'eager', 'hopeful', 'optimistic']
}

def get_nrc_emotions(text):
    """Extract NRC EmoLex emotions from text"""
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, words in NRC_EMOTIONS.items():
        score = sum(1 for word in words if word in text_lower)
        emotion_scores[emotion] = score
    
    return emotion_scores

def get_distilroberta_emotions(text):
    """Get emotions from DistilRoBERTa transformer"""
    if not emotion_classifier:
        return {}
    
    try:
        results = emotion_classifier(text)
        emotion_scores = {}
        for result in results[0]:  # First prediction
            emotion_scores[result['label'].lower()] = result['score']
        return emotion_scores
    except Exception as e:
        print(f"Error in DistilRoBERTa: {e}")
        return {}

def analyze_emoji_sentiment(text):
    """Enhanced emoji sentiment analysis with stress category mapping"""
    emoji_sentiment_map = {
        # 🟢 Normal (0.0-0.2)
        '😀': 0.1, '😄': 0.1, '😊': 0.1, '🤗': 0.1, '😌': 0.1,
        '💪': 0.0, '🌈': 0.0, '☀️': 0.0, '🧘‍♀️': 0.0, '🎶': 0.0, '🕊️': 0.0,
        
        # 🟡 Mild Stress (0.3-0.5)
        '😅': 0.4, '😬': 0.4, '😕': 0.4, '😔': 0.4, '😞': 0.4, '🙁': 0.4,
        '☕': 0.3, '😴': 0.3, '💤': 0.3,
        
        # 🟠 High Stress (0.6-0.8)
        '😫': 0.7, '😩': 0.7, '😣': 0.7, '😢': 0.7, '😰': 0.7, '😤': 0.7,
        '😖': 0.7, '😟': 0.7, '🍷': 0.6, '🍫': 0.6,
        
        # 🔴 Burnout (0.9-1.0)
        '😭': 0.9, '😱': 0.9, '💔': 0.9, '😶‍🌫️': 0.9, '🤯': 0.9, '😵‍💫': 0.9
    }
    
    emoji_score = 0
    emoji_count = 0
    
    for emoji, sentiment in emoji_sentiment_map.items():
        if emoji in text:
            emoji_score += sentiment
            emoji_count += 1
    
    return emoji_score / emoji_count if emoji_count > 0 else 0

# Enhanced keyword scoring system with dataset-optimized terms
STRESS_KEYWORDS = {
    # 🔴 Burnout (0.65-1.0) - Emotional collapse / despair
    'burnout': 0.9, 'burned': 0.85, 'burnt out': 0.9, 'hopeless': 0.9, 'worthless': 0.85, 'done': 0.8, 'nothing matters': 0.9,
    'emotionally numb': 0.85, 'disconnected': 0.8, 'severe fatigue': 0.8, 'empty': 0.85, 'drained': 0.8,
    'can\'t continue': 0.9, 'don\'t care': 0.8, 'give up': 0.85, 'killing': 0.8, 'suicidal': 0.95, 'want to die': 0.95,
    'bottom': 0.8, 'last point': 0.8, 'terrified': 0.75, 'unconscious': 0.75, 'wrong thing': 0.7, 'hit the bottom': 0.85,
    'very last': 0.8, 'going to die': 0.9, 'become unconscious': 0.8, 'throat': 0.7, 'pressure on': 0.7,
    
    # 🟠 High Stress (0.45-0.65) - High anxiety / emotional fatigue
    'exhausted': 0.6, 'can\'t cope': 0.6, 'panic': 0.6, 'breaking down': 0.6, 'crying': 0.55,
    'emotionally drained': 0.6, 'no sleep': 0.55, 'mentally exhausted': 0.6, 'overwhelmed': 0.55,
    'too much workload': 0.55, 'anxious': 0.5, 'breaking': 0.55, 'draining': 0.6, 'need help': 0.5,
    'falling apart': 0.6, 'everything is falling apart': 0.6, 'afraid': 0.5, 'scared': 0.5, 'nervous': 0.5,
    'attacks': 0.6, 'cope': 0.5, 'depressed': 0.6, 'miserable': 0.6, 'unhappy': 0.5, 'grief': 0.6,
    
    # 🟡 Mild Stress (0.25-0.45) - Low anxiety / fatigue
    'tired': 0.35, 'a bit anxious': 0.35, 'busy': 0.3, 'can\'t focus': 0.35, 'low energy': 0.35,
    'stressed out': 0.4, 'pressure': 0.35, 'need rest': 0.35, 'worried': 0.35, 'overworked': 0.4,
    'mentally tired': 0.35, 'bit': 0.3, 'feeling a bit': 0.35, 'little bit': 0.3, 'somewhat': 0.3,
    'slightly': 0.3, 'a little': 0.3, 'kind of': 0.3, 'sort of': 0.3, 'presentation': 0.3,
    'irritated': 0.35, 'annoyed': 0.35, 'mad': 0.4, 'furious': 0.45, 'rage': 0.45,
    
    # 🟢 Normal (0.0-0.25) - Positive / Neutral
    'calm': 0.1, 'happy': 0.05, 'good day': 0.05, 'relaxed': 0.1, 'doing fine': 0.1, 'content': 0.1,
    'chill': 0.1, 'peaceful': 0.1, 'okay': 0.15, 'grateful': 0.05, 'hopeful': 0.05, 'productive': 0.05,
    'excited': 0.05, 'motivated': 0.05, 'fine': 0.15, 'good': 0.05, 'joyful': 0.05, 'cheerful': 0.05,
    'glad': 0.05, 'pleased': 0.05, 'delighted': 0.05, 'confident': 0.1, 'secure': 0.1, 'safe': 0.1,
    'eager': 0.05, 'optimistic': 0.05, 'trust': 0.1
}

def calculate_keyword_score(text):
    """Calculate stress score based on individual keywords"""
    text_lower = text.lower()
    max_score = 0.2  # Base score
    
    # Check each keyword and take the highest score
    for keyword, weight in STRESS_KEYWORDS.items():
        if keyword in text_lower:
            if weight > 0:  # Stress keywords
                max_score = max(max_score, weight)
            else:  # Positive keywords reduce score
                max_score += weight
    
    # Ensure score stays in valid range
    max_score = max(0, min(1, max_score))
    
    # Determine level based on score (lowered thresholds)
    if max_score >= 0.65: detected_level = 'Burnout'
    elif max_score >= 0.45: detected_level = 'High'
    elif max_score >= 0.25: detected_level = 'Mild'
    else: detected_level = 'Normal'
    
    return max_score, detected_level

def ml_stress_analyzer(text):
    """Advanced ML-based stress analysis with keyword training"""
    
    text_lower = text.lower()
    
    # 1. Level-specific keyword analysis
    keyword_score, keyword_level = calculate_keyword_score(text)
    
    # 2. DistilRoBERTa emotion classification
    roberta_emotions = get_distilroberta_emotions(text)
    
    # 3. NRC EmoLex emotion scoring
    nrc_emotions = get_nrc_emotions(text)
    
    # 4. Emoji sentiment analysis
    emoji_sentiment = analyze_emoji_sentiment(text)
    
    # 5. Combine all features with keyword training as primary
    stress_score = keyword_score  # Start with keyword-based score
    
    # Add DistilRoBERTa emotions (secondary)
    if roberta_emotions:
        roberta_boost = (
            roberta_emotions.get('sadness', 0) * 0.3 +
            roberta_emotions.get('anger', 0) * 0.25 +
            roberta_emotions.get('fear', 0) * 0.25 -
            roberta_emotions.get('joy', 0) * 0.1
        )
        stress_score += roberta_boost * 0.3
    
    # Add NRC emotions (tertiary)
    if nrc_emotions:
        total_negative = nrc_emotions.get('sadness', 0) + nrc_emotions.get('anger', 0) + nrc_emotions.get('fear', 0)
        total_positive = nrc_emotions.get('joy', 0) + nrc_emotions.get('trust', 0)
        nrc_boost = (total_negative - total_positive) / 5
        stress_score += nrc_boost * 0.2
    
    # Add emoji sentiment (quaternary)
    if emoji_sentiment < 0:
        stress_score += abs(emoji_sentiment) * 0.3
    
    # Normalize to 0-1 range
    stress_score = max(0, min(1, stress_score))
    
    # Use keyword-detected level as primary, adjust with ML score (lowered thresholds)
    if keyword_level == 'Burnout' or stress_score >= 0.65:
        final_level = 'Burnout'
    elif keyword_level == 'High' or stress_score >= 0.45:
        final_level = 'High'
    elif keyword_level == 'Mild' or stress_score >= 0.25:
        final_level = 'Mild'
    else:
        final_level = 'Normal'
    
    return {
        'level': final_level,
        'score': stress_score,
        'keyword_level': keyword_level,
        'keyword_score': keyword_score,
        'roberta_emotions': roberta_emotions,
        'nrc_emotions': nrc_emotions,
        'emoji_sentiment': emoji_sentiment
    }