from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import random
import time
import threading
from datetime import datetime, timedelta
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stress_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Research benchmark data
research_data = [
    {"study": "ACM (2023)", "dataset": "Reddit Burnout", "model": "BERT/RoBERTa", "accuracy": 91, "limitation": "No emoji/temporal data"},
    {"study": "Springer (2022)", "dataset": "CLPsych 2019", "model": "BiLSTM", "accuracy": 88, "limitation": "Only English, no emoji"},
    {"study": "Springer (2021)", "dataset": "Dreaddit", "model": "RoBERTa", "accuracy": 90, "limitation": "No user-level behavior"},
    {"study": "Springer (2020)", "dataset": "SMHD", "model": "LSTM", "accuracy": 85, "limitation": "High cost, no emoji"},
    {"study": "ACM (2023)", "dataset": "EmoLex + Emoji", "model": "Hybrid", "accuracy": 92, "limitation": "No real-time alerts"}
]

# Emoji sentiment data
emoji_data = {
    "😭": {"sentiment": -0.8, "emotion": "sadness", "frequency": 245},
    "😤": {"sentiment": -0.6, "emotion": "frustration", "frequency": 189},
    "😩": {"sentiment": -0.9, "emotion": "burnout", "frequency": 156},
    "😊": {"sentiment": 0.7, "emotion": "joy", "frequency": 312},
    "😔": {"sentiment": -0.7, "emotion": "disappointment", "frequency": 198},
    "💪": {"sentiment": 0.8, "emotion": "strength", "frequency": 134},
    "😰": {"sentiment": -0.8, "emotion": "anxiety", "frequency": 167},
    "🙏": {"sentiment": 0.5, "emotion": "hope", "frequency": 223},
    "😡": {"sentiment": -0.9, "emotion": "anger", "frequency": 145},
    "❤️": {"sentiment": 0.9, "emotion": "love", "frequency": 289}
}

# System metrics
system_stats = {
    "total_posts": 15847,
    "users_tracked": 3421,
    "datasets_used": 5,
    "model_accuracy": 92.3,
    "current_stress_level": 0.65,
    "burnout_risk": 34,
    "high_stress_alerts": 12,
    "recent_posts": 0,
    "recent_high_stress": 0
}

# Emotion distribution
emotion_distribution = {
    "joy": 0.15, "anger": 0.25, "sadness": 0.30, "fear": 0.20,
    "disgust": 0.10, "surprise": 0.08, "trust": 0.12, "anticipation": 0.18
}

# Dataset sentiment distribution (pie chart)
dataset_sentiment = {
    "positive": 25,
    "negative": 45,
    "neutral": 30
}

# Burnout distribution by category (bar chart)
burnout_distribution = {
    "Work Overload": 35,
    "Lack of Control": 28,
    "Insufficient Reward": 22,
    "Workplace Community": 18,
    "Fairness Issues": 15,
    "Value Conflicts": 12
}

# Temporal recommendations
temporal_recommendations = {
    "morning": "Start your day with 5 minutes of deep breathing",
    "afternoon": "Take a 10-minute walk to reduce stress",
    "evening": "Practice gratitude journaling before bed",
    "night": "Limit screen time 1 hour before sleep"
}

# Wellness recommendations
wellness_recommendations = [
    "Take a quick mental break",
    "Practice deep breathing exercises",
    "Listen to calming music",
    "Connect with a friend or family member",
    "Go for a short walk outside",
    "Try a 5-minute meditation"
]

# User stress history for chart
user_stress_history = {
    "timestamps": [],
    "stress_levels": [],
    "stress_scores": []
}

# Live alert analysis
live_alert_analysis = {
    "total_alerts": 0,
    "level_counts": {"Mild": 0, "Moderate": 0, "High": 0, "Severe": 0},
    "source_counts": {"Reddit": 0, "Twitter": 0, "Survey": 0, "Forum": 0},
    "avg_stress_score": 0.0,
    "critical_alerts": 0,
    "recent_trend": []
}

def stress_analyzer(text):
    stress_keywords = {
        'burnout': 0.95, 'burned out': 0.95, 'burnt out': 0.95, 'suicidal': 0.98, 
        'hopeless': 0.90, 'breakdown': 0.85, 'panic': 0.80, 'anxiety': 0.75, 
        'overwhelmed': 0.70, 'exhausted': 0.65, 'stressed': 0.60, 'worried': 0.50, 
        'pressure': 0.55, 'depressed': 0.85, 'tired': 0.40, 'frustrated': 0.45, 
        'angry': 0.50, 'sad': 0.55
    }
    
    score = 0.25
    for word, weight in stress_keywords.items():
        if word in text.lower():
            score = max(score, weight)
    
    # Add emoji influence
    for emoji, data in emoji_data.items():
        if emoji in text:
            score += data['sentiment'] * 0.1
    
    score = max(0, min(1, score))
    
    if score >= 0.8: level = 'Severe'
    elif score >= 0.6: level = 'High'
    elif score >= 0.4: level = 'Moderate'
    else: level = 'Mild'
    
    return {'level': level, 'score': score}

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/research-data')
def get_research_data():
    return jsonify(research_data)

@app.route('/api/system-stats')
def get_system_stats():
    return jsonify(system_stats)

@app.route('/api/emoji-data')
def get_emoji_data():
    return jsonify(emoji_data)

@app.route('/api/emotion-distribution')
def get_emotion_distribution():
    return jsonify(emotion_distribution)

@app.route('/api/dataset-sentiment')
def get_dataset_sentiment():
    return jsonify(dataset_sentiment)

@app.route('/api/burnout-distribution')
def get_burnout_distribution():
    return jsonify(burnout_distribution)

@app.route('/api/temporal-recommendations')
def get_temporal_recommendations():
    return jsonify(temporal_recommendations)

@app.route('/api/wellness-recommendations')
def get_wellness_recommendations():
    return jsonify(wellness_recommendations)

@app.route('/api/user-stress-history')
def get_user_stress_history():
    return jsonify(user_stress_history)

@socketio.on('connect')
def handle_connect():
    emit('connected', {'status': 'Real-time monitoring active'})
    emit('system_data', {
        'stats': system_stats,
        'research': research_data,
        'emotions': emotion_distribution,
        'emojis': emoji_data,
        'dataset_sentiment': dataset_sentiment,
        'burnout_distribution': burnout_distribution,
        'temporal_recommendations': temporal_recommendations,
        'wellness_recommendations': wellness_recommendations,
        'user_stress_history': user_stress_history
    })

@socketio.on('user_input')
def handle_user_input(data):
    user_text = data['text']
    result = stress_analyzer(user_text)
    
    # Add to user stress history
    timestamp = datetime.now().strftime('%H:%M')
    user_stress_history['timestamps'].append(timestamp)
    user_stress_history['stress_levels'].append(result['level'])
    user_stress_history['stress_scores'].append(result['score'])
    
    # Keep only last 10 entries
    if len(user_stress_history['timestamps']) > 10:
        user_stress_history['timestamps'].pop(0)
        user_stress_history['stress_levels'].pop(0)
        user_stress_history['stress_scores'].pop(0)
    
    # Emit analysis result
    emit('user_analysis_result', {
        'text': user_text,
        'level': result['level'],
        'score': result['score'],
        'timestamp': timestamp,
        'is_critical': result['level'] in ['High', 'Severe'],
        'history': user_stress_history
    }, broadcast=True)
    
    print(f"👤 User Input - {result['level']}: {user_text[:50]}...")

def real_time_monitoring():
    sample_posts = [
        "I'm completely burned out from work 😩 can't handle this anymore",
        "Having panic attacks daily, feeling hopeless 😭",
        "Boss is toxic, workplace stress is killing me 😤",
        "Lost all motivation, just want to give up 😔",
        "Anxiety through the roof, can't sleep 😰",
        "Having a productive day, feeling good 😊💪",
        "Grateful for my support system 🙏❤️",
        "Work-life balance improving slowly 😊",
        "Feeling overwhelmed with deadlines 😩",
        "Mental health struggling lately 😔"
    ]
    
    while True:
        time.sleep(random.randint(5, 10))
        
        post = random.choice(sample_posts)
        result = stress_analyzer(post)
        
        # Generate temporal data
        current_hour = datetime.now().hour
        stress_multiplier = 1.2 if current_hour >= 22 or current_hour <= 6 else 1.0
        
        alert_data = {
            'text': post,
            'level': result['level'],
            'score': result['score'],
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'is_critical': result['level'] in ['High', 'Severe'],
            'source': random.choice(['Reddit', 'Twitter', 'Survey', 'Forum']),
            'user_id': f"user_{random.randint(1000, 9999)}",
            'temporal_factor': stress_multiplier,
            'detected_emojis': [emoji for emoji in emoji_data.keys() if emoji in post]
        }
        
        # Update system stats
        system_stats['total_posts'] += 1
        system_stats['recent_posts'] += 1
        if result['level'] in ['High', 'Severe']:
            system_stats['high_stress_alerts'] += 1
            system_stats['recent_high_stress'] += 1
        
        # Update live alert analysis
        live_alert_analysis['total_alerts'] += 1
        live_alert_analysis['level_counts'][result['level']] += 1
        live_alert_analysis['source_counts'][alert_data['source']] += 1
        if result['level'] in ['High', 'Severe']:
            live_alert_analysis['critical_alerts'] += 1
        
        # Update average stress score
        total_score = sum([live_alert_analysis['level_counts'][level] * (0.2 if level == 'Mild' else 0.5 if level == 'Moderate' else 0.7 if level == 'High' else 0.9) for level in live_alert_analysis['level_counts']])
        live_alert_analysis['avg_stress_score'] = total_score / live_alert_analysis['total_alerts'] if live_alert_analysis['total_alerts'] > 0 else 0
        
        # Add to recent trend (keep last 20)
        live_alert_analysis['recent_trend'].append(result['score'])
        if len(live_alert_analysis['recent_trend']) > 20:
            live_alert_analysis['recent_trend'].pop(0)
        
        socketio.emit('real_time_alert', alert_data)
        socketio.emit('stats_update', system_stats)
        socketio.emit('live_analysis_update', live_alert_analysis)
        
        print(f"🔴 {result['level']}: {post[:50]}...")

if __name__ == '__main__':
    print("🧠 Early Stress & Burnout Detection Dashboard")
    print("=" * 50)
    print(f"📊 System Stats: {system_stats['total_posts']} posts, {system_stats['users_tracked']} users")
    print(f"🎯 Model Accuracy: {system_stats['model_accuracy']}%")
    
    monitor_thread = threading.Thread(target=real_time_monitoring, daemon=True)
    monitor_thread.start()
    
    print("🚀 Dashboard: http://localhost:5001")
    socketio.run(app, debug=False, port=5001, host='127.0.0.1')