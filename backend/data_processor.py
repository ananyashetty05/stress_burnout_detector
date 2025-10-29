import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
import os

class StressBurnoutProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = None
        self.data_stats = {}
        
    def load_and_combine_data(self):
        """Load and combine all datasets"""
        datasets = []
        
        # Load Dreaddit data
        try:
            dreaddit_train = pd.read_csv('../data/dreaddit-train.csv')
            dreaddit_test = pd.read_csv('../data/dreaddit-test.csv')
            dreaddit_combined = pd.concat([dreaddit_train, dreaddit_test])
            dreaddit_combined['text_clean'] = dreaddit_combined['text'].fillna('')
            dreaddit_combined['stress_label'] = dreaddit_combined['label']
            datasets.append(dreaddit_combined[['text_clean', 'stress_label', 'subreddit']])
            print(f"✅ Loaded Dreaddit: {len(dreaddit_combined)} records")
        except Exception as e:
            print(f"❌ Dreaddit error: {e}")
            
        # Load Reddit Combined data
        try:
            reddit_data = pd.read_csv('../data/Reddit_Combi.csv', delimiter=';')
            reddit_data['text_clean'] = (reddit_data['title'].fillna('') + ' ' + reddit_data['body'].fillna('')).str.strip()
            reddit_data['stress_label'] = reddit_data['label']
            reddit_data['subreddit'] = 'reddit_combined'
            datasets.append(reddit_data[['text_clean', 'stress_label', 'subreddit']])
            print(f"✅ Loaded Reddit Combined: {len(reddit_data)} records")
        except Exception as e:
            print(f"❌ Reddit Combined error: {e}")
            
        # Load Twitter data
        try:
            twitter_data = pd.read_csv('../data/Twitter_Full.csv')
            if 'text' in twitter_data.columns:
                twitter_data['text_clean'] = twitter_data['text'].fillna('')
                twitter_data['stress_label'] = twitter_data.get('label', 0)  # Default to 0 if no label
                twitter_data['subreddit'] = 'twitter'
                datasets.append(twitter_data[['text_clean', 'stress_label', 'subreddit']])
                print(f"✅ Loaded Twitter: {len(twitter_data)} records")
        except Exception as e:
            print(f"❌ Twitter error: {e}")
            
        if not datasets:
            raise Exception("No datasets could be loaded!")
            
        # Combine all datasets
        combined_data = pd.concat(datasets, ignore_index=True)
        combined_data = combined_data.dropna(subset=['text_clean'])
        combined_data = combined_data[combined_data['text_clean'].str.len() > 10]
        
        print(f"🔄 Total combined records: {len(combined_data)}")
        return combined_data
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
        
    def extract_stress_features(self, data):
        """Extract stress-related features from text"""
        stress_keywords = {
            'severe': ['suicide', 'kill myself', 'end it all', 'hopeless', 'worthless'],
            'high': ['burnout', 'breakdown', 'panic', 'overwhelmed', 'exhausted'],
            'moderate': ['stressed', 'anxious', 'worried', 'pressure', 'tired'],
            'mild': ['concerned', 'uneasy', 'tense', 'bothered', 'frustrated']
        }
        
        features = []
        for text in data['text_clean']:
            text_lower = str(text).lower()
            feature_vector = []
            
            # Keyword counts
            for level, keywords in stress_keywords.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                feature_vector.append(count)
                
            # Text length features
            feature_vector.extend([
                len(str(text).split()),  # Word count
                len([w for w in str(text).split() if len(w) > 6]),  # Long words
                text_lower.count('!'),  # Exclamation marks
                text_lower.count('?'),  # Question marks
            ])
            
            features.append(feature_vector)
            
        return np.array(features)
        
    def train_model(self, data):
        """Train ensemble model on the data"""
        print("🔄 Preprocessing text...")
        data['text_processed'] = data['text_clean'].apply(self.preprocess_text)
        
        # Create TF-IDF features
        print("🔄 Creating TF-IDF features...")
        tfidf_features = self.vectorizer.fit_transform(data['text_processed'])
        
        # Extract additional features
        print("🔄 Extracting stress features...")
        stress_features = self.extract_stress_features(data)
        
        # Combine features
        from scipy.sparse import hstack
        X = hstack([tfidf_features, stress_features])
        y = data['stress_label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create ensemble model
        print("🔄 Training ensemble model...")
        models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('svm', SVC(probability=True, random_state=42))
        ]
        
        self.model = VotingClassifier(models, voting='soft')
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained! Accuracy: {accuracy:.3f}")
        
        # Store statistics
        self.data_stats = {
            'total_records': len(data),
            'stress_distribution': data['stress_label'].value_counts().to_dict(),
            'subreddit_distribution': data['subreddit'].value_counts().to_dict(),
            'accuracy': accuracy,
            'avg_text_length': data['text_clean'].str.len().mean(),
            'datasets_used': data['subreddit'].nunique()
        }
        
        return accuracy
        
    def predict_stress(self, text):
        """Predict stress level for new text"""
        if not self.model:
            return {'level': 'Unknown', 'score': 0.5, 'confidence': 0.5}
            
        processed_text = self.preprocess_text(text)
        tfidf_features = self.vectorizer.transform([processed_text])
        
        # Extract stress features for single text
        stress_features = self.extract_stress_features(pd.DataFrame({'text_clean': [text]}))
        
        from scipy.sparse import hstack
        X = hstack([tfidf_features, stress_features])
        
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        level_map = {0: 'Normal', 1: 'Stressed'}
        return {
            'level': level_map.get(prediction, 'Unknown'),
            'score': probabilities[1] if len(probabilities) > 1 else 0.5,
            'confidence': confidence
        }
        
    def save_model(self):
        """Save trained model and vectorizer"""
        with open('stress_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open('data_stats.pkl', 'wb') as f:
            pickle.dump(self.data_stats, f)
        print("✅ Model saved successfully!")
        
    def load_model(self):
        """Load pre-trained model"""
        try:
            with open('stress_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open('data_stats.pkl', 'rb') as f:
                self.data_stats = pickle.load(f)
            print("✅ Model loaded successfully!")
            return True
        except:
            print("❌ No pre-trained model found")
            return False

if __name__ == "__main__":
    processor = StressBurnoutProcessor()
    
    # Try to load existing model first
    if not processor.load_model():
        print("🔄 Training new model...")
        data = processor.load_and_combine_data()
        processor.train_model(data)
        processor.save_model()
    
    print("📊 Data Statistics:")
    for key, value in processor.data_stats.items():
        print(f"  {key}: {value}")
        
    # Test prediction
    test_text = "I'm completely overwhelmed with work and having panic attacks daily"
    result = processor.predict_stress(test_text)
    print(f"\n🧪 Test prediction: {result}")