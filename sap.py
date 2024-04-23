import speech_recognition as sr
from textblob import TextBlob
import nltk
from nltk import word_tokenize, pos_tag, FreqDist
from nltk.corpus import stopwords, wordnet
import language_tool_python
from pydub import AudioSegment
from gtts import gTTS
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from parrot import Parrot
import torch
from transformers import AutoTokenizer
import re 

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    print("Overall sentiment dictionary is: ", sentiment_dict)
    print("Sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    print("Sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
    print("Sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
    print("Sentence Overall Rated As", end=" ")
    if sentiment_dict['compound'] >= 0.05:
        print("Positive")
    elif sentiment_dict['compound'] <= -0.05:
        print("Negative")
    else:
        print("Neutral")

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

def tokenize_and_extract_features(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    word_freq = FreqDist(filtered_tokens)
    return tokens, pos_tags, filtered_tokens, word_freq

def check_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return matches

def provide_grammar_feedback(matches):
    if len(matches) == 0:
        print("No significant grammar issues found. Great job!")
    else:
        print("Grammar Issues:")
        for match in matches:
            # Filter out minor issues like capitalization and missing punctuation
            if 'capitalization' not in match.ruleId and 'punctuation' not in match.ruleId:
                print(f"Rule ID: {match.ruleId} | Message: {match.message} | Correction: {match.replacements}")


def calculate_speech_rate(audio_file):
    sound = AudioSegment.from_wav(audio_file)
    duration_in_seconds = len(sound) / 1000.0
    text = transcribe_audio(audio_file)
    word_count = len(word_tokenize(text))
    speech_rate = word_count / (duration_in_seconds / 60)  # Words per minute
    return speech_rate

def speech_rate_report(speech_rate):
    if speech_rate < 100:
        return "The speech rate is too slow. Try to speak a bit faster.\n\n"
    elif 100 <= speech_rate < 150:
        return "The speech rate is suitable for presentations and conversations.\n\n"
    elif 150 <= speech_rate < 160:
        return "The speech rate is suitable for audiobooks, radio hosts, and podcasters.\n\n"
    elif 160 <= speech_rate < 250:
        return "The speech rate is a bit fast. It's suitable for radio hosts and podcasters, but you might want to slow down for other contexts.\n\n"
    elif speech_rate == 250:
        return "The speech rate is suitable for auctioneers.\n\n"
    elif 250 < speech_rate <= 400:
        return "The speech rate is suitable for commentators.\n\n"
    else:
        return "The speech rate is too fast. Try to slow down.\n We need a rate of 100-160 for presentations and conversations\n\n"

def suggest_alternative_words(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    suggestions = {}
    for token in filtered_tokens:
        synsets = wordnet.synsets(token)
        if synsets:
            alternative_words = set()
            for synset in synsets:
                alternative_words.update(synset.lemma_names())
            alternative_words.discard(token)
            suggestions[token] = list(alternative_words)[:3]
    return suggestions

def paraphraser(text):
    model_tag = "prithivida/parrot_paraphraser_on_T5"
    tokenizer = AutoTokenizer.from_pretrained(model_tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parrot = Parrot(model_tag=model_tag, use_gpu=True)
    phrases = [
        (text)
    ]
    for phrase in phrases:
        print('-'*110)
        print('Input Phrase:', phrase)
        print('-'*110)
        paraphrases = parrot.augment(input_phrase=phrase)
        if paraphrases:
            sorted_sentences = sorted(paraphrases, key=lambda x: x[1])
            for paraphrase, numbers in sorted_sentences:
                print(paraphrase)

def text_to_audio(text, language='en', output_file='SPEAKER_02.wav'):
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_file)
    except Exception as e:
        print(f"Error: {e}")

def find_fillers(text):
    filler_words = ["um", "uh", "like", "you know", "so", "well"]
    filler_count = {filler: 0 for filler in filler_words}

    for filler in filler_words:
        filler_count[filler] += len(re.findall(r'\b' + re.escape(filler) + r'\b', text, flags=re.IGNORECASE))

    return filler_count        

# Add more functions for other features as needed

if __name__ == "__main__":
    audio_file_path = "SPEAKER_01.wav"
    transcribed_text = transcribe_audio(audio_file_path)
    print(f"Transcribed Text: {transcribed_text}")

    sentiment_result = sentiment_scores(transcribed_text)

    tokens, pos_tags, filtered_tokens, word_freq = tokenize_and_extract_features(transcribed_text)

    grammar_matches = check_grammar(transcribed_text)
    provide_grammar_feedback(grammar_matches)

    speech_rate = calculate_speech_rate(audio_file_path)
    print(f"Speech Rate: {speech_rate} words per minute")
    print(speech_rate_report(speech_rate))

    print(f"Alternate sentences\n")
    paraphraser(transcribed_text)
    word_suggestions = suggest_alternative_words(transcribed_text)
    print("Word Suggestions:")
    for original_word, alternatives in word_suggestions.items():
        print(f"Original Word: {original_word} | Alternatives: {', '.join(alternatives)}")
    filler_count = find_fillers(transcribed_text)
    print("Filler Words Count:")
    for filler, count in filler_count.items():
        print(f"{filler}: {count}")    

    text_to_audio(transcribed_text, "en", "SPEAKER_02.wav")
