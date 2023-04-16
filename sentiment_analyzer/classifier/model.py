import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

from sentiment_analyzer.classifier.sentiment_classifier import BertCNN
from sentiment_analyzer.classifier.preprocessor import Preprocessor

pretrained_model_name = 'bert-base-cased'
best_model_state = "../BertCNN_bestModel.bin"
class_names = ['negative', 'neutral', 'positive']
MAX_LEN = 160

class Model:
    def __init__(self):

        self.device = torch.device("cpu")

        self.preprocessor = Preprocessor()

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        classifier = BertCNN(len(class_names))
        classifier.load_state_dict(
            torch.load(best_model_state, map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        preprocessed_text = self.preprocessor.process_tweet(text)
        encoded_text = self.tokenizer.encode_plus(
            preprocessed_text,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.classifier(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            predicted_class = predicted_class.cpu().item()
            probabilities = probabilities.flatten().cpu().numpy().tolist()
        return (
            class_names[predicted_class],
            confidence,
            dict(zip(class_names, probabilities)),
        )


model = Model()


def get_model():
    return model